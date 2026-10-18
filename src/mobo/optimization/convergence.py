"""优化过程只读观测与任务级曲线落盘"""

from contextlib import contextmanager
from contextvars import ContextVar
from collections import deque
import json
from pathlib import Path

import numpy as np

_RECORDER = ContextVar("optimization_convergence", default=None)


def calculate_change_rate(values):
    """计算收敛窗口的相对变化率。"""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("values不能为空")
    scale = max(float(np.max(np.abs(values))), np.finfo(float).eps)
    return float((np.max(values) - np.min(values)) / scale)


def empty_curve(mode, objectives):
    if mode is None:
        direction, x_label, metric = None, None, None
    elif mode == "reinforcement_learning":
        direction, x_label, metric = "max", "episode", "episode_reward_mean_100"
    elif mode == "single":
        direction, x_label, metric = "min", "generation", "best_feasible_weighted_objective"
    else:
        direction, x_label, metric = "max", "generation", "hypervolume"
    return {
        "mode": mode, "metric": metric, "x_label": x_label, "x": [],
        "y": [], "direction": direction, "hv_config": None,
        "early_stopping": None,
    }


def read_curve(path, mode, objectives):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data.get("y"), list):
        return data
    # 旧版多目标逐目标最优值无法恢复Pareto前沿 不能伪造HV
    curve = empty_curve(mode, objectives)
    key = "weighted_objective" if mode == "single" else "mean_episode_reward"
    if mode in {"single", "reinforcement_learning"}:
        curve["y"] = data.get("y", {}).get(key, [])
        curve["x"] = (list(range(1, len(curve["y"]) + 1))
                      if mode == "reinforcement_learning" else data.get("x", []))
    return curve


class CurveRecorder:
    def __init__(self, path, mode, objectives):
        self.path = Path(path)
        self.data = empty_curve(mode, objectives)
        self.save()

    def append(self, x, value):
        self.data["x"].append(int(x))
        self.data["y"].append(float(value) if np.isfinite(value) else None)
        self.save()

    def save(self):
        # 原子替换使运行中的查询不会读到半截JSON
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(self.path)


class EarlyStoppingMonitor:
    """Stop when the relative spread of the trailing convergence window is small."""

    def __init__(self, request, recorder):
        config = request.get("optimizer_config") or {}
        self.enabled = bool(config.get("early_stop_enabled", True))
        self.threshold = float(config.get("early_stop_threshold", 0.05))
        self.window = int(config.get("early_stop_window", 10))
        self.min_points = int(config.get("early_stop_min_points", 20))
        self.values = deque(maxlen=self.window)
        self.point_count = 0
        self.stopped = False
        self.stop_point = None
        self.change_rate = None
        self.recorder = recorder
        if recorder is not None:
            recorder.data["early_stopping"] = self.as_dict()
            recorder.save()

    def as_dict(self):
        return {
            "enabled": self.enabled,
            "threshold": self.threshold,
            "window": self.window,
            "min_points": self.min_points,
            "stopped_early": self.stopped,
            "stop_point": self.stop_point,
            "change_rate": self.change_rate,
        }

    def update(self, point, value):
        self.point_count += 1
        if not self.enabled or value is None or not np.isfinite(value):
            return False
        self.values.append(float(value))
        if self.point_count < self.min_points or len(self.values) < self.window:
            return False
        self.change_rate = calculate_change_rate(self.values)
        if self.change_rate > self.threshold:
            return False
        self.stopped = True
        self.stop_point = int(point)
        if self.recorder is not None:
            self.recorder.data["early_stopping"] = self.as_dict()
            self.recorder.save()
        return True


@contextmanager
def record_convergence(path, mode, objectives):
    token = _RECORDER.set(CurveRecorder(path, mode, objectives))
    try:
        yield
    finally:
        _RECORDER.reset(token)


class HypervolumeCurve:
    def __init__(self, problem, first_values, recorder):
        from pymoo.indicators.hv import HV

        objectives = problem.objectives
        scalers = [
            getattr(item, "scaler_y", None)
            or problem.scalers[f"scaler_y_{item.y_index}"]
            for item in objectives
        ]
        signs = np.array([1 if item.minimize else -1 for item in objectives])
        self.offset = signs * np.array([float(scaler.mean_[0]) for scaler in scalers])
        self.scale = np.array([float(scaler.scale_[0]) for scaler in scalers])
        normalized = self.normalize(first_values)
        finite = normalized[np.isfinite(normalized).all(axis=1)]
        # 只在第一代确定参考点 后续固定尺度和参考点避免指标漂移
        worst = np.max(finite, axis=0) if len(finite) else np.full(len(objectives), 3.0)
        self.reference = np.maximum(worst, 3.0) + 1.0
        self.indicator = HV(ref_point=self.reference)
        if recorder is not None:
            recorder.data["hv_config"] = {
                "objective_names": [item.name for item in objectives],
                "offset": self.offset.tolist(), "scale": self.scale.tolist(),
                "reference_point": self.reference.tolist(),
            }

    def normalize(self, values):
        return (values - self.offset) / self.scale

    def value(self, values):
        normalized = self.normalize(values)
        selected = normalized[np.all(normalized < self.reference, axis=1)]
        if not len(selected):
            return 0.0
        # 单输出的multi请求退化为一维区间长度 不调用仅支持多维的底层HV实现
        if len(self.reference) == 1:
            return float(self.reference[0] - np.min(selected[:, 0]))
        return float(self.indicator.do(selected))


def ga_callback(request):
    recorder = _RECORDER.get()
    mode = recorder.data["mode"] if recorder is not None else request.get("mode", "multi")
    best = None
    hypervolume = None
    monitor = EarlyStoppingMonitor(request, recorder)

    def observe(algorithm):
        nonlocal best, hypervolume
        values = np.asarray(algorithm.pop.get("F"), dtype=float)
        feasible = np.asarray(algorithm.pop.get("FEAS"), dtype=bool).reshape(-1)
        selected = values[feasible & np.isfinite(values).all(axis=1)]
        if mode == "single":
            current = float(np.min(selected)) if len(selected) else np.inf
            best = current if best is None else min(best, current)
            value = best
        else:
            if hypervolume is None:
                hypervolume = HypervolumeCurve(algorithm.problem, values, recorder)
            value = hypervolume.value(selected)
        if recorder is not None:
            recorder.append(algorithm.n_gen, value)
        if len(selected) and monitor.update(algorithm.n_gen, value):
            algorithm.termination.terminate()

    observe.early_stopping = monitor
    return observe


def rl_callback(request=None):
    recorder = _RECORDER.get()
    if recorder is None and request is None:
        return None
    monitor = EarlyStoppingMonitor(request or {}, recorder)
    from stable_baselines3.common.callbacks import BaseCallback

    class RewardCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.returns = None
            self.completed = deque(maxlen=100)
            self.episodes = 0

        def _on_step(self):
            rewards = np.asarray(self.locals["rewards"], dtype=float)
            if self.returns is None:
                self.returns = np.zeros_like(rewards)
            self.returns += rewards
            dones = np.asarray(self.locals["dones"], dtype=bool)
            for index in np.flatnonzero(dones):
                self.completed.append(float(self.returns[index]))
                self.returns[index] = 0
                self.episodes += 1
                value = float(np.mean(self.completed))
                if recorder is not None:
                    recorder.append(self.episodes, value)
                if monitor.update(self.episodes, value):
                    return False
            return True

    callback = RewardCallback()
    callback.early_stopping = monitor
    return callback
