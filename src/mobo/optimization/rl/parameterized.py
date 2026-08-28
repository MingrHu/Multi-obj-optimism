"""由 HTTP 协议参数构建通用代理模型 PPO 优化环境"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import gymnasium as gym
import joblib
import numpy as np
from gymnasium import spaces

from mobo.optimization.ga.run import _load_model


class SurrogatePPOEnv(gym.Env):
    def __init__(self, request: dict[str, Any], model_dir: str):
        super().__init__()
        self.request = request
        self.objectives = request["objective_config"]
        self.constraints = request["constraints"]
        self.indices = request["decision_var_indices"]
        bounds = request["decision_bounds"]
        self.low = np.asarray([item["lower"] for item in bounds], dtype=np.float32)
        self.high = np.asarray([item["upper"] for item in bounds], dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=self.low.shape, dtype=np.float32)
        output_names = request["all_var_list"][request["input_var_count"]:]
        target_names = list(dict.fromkeys([
            *request["objective_names"],
            *[item["target_obj"] for item in self.constraints if isinstance(item["target_obj"], str)],
        ]))
        first_target = request["objective_names"][0]
        self.scalers = joblib.load(Path(model_dir) / f"{first_target}_scalers.pkl")
        self.scaler_x = self.scalers["scaler_X"]
        self.models = {name: _load_model(model_dir, name) for name in target_names}
        self.output_indices = {name: output_names.index(name) for name in target_names}
        self.base = np.asarray(self.scaler_x.mean_, dtype=float).reshape(-1)
        config = request["optimizer_config"]
        self.step_ratio = float(config.get("action_step_ratio", 0.05))
        self.episode_steps = int(config.get("episode_steps", 100))
        self.constraint_penalty = float(config.get("constraint_penalty", 5.0))
        self.current = self.low.copy()
        self.steps = 0

    def _predict(self, decision: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
        full = self.base.copy()
        full[self.indices] = decision
        scaled_x = self.scaler_x.transform(full.reshape(1, -1))
        raw, normalized = {}, {}
        for name, model in self.models.items():
            scaled_y = float(np.asarray(model.predict(scaled_x)).reshape(-1)[0])
            scaler_y = self.scalers[f"scaler_y_{self.output_indices[name]}"]
            raw[name] = float(scaler_y.inverse_transform([[scaled_y]])[0, 0])
            normalized[name] = scaled_y
        return raw, normalized

    def _reward(self, raw: dict[str, float], normalized: dict[str, float]) -> tuple[float, bool]:
        cost = sum(
            item["weight"] * (normalized[item["name"]] if item["minimize"] else -normalized[item["name"]])
            for item in self.objectives
        )
        violation = 0.0
        for item in self.constraints:
            value, limit = raw[item["target_obj"]], float(item["limit_value"])
            delta = max(0.0, value - limit) if item["constraint_kind"] == "upper" else max(0.0, limit - value)
            violation += delta / max(abs(limit), 1.0)
        return -float(cost) - self.constraint_penalty * violation, violation == 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current = self.np_random.uniform(self.low, self.high).astype(np.float32)
        self.steps = 0
        return self.current.copy(), {}

    def step(self, action):
        delta = np.asarray(action, dtype=float) * (self.high - self.low) * self.step_ratio
        self.current = np.clip(self.current + delta, self.low, self.high).astype(np.float32)
        self.steps += 1
        raw, normalized = self._predict(self.current)
        reward, feasible = self._reward(raw, normalized)
        info = {"objectives": raw, "feasible": feasible, "decision": self.current.copy()}
        return self.current.copy(), reward, False, self.steps >= self.episode_steps, info


def _write_rl_solutions(records, output_path: Path, request) -> dict[str, Any]:
    records = sorted(records, key=lambda item: item["reward"], reverse=True)
    unique, seen = [], set()
    for record in records:
        key = tuple(np.round(record["decision"], 8))
        if key not in seen:
            seen.add(key)
            unique.append(record)
        if len(unique) >= request["optimizer_config"].get("max_solutions", 100):
            break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".rl_", suffix=".tmp", dir=output_path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            for record in unique:
                values = [*record["decision"], *[record["objectives"][name] for name in request["objective_names"]]]
                cells = [f"{float(value):.12g}" for value in values]
                cells.append("true" if record["feasible"] else "false")
                stream.write("\t".join(cells) + "\n")
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    columns = [*request["decision_var_names"], *request["objective_names"], "feasible"]
    return {
        "solution_count": len(unique),
        "all_solution_feasible": bool(unique) and all(item["feasible"] for item in unique),
        "columns": columns,
    }


def run_parameterized_rl(request: dict[str, Any], *, model_dir: str, output_path: str) -> dict[str, Any]:
    from stable_baselines3 import PPO

    env = SurrogatePPOEnv(request, model_dir)
    config = request["optimizer_config"]
    model = PPO(
        "MlpPolicy", env, verbose=0,
        learning_rate=float(config.get("learning_rate", 0.001)),
        seed=int(config.get("seed", 42)),
    )
    model.learn(total_timesteps=int(config.get("total_timesteps", 20000)))
    records = []
    for episode in range(int(config.get("evaluation_episodes", 10))):
        observation, _ = env.reset(seed=int(config.get("seed", 42)) + episode)
        done = False
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            records.append({**info, "reward": float(reward)})
            done = terminated or truncated
    output = Path(output_path)
    summary = _write_rl_solutions(records, output, request)
    return {"solution_txt_path": str(output.resolve()), **summary}


__all__ = ["SurrogatePPOEnv", "run_parameterized_rl"]
