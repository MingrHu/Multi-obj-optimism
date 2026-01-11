from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pymoo.core.problem import ElementwiseProblem


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    model: Any
    y_index: int
    minimize: bool = True


@dataclass(frozen=True)
class ConstraintSpec:
    objective: Union[str, int]
    kind: str  # "upper" or "lower"
    value: float


class SurrogateOptimizationProblem(ElementwiseProblem):
    def __init__(
        self,
        *,
        objectives: Sequence[ObjectiveSpec],
        scalers: Dict[str, Any],
        decision_var_indices: Optional[Sequence[int]] = None,
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
        x_base: Optional[Sequence[float]] = None,
        fixed_values: Optional[Dict[int, float]] = None,
        constraints: Optional[Sequence[ConstraintSpec]] = None,
        bounds_from_scaler_std: float = 3.0,
    ):
        if len(objectives) == 0:
            raise ValueError("objectives 不能为空")

        self.objectives: List[ObjectiveSpec] = list(objectives)
        self.scalers = scalers
        self.constraints: List[ConstraintSpec] = list(constraints or [])
        self.fixed_values: Dict[int, float] = dict(fixed_values or {})

        if "scaler_X" not in self.scalers:
            raise KeyError("scalers 缺少 scaler_X（需要与 SurrogateModel/common.py 输出一致）")
        self.scaler_X = self.scalers["scaler_X"]

        if not hasattr(self.scaler_X, "mean_") or not hasattr(self.scaler_X, "scale_"):
            raise TypeError("scaler_X 不是 StandardScaler 或缺少 mean_/scale_ 属性")

        self.full_input_dim = int(np.asarray(self.scaler_X.mean_).reshape(-1).shape[0])

        if self.full_input_dim <= 0:
            raise ValueError("无法从 scaler_X 推断输入维度")

        if decision_var_indices is None:
            self.decision_var_indices = list(range(self.full_input_dim))
        else:
            self.decision_var_indices = list(decision_var_indices)

        if len(self.decision_var_indices) == 0:
            raise ValueError("decision_var_indices 不能为空（至少要优化 1 个变量）")

        if any((i < 0 or i >= self.full_input_dim) for i in self.decision_var_indices):
            raise ValueError(f"decision_var_indices 越界：输入维度为 {self.full_input_dim}")

        if len(set(self.decision_var_indices)) != len(self.decision_var_indices):
            raise ValueError("decision_var_indices 存在重复索引")

        if x_base is None:
            self.x_base = np.asarray(self.scaler_X.mean_, dtype=float).reshape(-1)
        else:
            self.x_base = np.asarray(list(x_base), dtype=float).reshape(-1)
            if self.x_base.shape[0] != self.full_input_dim:
                raise ValueError(f"x_base 长度必须等于 full_input_dim={self.full_input_dim}")

        xl, xu = self._resolve_bounds(
            bounds=bounds,
            bounds_from_scaler_std=bounds_from_scaler_std,
        )

        super().__init__(
            n_var=len(self.decision_var_indices),
            n_obj=len(self.objectives),
            n_constr=len(self.constraints),
            xl=xl,
            xu=xu,
        )

        self._objective_index_by_name: Dict[str, int] = {o.name: i for i, o in enumerate(self.objectives)}

        for o in self.objectives:
            key = f"scaler_y_{o.y_index}"
            if key not in self.scalers:
                raise KeyError(
                    f"scalers 缺少 {key}。"
                    f"你训练时保存的标准化器应包含 scaler_y_0/scaler_y_1/...（来自 split_data_*）。"
                )

    def _resolve_bounds(
        self,
        *,
        bounds: Optional[Sequence[Tuple[float, float]]],
        bounds_from_scaler_std: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if bounds is not None:
            bounds = list(bounds)

            if len(bounds) == len(self.decision_var_indices):
                chosen = bounds
            elif len(bounds) == self.full_input_dim:
                chosen = [bounds[i] for i in self.decision_var_indices]
            else:
                raise ValueError(
                    "bounds 长度必须等于 len(decision_var_indices) 或 full_input_dim。"
                    f"当前 bounds={len(bounds)}, decision={len(self.decision_var_indices)}, full={self.full_input_dim}"
                )

            xl = np.array([b[0] for b in chosen], dtype=float)
            xu = np.array([b[1] for b in chosen], dtype=float)
            return xl, xu

        mean = np.asarray(self.scaler_X.mean_, dtype=float).reshape(-1)
        scale = np.asarray(self.scaler_X.scale_, dtype=float).reshape(-1)
        xl_full = mean - bounds_from_scaler_std * scale
        xu_full = mean + bounds_from_scaler_std * scale

        xl = np.array([xl_full[i] for i in self.decision_var_indices], dtype=float)
        xu = np.array([xu_full[i] for i in self.decision_var_indices], dtype=float)
        return xl, xu

    def _assemble_full_x(self, x_decision: np.ndarray) -> np.ndarray:
        x_full = self.x_base.copy()
        x_decision = np.asarray(x_decision, dtype=float).reshape(-1)

        if x_decision.shape[0] != len(self.decision_var_indices):
            raise ValueError(
                f"x_decision 维度不匹配：期望 {len(self.decision_var_indices)}，实际 {x_decision.shape[0]}"
            )

        for local_i, full_i in enumerate(self.decision_var_indices):
            x_full[full_i] = x_decision[local_i]

        for full_i, val in self.fixed_values.items():
            if full_i < 0 or full_i >= self.full_input_dim:
                raise ValueError(f"fixed_values index 越界：{full_i}")
            x_full[full_i] = float(val)

        return x_full

    @staticmethod
    def _predict_scalar(model: Any, x_scaled_2d: np.ndarray) -> float:
        y = model.predict(x_scaled_2d)
        return float(np.asarray(y).reshape(-1)[0])

    def _evaluate(self, x, out, *args, **kwargs):
        x_full = self._assemble_full_x(np.array(x, dtype=float))
        x_scaled = self.scaler_X.transform(x_full.reshape(1, -1))

        raw_objective_vals: List[float] = []
        raw_by_name: Dict[str, float] = {}

        for obj in self.objectives:
            y_scaled = self._predict_scalar(obj.model, x_scaled)
            scaler_y = self.scalers[f"scaler_y_{obj.y_index}"]
            y_val = float(scaler_y.inverse_transform(np.array([[y_scaled]], dtype=float))[0, 0])

            raw_objective_vals.append(y_val)
            raw_by_name[obj.name] = y_val

        F = []
        for i, obj in enumerate(self.objectives):
            v = raw_objective_vals[i]
            F.append(v if obj.minimize else -v)

        out["F"] = np.array(F, dtype=float)

        if self.constraints:
            G = []
            for c in self.constraints:
                if c.kind not in ("upper", "lower"):
                    raise ValueError(f"Unsupported constraint kind: {c.kind}")

                if isinstance(c.objective, int):
                    idx = c.objective
                    if idx < 0 or idx >= len(raw_objective_vals):
                        raise ValueError(f"constraint objective index 越界：{idx}")
                    v = raw_objective_vals[idx]
                else:
                    if c.objective not in raw_by_name:
                        raise ValueError(f"constraint objective name 不存在：{c.objective}")
                    v = raw_by_name[c.objective]

                if c.kind == "upper":
                    G.append(v - float(c.value))
                else:
                    G.append(float(c.value) - v)

            out["G"] = np.array(G, dtype=float)