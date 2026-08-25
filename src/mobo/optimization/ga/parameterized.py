"""由协议参数装配并执行 NSGA-II。

历史 :func:`mobo.optimization.ga.run.NSGA2_run` 保留为演示入口。本模块复用同一问题类、
交叉算子和模型加载逻辑，为 service 层提供不含硬编码业务参数的运行函数。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from joblib import load as joblib_load
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

from .operators import AdaptiveSBX
from .problem import ConstraintSpec, ObjectiveSpec, SurrogateOptimizationProblem
from .run import _load_model


def _write_solutions(
    result: Any,
    output_path: Path,
    decision_names: list[str],
    objective_config: list[dict[str, Any]],
) -> dict[str, Any]:
    """以 UTF-8 TSV 原子写出解集，并返回可用于响应的汇总信息。"""
    if result.X is None or result.F is None:
        x_values = np.empty((0, len(decision_names)), dtype=float)
        objective_values = np.empty((0, len(objective_config)), dtype=float)
    else:
        x_values = np.atleast_2d(np.asarray(result.X, dtype=float))
        minimized_values = np.atleast_2d(np.asarray(result.F, dtype=float))
        signs = np.array(
            [1.0 if item["minimize"] else -1.0 for item in objective_config], dtype=float
        )
        objective_values = minimized_values * signs

    raw_constraints = getattr(result, "G", None)
    if raw_constraints is None or len(x_values) == 0:
        feasible = np.ones(len(x_values), dtype=bool)
    else:
        constraint_values = np.atleast_2d(np.asarray(raw_constraints, dtype=float))
        feasible = np.all(constraint_values <= 0, axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".pareto_", suffix=".tmp", dir=str(output_path.parent), text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            headers = [*decision_names, *[item["name"] for item in objective_config], "feasible"]
            stream.write("\t".join(headers) + "\n")
            for row_index, x_row in enumerate(x_values):
                values = [*x_row, *objective_values[row_index]]
                cells = [f"{float(value):.12g}" for value in values]
                cells.append("true" if feasible[row_index] else "false")
                stream.write("\t".join(cells) + "\n")
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)

    return {
        "solution_count": int(len(x_values)),
        "all_solution_feasible": bool(np.all(feasible)) if len(feasible) else False,
    }


def run_parameterized_nsga2(
    request: dict[str, Any],
    *,
    model_dir: str,
    output_path: str,
) -> dict[str, Any]:
    """按已校验协议参数运行 NSGA-II，并返回产物与约束汇总。"""
    objective_names = request["objective_names"]
    all_var_list = request["all_var_list"]
    input_var_count = request["input_var_count"]
    output_names = all_var_list[input_var_count:]

    scalers_path = os.path.join(model_dir, f"{objective_names[0]}_scalers.pkl")
    if not os.path.isfile(scalers_path):
        raise FileNotFoundError(f"找不到标准化器文件：{scalers_path}")
    scalers = joblib_load(scalers_path)

    objectives = []
    minimize_by_name = {
        item["name"]: item["minimize"] for item in request["objective_config"]
    }
    for name in objective_names:
        objectives.append(ObjectiveSpec(
            name=name,
            model=_load_model(model_dir, name),
            y_index=output_names.index(name),
            minimize=minimize_by_name[name],
        ))

    constraints = [
        ConstraintSpec(
            objective=item["target_obj"],
            kind=item["constraint_kind"],
            value=item["limit_value"],
        )
        for item in request["constraints"]
    ]
    bounds = [(item["lower"], item["upper"]) for item in request["decision_bounds"]]
    problem = SurrogateOptimizationProblem(
        objectives=objectives,
        scalers=scalers,
        decision_var_indices=request["decision_var_indices"],
        bounds=bounds,
        constraints=constraints,
    )

    config = request["optimizer_config"]
    algorithm = NSGA2(
        pop_size=config["pop_size"],
        n_offsprings=config["n_offsprings"],
        sampling=FloatRandomSampling(),
        crossover=AdaptiveSBX(eta_c_min=20, eta_c_max=5, prob=0.95),
        mutation=PM(eta=20),
        eliminate_duplicates=config["eliminate_duplicates"],
    )
    result = minimize(
        problem,
        algorithm,
        ("n_gen", config["n_gen"]),
        seed=config["seed"],
        verbose=False,
        save_history=False,
    )

    summary = _write_solutions(
        result,
        Path(output_path),
        request["decision_var_names"],
        request["objective_config"],
    )
    return {
        "solution_txt_path": str(Path(output_path).resolve()),
        **summary,
    }


__all__ = ["run_parameterized_nsga2"]
