"""参数化 NSGA-II 的结果导出测试。"""

from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from mobo.optimization.ga.parameterized import _write_solutions, run_parameterized_nsga2


def test_write_solutions_restores_maximized_objective_sign(tmp_path):
    result = SimpleNamespace(
        X=np.array([[900.0]]),
        F=np.array([[-12.5, 3.0]]),
        G=np.array([[-1.0]]),
    )
    output = tmp_path / "pareto.tsv"

    summary = _write_solutions(
        result,
        output,
        ["temperature"],
        [
            {"name": "strength", "minimize": False},
            {"name": "grain", "minimize": True},
        ],
    )

    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "temperature\tstrength\tgrain\tfeasible"
    assert lines[1] == "900\t12.5\t3\ttrue"
    assert summary == {"solution_count": 1, "all_solution_feasible": True}


def test_run_parameterized_nsga2_end_to_end(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([[0.0], [2.0], [4.0], [6.0]])
    scaler_x = StandardScaler().fit(x)
    scaler_y = StandardScaler().fit(y)
    model = LinearRegression().fit(scaler_x.transform(x), scaler_y.transform(y).ravel())
    joblib.dump(model, model_dir / "result_model.pkl")
    joblib.dump(
        {"scaler_X": scaler_x, "scaler_y_0": scaler_y},
        model_dir / "result_scalers.pkl",
    )
    output = tmp_path / "result.tsv"
    request = {
        "objective_names": ["result"],
        "all_var_list": ["x", "result"],
        "input_var_count": 1,
        "objective_config": [{"name": "result", "minimize": True}],
        "constraints": [],
        "decision_bounds": [{"lower": 0.0, "upper": 3.0}],
        "decision_var_indices": [0],
        "decision_var_names": ["x"],
        "optimizer_config": {
            "pop_size": 10,
            "n_offsprings": 10,
            "eliminate_duplicates": True,
            "n_gen": 2,
            "seed": 42,
        },
    }

    summary = run_parameterized_nsga2(
        request, model_dir=str(model_dir), output_path=str(output)
    )

    assert summary["solution_count"] >= 1
    assert summary["solution_txt_path"] == str(output.resolve())
    assert output.read_text(encoding="utf-8").startswith("x\tresult\tfeasible\n")
