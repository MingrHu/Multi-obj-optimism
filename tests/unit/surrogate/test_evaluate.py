"""SurrogateModelEvaluator 测试（非 DNN 路径，快速）。"""

import numpy as np
import pytest

from mobo.surrogate.evaluate import SurrogateModelEvaluator, TargetCVSummary


@pytest.fixture
def small_evaluator(simulated_data_file):
    return SurrogateModelEvaluator(
        data_file=simulated_data_file,
        vars_out=["a", "b", "c", "grain", "load"],
        n_vars=3,
        n_splits=3,
        model_params={
            "PRG": {"degree": 2},
            "SVR": {"kernel": "rbf", "C": 1.0, "epsilon": 0.1},
            "RF": {"n_estimators": 20, "n_jobs": 1},
        },
    )


def test_n_splits_validation(simulated_data_file):
    with pytest.raises(ValueError):
        SurrogateModelEvaluator(simulated_data_file, ["a", "b", "c", "grain", "load"], 3, n_splits=1)


def test_evaluate_prg_svr_rf(small_evaluator):
    summaries = small_evaluator.evaluate(
        models=["PRG", "SVR", "RF"],
        target_indices=[0, 1],
    )
    assert len(summaries) == 3 * 2
    assert all(isinstance(s, TargetCVSummary) for s in summaries)
    # 每个摘要都应被打分
    assert all(s.score is not None for s in summaries)


def test_evaluate_target_index_out_of_range(small_evaluator):
    with pytest.raises(ValueError):
        small_evaluator.evaluate(models=["PRG"], target_indices=[5])


def test_format_report_lines(small_evaluator):
    summaries = small_evaluator.evaluate(models=["PRG"], target_indices=[0])
    lines = small_evaluator.format_report_lines(summaries)
    assert any("PRG" in ln for ln in lines)


def test_save_report_writes_files(small_evaluator, tmp_path):
    summaries = small_evaluator.evaluate(models=["PRG"], target_indices=[0])
    text_path = tmp_path / "report.txt"
    json_path = tmp_path / "report.json"
    small_evaluator.save_report(summaries, text_path=str(text_path), json_path=str(json_path))
    assert text_path.exists() and text_path.stat().st_size > 0
    assert json_path.exists() and json_path.stat().st_size > 0


def test_build_model_unsupported(small_evaluator):
    with pytest.raises(ValueError):
        small_evaluator._build_model(model_name="NOPE", input_dim=3)
