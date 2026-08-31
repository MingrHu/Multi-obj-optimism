"""工程质量评分工具测试"""

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[3] / "tools" / "quality_score.py"
    spec = importlib.util.spec_from_file_location("quality_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics():
    return {
        "complexity": {
            "maintainability_average": 80,
            "block_count": 10,
            "complexity_maximum": 10,
            "complexity_ranks": {"A": 10},
        },
        "coverage": {"percent": 80},
        "ruff": {"violation_count": 0},
        "dead_code": {"certain_count": 0},
        "documentation": {"passed": True},
    }


def test_components_give_full_complexity_score_for_simple_code():
    module = _load_module()

    assert module._components(_metrics())["complexity"] == 20


def test_complexity_score_uses_ratio_instead_of_repository_size():
    module = _load_module()
    metrics = _metrics()
    metrics["complexity"]["block_count"] = 100
    metrics["complexity"]["complexity_ranks"] = {"A": 90, "C": 10}

    assert module._components(metrics)["complexity"] == 19.16


def test_grade_boundaries():
    module = _load_module()

    assert [module._grade(score) for score in (90, 80, 70, 60, 59)] == ["A", "B", "C", "D", "E"]
