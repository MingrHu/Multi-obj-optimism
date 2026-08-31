"""仓库质量检查总入口测试"""

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[3] / "tools" / "run_quality_checks.py"
    spec = importlib.util.spec_from_file_location("run_quality_checks", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_commands_add_security_only_when_requested(tmp_path):
    module = _load_module()

    basic = [name for name, _ in module._commands(tmp_path, 60, False)]
    secured = [name for name, _ in module._commands(tmp_path, 60, True)]

    assert basic == ["tests", "ruff", "complexity", "documentation", "score"]
    assert secured[-1] == "dependency-security"


def test_markdown_contains_score_and_failed_check():
    module = _load_module()
    result = module.CheckResult("ruff", ["ruff"], 1, 1.25, "ruff.log")

    report = module._markdown([result], {"score": 75.0, "grade": "C"})

    assert "75.0/100" in report
    assert "失败 (1)" in report
