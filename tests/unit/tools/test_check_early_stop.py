"""变化率早停判断工具测试。"""

import importlib.util
from io import StringIO
from pathlib import Path
import sys


def _load_module():
    path = Path(__file__).resolve().parents[3] / "tools" / "check_early_stop.py"
    spec = importlib.util.spec_from_file_location("check_early_stop", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(monkeypatch, capsys, text):
    module = _load_module()
    monkeypatch.setattr(sys, "stdin", StringIO(text))
    code = module.main()
    return code, capsys.readouterr()


def test_reports_satisfied_when_change_rate_is_below_five_percent(monkeypatch, capsys):
    code, output = _run(monkeypatch, capsys, "3\n100 99 96\n不会被继续读取")

    assert code == 0
    assert output.out == "变化率：4.000000%\n满足停止条件\n"


def test_five_percent_exactly_satisfies_condition(monkeypatch, capsys):
    code, output = _run(monkeypatch, capsys, "2\n100\n95\n")

    assert code == 0
    assert output.out == "变化率：5.000000%\n满足停止条件\n"


def test_rejects_incorrect_value_count(monkeypatch, capsys):
    code, output = _run(monkeypatch, capsys, "3\n1 2\n")

    assert code == 2
    assert "需要输入3个数值，实际输入2个" in output.err
