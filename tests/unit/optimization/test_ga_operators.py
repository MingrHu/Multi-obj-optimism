"""GA 算子与 pareto 输出/解析工具测试。"""

import types

import numpy as np
import pytest

from mobo.optimization.ga.operators import (
    AdaptiveSBX,
    get_paretodata,
    save_pareto_solutions,
)


class _FakeRes:
    def __init__(self, X, F, G=None):
        self.X = X
        self.F = F
        self.G = G


def test_save_pareto_solutions_writes(tmp_path):
    res = _FakeRes(
        X=np.array([[900.0, 500.0], [910.0, 480.0]]),
        F=np.array([[25.0, 3.0e5], [26.0, 2.9e5]]),
    )
    out = tmp_path / "pareto.txt"
    save_pareto_solutions(res, filename=str(out), var_names=["temp", "die"], obj_names=["grain", "load"])
    text = out.read_text(encoding="utf-8")
    assert "Pareto前沿最优解集" in text
    assert "temp" in text and "grain" in text
    assert "解数量: 2" in text


def test_save_pareto_solutions_name_length_check(tmp_path):
    res = _FakeRes(X=np.zeros((1, 2)), F=np.zeros((1, 2)))
    with pytest.raises(ValueError):
        save_pareto_solutions(res, filename=str(tmp_path / "x.txt"), var_names=["only_one"])


def test_save_pareto_solutions_default_names(tmp_path):
    res = _FakeRes(X=np.zeros((1, 2)), F=np.zeros((1, 2)))
    out = tmp_path / "p.txt"
    save_pareto_solutions(res, filename=str(out))
    text = out.read_text(encoding="utf-8")
    assert "x1" in text and "f1" in text


def test_adaptive_sbx_eta_updates():
    sbx = AdaptiveSBX(eta_c_min=5, eta_c_max=20, prob=0.9)
    # 构造带 n_gen / termination.n_max_gen 的假算法对象
    algo = types.SimpleNamespace(n_gen=1, termination=types.SimpleNamespace(n_max_gen=100))

    captured = {}

    # 用父类 _do 的 monkeypatch 记录 eta 而不真正执行交叉
    # 注意：SBX._do 以绑定方法调用，第一个参数为 self
    def fake_super_do(_self, problem, X, **kwargs):
        captured["eta"] = _self.eta
        return X

    # 直接调用自适应逻辑：progress 很小 -> eta 接近 eta_c_max
    import mobo.optimization.ga.operators as ops
    orig = ops.SBX._do
    ops.SBX._do = fake_super_do
    try:
        sbx._do(problem=None, X=np.zeros((2, 3)), algorithm=algo)
    finally:
        ops.SBX._do = orig

    # progress=0.01 -> eta = 20 + (5-20)*sqrt(0.01) = 20 - 1.5 = 18.5
    assert captured["eta"] == pytest.approx(18.5, rel=1e-6)


def test_adaptive_sbx_no_algorithm_falls_back():
    sbx = AdaptiveSBX()
    import mobo.optimization.ga.operators as ops
    called = {"n": 0}

    def fake_super_do(_self, problem, X, **kwargs):
        called["n"] += 1
        return X

    orig = ops.SBX._do
    ops.SBX._do = fake_super_do
    try:
        sbx._do(problem=None, X=np.zeros((2, 3)))
    finally:
        ops.SBX._do = orig
    assert called["n"] == 1


def test_get_paretodata(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text(
        "1 | 900 | 500 | 30 | 25.0 | 300000\n"
        "2 | 910 | 480 | 28 | 26.0 | 290000\n",
        encoding="utf-8",
    )
    dst = tmp_path / "out.txt"
    get_paretodata(str(src), str(dst))
    lines = dst.read_text(encoding="utf-8").strip().splitlines()
    # 第 5、6 列 (stdv, load) -> load 除以 1000
    assert lines[0].split("\t") == ["25.0", "300.00"]
    assert lines[1].split("\t") == ["26.0", "290.00"]
