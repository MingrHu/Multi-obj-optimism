"""集成测试：用真实 PRG 模型跑一次小规模 NSGA-II（输出重定向到 tmp）。"""

import os

import pytest

pytestmark = pytest.mark.integration


def test_nsga2_run_small(prg_model_dir, monkeypatch, tmp_path):
    """缩小 pop_size/n_gen 跑通 NSGA2_run 全流程，验证能加载模型并输出结果。"""
    import mobo.optimization.ga.run as run_mod

    # 输出重定向到 tmp，避免污染仓库 data/
    monkeypatch.setattr(run_mod, "DATA_DIR", tmp_path)

    # 缩小算法规模以加速：patch NSGA2 与 minimize 的默认调用
    from pymoo.algorithms.moo.nsga2 import NSGA2 as RealNSGA2
    from pymoo.optimize import minimize as real_minimize

    def small_nsga2(*args, **kwargs):
        kwargs["pop_size"] = 8
        kwargs["n_offsprings"] = 8
        return RealNSGA2(*args, **kwargs)

    def small_minimize(problem, algorithm, termination, **kwargs):
        return real_minimize(problem, algorithm, ("n_gen", 3), **{**kwargs, "verbose": False})

    monkeypatch.setattr(run_mod, "NSGA2", small_nsga2)
    monkeypatch.setattr(run_mod, "minimize", small_minimize)

    run_mod.NSGA2_run()

    assert (tmp_path / "pareto_solutions.txt").exists()
    assert (tmp_path / "pareto_front.png").exists()
