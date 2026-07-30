"""采样模块 (:mod:`mobo.automation.sampling`) 纯逻辑测试。"""

import pandas as pd
import pytest

from mobo.automation import sampling


def test_lhs_samples_shape_and_range():
    param_ranges = {"t1": (0.0, 10.0), "t2": (100.0, 200.0)}
    df = sampling.lhs_samples(8, param_ranges)
    assert df.shape == (8, 2)
    assert (df["t1"] >= 0.0).all() and (df["t1"] <= 10.0).all()
    assert (df["t2"] >= 100.0).all() and (df["t2"] <= 200.0).all()


def test_boundary_samples_cartesian():
    param_ranges = {"t1": (0.0, 10.0), "t2": (100.0, 200.0)}
    df = sampling.boundary_samples(param_ranges)
    # 2 个参数各 2 个边界 -> 4 个组合
    assert df.shape == (4, 2)
    assert set(df["t1"]) == {0.0, 10.0}


def test_generate_lhs_includes_boundaries_as_strings():
    param_ranges = {"t1": (0.0, 10.0), "t2": (100.0, 200.0)}
    df = sampling.generate_lhs(5, param_ranges)
    # LHS(5) + 边界(4)，去重后 >= 5
    assert df.shape[0] >= 5
    assert df.shape[1] == 2
    # 元素被格式化成两位小数字符串
    assert all(isinstance(v, str) and "." in v for v in df.iloc[0].tolist())


def test_generate_full_factorial_counts():
    param_ranges = {"t1": (0.0, 10.0), "t2": (100.0, 200.0)}
    df = sampling.generate_full_factorial(param_ranges, [3, 2])
    assert df.shape == (6, 2)  # 3 * 2


def test_generate_full_factorial_level_mismatch():
    with pytest.raises(ValueError, match="level_nums"):
        sampling.generate_full_factorial({"t1": (0.0, 1.0)}, [2, 3])


def test_save_samples_writes_tab_separated(tmp_path):
    df = pd.DataFrame({"a": ["1.00", "2.00"], "b": ["3.00", "4.00"]})
    out = sampling.save_samples("t1", df, "unit", str(tmp_path))
    assert out.endswith("t1-unit.txt")
    loaded = pd.read_csv(out, sep="\t", header=None)
    assert loaded.shape == (2, 2)


def test_generate_samples_lhs(tmp_path):
    out = sampling.generate_samples("t1", "lhs", {"t1": (0.0, 10.0)}, str(tmp_path), n_samples=4)
    assert out.endswith("t1-lhs.txt")


def test_generate_samples_full(tmp_path):
    out = sampling.generate_samples(
        "t1", "full", {"t1": (0.0, 10.0), "t2": (0.0, 5.0)}, str(tmp_path), level_nums=[2, 2]
    )
    assert out.endswith("t1-fullfactorial.txt")


def test_generate_samples_full_requires_levels(tmp_path):
    with pytest.raises(ValueError, match="level_nums"):
        sampling.generate_samples("t1", "full", {"t1": (0.0, 10.0)}, str(tmp_path))


def test_generate_samples_unsupported(tmp_path):
    with pytest.raises(ValueError, match="不支持的采样方法"):
        sampling.generate_samples("t1", "bogus", {"t1": (0.0, 10.0)}, str(tmp_path))
