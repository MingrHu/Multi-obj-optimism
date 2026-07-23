"""deform_utils 纯逻辑函数测试（不启动 DEFORM 子进程）。"""

import os

import pandas as pd
import pytest

from mobo.automation.deform_utils import (
    FormatFloat,
    FullSampleGenerate,
    GetNewFilePath,
    LHSSampleGenerate,
    SaveResult,
)


def test_format_float_scientific():
    assert FormatFloat("0") == "0.0000000000E+000"
    # 正常数值 -> 大写科学计数法，指数补足 3 位
    s = FormatFloat("1234.5")
    assert "E+" in s
    mantissa, exp = s.split("E")
    assert len(exp) == 4  # 符号 + 3 位


def test_format_float_negative_exponent():
    s = FormatFloat("0.001")
    assert "E-" in s


def test_format_float_non_numeric():
    assert FormatFloat("abc") == "abc"


def test_get_new_file_path():
    res = GetNewFilePath("/a/b/model.KEY", "/out", "3", "DB")
    assert res == os.path.join("/out", "model3.DB")


def test_save_result_writes(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    SaveResult(df, "test", str(tmp_path))
    out = tmp_path / "INtest.txt"
    assert out.exists()
    content = out.read_text()
    assert "1" in content


def test_lhs_sample_generate(tmp_path):
    param_ranges = {"t1": (0.0, 10.0), "t2": (100.0, 200.0)}
    LHSSampleGenerate(5, param_ranges, str(tmp_path))
    out = tmp_path / "INlhs.txt"
    assert out.exists()
    df = pd.read_csv(out, sep="\t", header=None)
    # 5 个 LHS 样本 + 边界组合(2*2=4)，去重后 >= 5
    assert df.shape[0] >= 5
    assert df.shape[1] == 2


def test_full_sample_generate(tmp_path):
    param_ranges = {"t1": (0.0, 10.0), "t2": (100.0, 200.0)}
    FullSampleGenerate(param_ranges, str(tmp_path), [3, 2])
    out = tmp_path / "INfullfactorial.txt"
    assert out.exists()
    df = pd.read_csv(out, sep="\t", header=None)
    # 全因子：3 * 2 = 6 个组合
    assert df.shape[0] == 6
