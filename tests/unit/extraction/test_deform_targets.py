"""deform_targets 原子提取函数与 von Mises 数值测试。"""

import math

import numpy as np
import pytest

from mobo.extraction.deform_targets import (
    _extractGrainStdv,
    _extractMaxLoad,
    _extractMaxStress,
    calculate_von_mises,
)


def test_calculate_von_mises_uniaxial():
    # 单轴应力 sxx=10，其余为 0 -> von Mises = 10
    assert calculate_von_mises([10, 0, 0, 0, 0, 0]) == pytest.approx(10.0)


def test_calculate_von_mises_hydrostatic():
    # 纯静水压 -> von Mises = 0
    assert calculate_von_mises([5, 5, 5, 0, 0, 0]) == pytest.approx(0.0)


def test_extract_max_load_last_frame():
    frame = [
        "some header line",
        "FORCE 2 0 0 123456.0",
        "other",
    ]
    # obj_id 为对象 ID 字符串，需与 FORCE 行的第 2 列 (arry[1]) 一致
    result = _extractMaxLoad([frame], obj_id="2", inprogress=False)
    assert result == "123456.00"


def test_extract_max_load_inprogress_takes_max():
    f1 = ["FORCE 2 0 0 100.0"]
    f2 = ["FORCE 2 0 0 250.0"]
    f3 = ["FORCE 2 0 0 180.0"]
    result = _extractMaxLoad([f1, f2, f3], obj_id="2", inprogress=True)
    assert result == "250.00"


def test_extract_grain_stdv():
    # USRELM 1 3 -> 随后 3 行，arr[3] 为晶粒尺寸；obj_id 为对象 ID (arry[1])
    frame = [
        "USRELM 1 3 x x",
        "1 a b 10.0",
        "2 a b 20.0",
        "3 a b 30.0",
    ]
    result = _extractGrainStdv([frame], obj_id="1", inprogress=False)
    import statistics
    expected = statistics.stdev([10.0, 20.0, 30.0])
    assert float(result) == pytest.approx(expected, rel=1e-6)


def test_extract_max_stress():
    # STRESS <obj> <num>=1 <x> -> 1 对数据行
    # arry1: 索引1..5 为 sxx..syz，arry2[0] 为 sxz
    frame = [
        "STRESS 1 1 x",
        "id 10.0 0.0 0.0 0.0 0.0",
        "0.0",
    ]
    result = _extractMaxStress([frame], obj_id="1", inprogress=False)
    # 单轴 sxx=10 -> von Mises = 10
    assert result == "10.00"
