"""deform_targets 原子提取函数与 von Mises 数值测试。"""

import statistics

import pytest

from mobo.extraction.deform_targets import (
    _extractGrainMorph,
    _extractMaxLoad,
    _extractMaxStress,
    _extractUsrGrainStdv,
    calculate_von_mises,
)


def test_calculate_von_mises_uniaxial():
    # 单轴应力 sxx=10，其余为 0 -> von Mises = 10
    assert calculate_von_mises([10, 0, 0, 0, 0, 0]) == pytest.approx(10.0)


def test_calculate_von_mises_hydrostatic():
    # 纯静水压 -> von Mises = 0
    assert calculate_von_mises([5, 5, 5, 0, 0, 0]) == pytest.approx(0.0)


def test_extract_max_load_selects_component():
    frame = ["FORCE 2 10.0 20.0 30.0"]
    assert _extractMaxLoad([frame], obj_id="2", inprogress=False, select_component=0) == "10.00"
    assert _extractMaxLoad([frame], obj_id="2", inprogress=False, select_component=2) == "30.00"


def test_extract_max_load_variable_components():
    # 只有 x 分量时，越界分量返回 0
    frame = ["FORCE 2 15.0"]
    assert _extractMaxLoad([frame], obj_id="2", inprogress=False, select_component=0) == "15.00"
    assert _extractMaxLoad([frame], obj_id="2", inprogress=False, select_component=2) == "0.00"


def test_extract_max_load_inprogress_takes_max():
    f1 = ["FORCE 2 100.0 0 0"]
    f2 = ["FORCE 2 250.0 0 0"]
    f3 = ["FORCE 2 180.0 0 0"]
    result = _extractMaxLoad([f1, f2, f3], obj_id="2", inprogress=True, select_component=0)
    assert result == "250.00"


def test_extract_usr_grain_stdv():
    # USRELM 1 3 -> 随后 3 行，select_component 选列（默认 3）
    frame = [
        "USRELM 1 3 x x",
        "1 a b 10.0",
        "2 a b 20.0",
        "3 a b 30.0",
    ]
    result = _extractUsrGrainStdv([frame], obj_id="1", inprogress=False, select_component=3)
    expected = statistics.stdev([10.0, 20.0, 30.0])
    assert float(result) == pytest.approx(expected, rel=1e-6)


def test_extract_grain_morph_selects_component():
    # GRAIN 1 2 16 -> 2 个单元，每单元 16 个值（跨多行），comp 选单元内分量
    frame = [
        "GRAIN 1 2 16 0.0",
        "1 1.0 40 40 0 0",
        "0 0 0 0 4.0",
        "0 0 0 0 0",
        "9.0",
        "2 2.0 40 40 0 0",
        "0 0 0 0 4.0",
        "0 0 0 0 0",
        "1.0",
    ]
    # 单元内索引 0 -> [1.0, 2.0]
    r0 = _extractGrainMorph([frame], obj_id="1", inprogress=False, select_component=0)
    assert r0 == "{:.2f}".format(statistics.stdev([1.0, 2.0]))
    # 单元内索引 15 -> [9.0, 1.0]
    r15 = _extractGrainMorph([frame], obj_id="1", inprogress=False, select_component=15)
    assert r15 == "{:.2f}".format(statistics.stdev([9.0, 1.0]))


def test_extract_grain_morph_missing_block():
    assert _extractGrainMorph([["nothing here"]], obj_id="1", inprogress=False) == "0.00"


def test_extract_max_stress():
    # STRESS <obj> <num>=1 <x> -> 1 对数据行
    frame = [
        "STRESS 1 1 x",
        "id 10.0 0.0 0.0 0.0 0.0",
        "0.0",
    ]
    result = _extractMaxStress([frame], obj_id="1", inprogress=False)
    # 单轴 sxx=10 -> von Mises = 10
    assert result == "10.00"
