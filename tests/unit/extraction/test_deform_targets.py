"""deform_targets 原子提取函数与 von Mises 数值测试。"""

import statistics

import pytest

from mobo.extraction.deform_targets import (
    _extractEffectiveStrainStdv,
    _extractGrainMorph,
    _extractMaxLoad,
    _extractMaxStress,
    _extractUsrGrainStdv,
    calculate_von_mises,
)


def _valid_frame(lines):
    return lines + [""] * (10001 - len(lines))


def test_calculate_von_mises_uniaxial():
    # 单轴应力 sxx=10，其余为 0 -> von Mises = 10
    assert calculate_von_mises([10, 0, 0, 0, 0, 0]) == pytest.approx(10.0)


def test_calculate_von_mises_hydrostatic():
    # 纯静水压 -> von Mises = 0
    assert calculate_von_mises([5, 5, 5, 0, 0, 0]) == pytest.approx(0.0)


def test_extract_max_load_selects_component():
    frame = _valid_frame(["FORCE 2 10.0 20.0 30.0"])
    assert _extractMaxLoad([frame], obj_id="2", inprogress=False, select_component=0) == "10.00"
    assert _extractMaxLoad([frame], obj_id="2", inprogress=False, select_component=2) == "30.00"


def test_extract_max_load_variable_components():
    # 只有 x 分量时，越界分量返回 0
    frame = _valid_frame(["FORCE 2 15.0"])
    assert _extractMaxLoad([frame], obj_id="2", inprogress=False, select_component=0) == "15.00"
    assert _extractMaxLoad([frame], obj_id="2", inprogress=False, select_component=2) == "0.00"


def test_extract_max_load_inprogress_takes_max():
    f1 = _valid_frame(["FORCE 2 100.0 0 0"])
    f2 = _valid_frame(["FORCE 2 250.0 0 0"])
    f3 = _valid_frame(["FORCE 2 180.0 0 0"])
    result = _extractMaxLoad([f1, f2, f3], obj_id="2", inprogress=True, select_component=0)
    assert result == "250.00"


def test_extract_max_load_handles_negative_y_force():
    frames = [
        _valid_frame(["FORCE 3 0 -125.0 0"]),
        _valid_frame(["FORCE 3 0 -250.0 0"]),
    ]
    assert _extractMaxLoad(frames, "3", True, 1) == "250.00"


def test_extract_max_load_final_frame_keeps_magnitude_not_sign():
    frame = _valid_frame(["FORCE 3 0 -125.0 0"])
    assert _extractMaxLoad([frame], "3", False, 1) == "125.00"


def test_extract_effective_strain_standard_deviation():
    frame = _valid_frame(["STRAIN 1 3 0", "1 0.1", "2 0.4", "3 0.7"])
    expected = statistics.stdev([0.1, 0.4, 0.7])
    assert float(_extractEffectiveStrainStdv([frame], "1", False)) == pytest.approx(expected)


def test_extract_usr_grain_stdv():
    # USRELM 1 3 -> 随后 3 行，select_component 选列（默认 3）
    frame = _valid_frame([
        "USRELM 1 3 x x",
        "1 a b 10.0",
        "2 a b 20.0",
        "3 a b 30.0",
    ])
    result = _extractUsrGrainStdv([frame], obj_id="1", inprogress=False, select_component=3)
    expected = statistics.stdev([10.0, 20.0, 30.0])
    assert float(result) == pytest.approx(expected, rel=1e-6)


def test_extract_grain_morph_selects_component():
    # GRAIN 1 2 16 -> 2 个单元，每单元 16 个值（跨多行），comp 选单元内分量
    frame = _valid_frame([
        "GRAIN 1 2 16 0.0",
        "1 1.0 40 40 0 0",
        "0 0 0 0 4.0",
        "0 0 0 0 0",
        "9.0",
        "2 2.0 40 40 0 0",
        "0 0 0 0 4.0",
        "0 0 0 0 0",
        "1.0",
    ])
    # 单元内索引 0 -> [1.0, 2.0]
    r0 = _extractGrainMorph([frame], obj_id="1", inprogress=False, select_component=0)
    assert r0 == "{:.2f}".format(statistics.fmean([1.0, 2.0]))
    # 单元内索引 15 -> [9.0, 1.0]
    r15 = _extractGrainMorph([frame], obj_id="1", inprogress=False, select_component=15)
    assert r15 == "{:.2f}".format(statistics.fmean([9.0, 1.0]))


def test_extract_grain_morph_missing_block():
    assert _extractGrainMorph([["nothing here"]], obj_id="1", inprogress=False) == "0.00"


def test_extractors_ignore_small_incomplete_frames():
    small = ["FORCE 2 999 0 0", "GRAIN 1 1 16 0"]
    valid = _valid_frame(["FORCE 2 10 0 0"])
    assert _extractMaxLoad([valid, small], "2", False, 0) == "10.00"
    assert _extractMaxLoad([small], "2", False, 0) == "0.00"


def test_extract_max_stress():
    # STRESS <obj> <num>=1 <x> -> 1 对数据行
    frame = _valid_frame([
        "STRESS 1 1 x",
        "id 10.0 0.0 0.0 0.0 0.0",
        "0.0",
    ])
    result = _extractMaxStress([frame], obj_id="1", inprogress=False)
    # 单轴 sxx=10 -> von Mises = 10
    assert result == "10.00"
