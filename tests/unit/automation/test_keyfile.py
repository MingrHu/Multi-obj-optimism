"""KEY 文件文本处理 (:mod:`mobo.automation.keyfile`) 测试。

使用合成模板验证 :func:`generate_key_files` 的核心语义：param_table 前两行为
表头（参数名 / 对象名），从第 2 行起每行是一个样本，为每个样本生成一个 KEY，
并按 (关键字, 对象ID) 定位替换目标行末尾的数值。
"""

import os
from pathlib import Path

import pytest

from mobo.automation import keyfile
from mobo.replacement.base import LineReplacement
from mobo.replacement.deform_parameters import (
    scale_abs,
    scale_movctl_block,
    scale_speed_line,
)
from mobo.replacement.registry import ReplacementRegistry


def test_format_deform_float_zero():
    assert keyfile.format_deform_float("0") == "0.0000000000E+000"


def test_format_deform_float_positive_exponent():
    s = keyfile.format_deform_float("1234.5")
    assert "E+" in s
    mantissa, exp = s.split("E")
    assert len(exp) == 4  # 符号 + 3 位


def test_format_deform_float_negative_exponent():
    assert "E-" in keyfile.format_deform_float("0.001")


def test_format_deform_float_non_numeric():
    assert keyfile.format_deform_float("abc") == "abc"


def test_derive_output_path():
    res = keyfile.derive_output_path("/a/b/model.KEY", "/out", "3", "DB")
    assert res == os.path.join("/out", "model3.DB")


def _write_template(tmp_path):
    """写一个含 NDTMP(工件温度) 与 MOVCTL(上模速度) 目标行的合成模板 KEY。"""
    template = tmp_path / "MODEL.KEY"
    template.write_text(
        "TITLE\n"
        "DEMO\n"
        "NDTMP       1       1    9.0000000000E+002\n"  # temp / workpiece(id=1)
        "MOVCTL       2       1    3.0000000000E+001\n"  # speed / topdie(id=2)
        "OTHER LINE UNTOUCHED\n",
        encoding="utf-8",
    )
    return template


def test_generate_key_files_replaces_target_values(tmp_path):
    template = _write_template(tmp_path)
    param_table = [
        ["temp", "speed"],          # 参数名表头
        ["workpiece", "topdie"],    # 对象名表头
        ["950.0", "40.0"],          # 样本 1
        ["875.0", "10.0"],          # 样本 2
    ]
    save_dir = tmp_path / "keys"
    generated = keyfile.generate_key_files(str(template), param_table, str(save_dir))

    assert len(generated) == 2
    assert generated[0] == os.path.join(str(save_dir), "MODEL0.KEY")

    content0 = open(generated[0], encoding="utf-8").read()
    # 目标行末尾数值应被替换为格式化后的样本值
    assert keyfile.format_deform_float("950.0") in content0
    assert keyfile.format_deform_float("40.0") in content0
    # 非目标行保持原样
    assert "OTHER LINE UNTOUCHED" in content0

    content1 = open(generated[1], encoding="utf-8").read()
    assert keyfile.format_deform_float("875.0") in content1
    assert keyfile.format_deform_float("10.0") in content1


def test_generate_key_files_reuses_existing_outputs(tmp_path):
    template = _write_template(tmp_path)
    param_table = [
        ["temp", "speed"],
        ["workpiece", "topdie"],
        ["950.0", "40.0"],
    ]
    generated = keyfile.generate_key_files(str(template), param_table, str(tmp_path / "keys"))
    Path(generated[0]).write_text("existing-key", encoding="utf-8")

    repeated = keyfile.generate_key_files(
        str(template), param_table, str(tmp_path / "keys")
    )

    assert repeated == generated
    assert Path(generated[0]).read_text(encoding="utf-8") == "existing-key"


def test_line_replacement_changes_only_last_matching_token():
    line = "MOVCTL 3 1 0 0.0 1.0 0.0 1.0\n"
    rendered = keyfile.apply_parameters(
        [line], ["pressure_roll_constant_speed"], ["pressure_roll"], ["0.1"]
    )[0]
    assert rendered.split()[5] == "1.0"
    assert rendered.split()[-1] == keyfile.format_deform_float("0.1")


def test_constant_speed_atomic_ignores_function_movctl_header():
    line = "MOVCTL 3 1 2 0.0 1.0 0.0 20\n"
    assert keyfile.apply_parameters(
        [line], ["pressure_roll_constant_speed"], ["pressure_roll"], ["0.1"]
    ) == [line]


def test_read_key_frames(tmp_path):
    f1 = tmp_path / "a.KEY"
    f2 = tmp_path / "b.KEY"
    f1.write_text("line1\nline2\n", encoding="utf-8")
    f2.write_text("only\n", encoding="utf-8")
    frames = keyfile.read_key_frames([str(f1), str(f2)])
    assert len(frames) == 2
    assert frames[0] == ["line1\n", "line2\n"]
    assert frames[1] == ["only\n"]


# ---- 碾环 MOVCTL 速度多行块等比例缩放 ----

def test_scale_abs_maps_range_and_preserves_sign():
    # 原绝对值范围 [0.2, 2.3] 映射到 [0.5, 1.5]
    assert scale_abs(2.3, 0.2, 2.3, 0.5, 1.5) == pytest.approx(1.5)   # 最大 -> upper
    assert scale_abs(0.2, 0.2, 2.3, 0.5, 1.5) == pytest.approx(0.5)   # 最小 -> lower
    # 负数：先按绝对值缩放再贴回符号；1.25 -> 0.5 + (1.25-0.2)/2.1 = 1.0
    assert scale_abs(-1.25, 0.2, 2.3, 0.5, 1.5) == pytest.approx(-1.0)


def test_scale_abs_zero_span_maps_to_midpoint():
    # 原范围跨度为 0（所有点绝对值相同）-> 目标区间中点，避免除零
    assert scale_abs(0.8, 0.8, 0.8, 0.5, 1.5) == pytest.approx(1.0)
    assert scale_abs(-0.8, 0.8, 0.8, 0.5, 1.5) == pytest.approx(-1.0)


def test_scale_speed_line_replaces_only_speed_column():
    line = "    1.0000000000E+000    2.3000000000E+000\n"
    out = scale_speed_line(line, 0.2, 2.3, 0.5, 1.5)
    # 时间列保留，速度列（最大绝对值）被映射到 upper=1.5
    assert out.split()[0] == "1.0000000000E+000"
    assert float(out.split()[1]) == pytest.approx(1.5)
    assert out.endswith("\n")


def test_scale_speed_line_ignores_unparseable():
    # 仅单个 token（无速度列）时原样返回
    single = "onlyonetoken\n"
    assert scale_speed_line(single, 0.2, 2.3, 0.5, 1.5) == single
    # 速度列非数值时原样返回
    bad = "    1.0000000000E+000    NaNaN\n"
    assert scale_speed_line(bad, 0.2, 2.3, 0.5, 1.5) == bad


def test_collect_speed_scale_specs_requires_both_bounds():
    # 只给 lower，不生成目标区间
    specs = keyfile._collect_speed_scale_specs(
        ["pressure_roll_speed_lower"], ["pressure_roll"], ["0.5"]
    )
    assert specs == {}
    # lower + upper 齐全 -> 生成区间（按对象 ID 聚合，取绝对值且 lower<=upper）
    specs = keyfile._collect_speed_scale_specs(
        ["pressure_roll_speed_upper", "pressure_roll_speed_lower"],
        ["pressure_roll", "pressure_roll"],
        ["-1.5", "0.5"],
    )
    assert specs == {"3": (0.5, 1.5)}


def test_scale_movctl_block_scales_m_lines_only():
    lines = [
        "MOVCTL       3       1       2    0.0    1.0    0.0       3\n",  # m=3
        "    1.0000000000E+000    2.2000000000E+000\n",   # abs 2.2 (max) -> 1.5
        "    2.0000000000E+000   -3.0000000000E-001\n",   # abs 0.3 (min) -> -0.5
        "    3.0000000000E+000    8.0000000000E-001\n",   # abs 0.8 -> 0.5+(0.8-0.3)/1.9
        "STROKE       3    0.0\n",                          # 块外，不动
    ]
    out = scale_movctl_block(lines, {"3": (0.5, 1.5)})
    assert out[0] == lines[0]                        # MOVCTL 头行（含行数 3）不变
    assert float(out[1].split()[1]) == pytest.approx(1.5)
    assert float(out[2].split()[1]) == pytest.approx(-0.5)
    assert float(out[3].split()[1]) == pytest.approx(0.5 + (0.8 - 0.3) / (2.2 - 0.3))
    assert out[4] == lines[4]                         # STROKE 行不动


def test_scale_movctl_block_noop_without_specs():
    lines = ["MOVCTL 3 1 2 0.0 1.0 0.0 3\n", "    1.0    2.2\n"]
    assert scale_movctl_block(lines, {}) is lines


def test_generate_key_files_scales_speed_block(tmp_path):
    template = tmp_path / "RING.KEY"
    template.write_text(
        "MOVCTL       3       1       2    0.0    1.0    0.0       3\n"
        "    1.0000000000E+000    2.2000000000E+000\n"
        "    2.0000000000E+000   -3.0000000000E-001\n"
        "    3.0000000000E+000    8.0000000000E-001\n"
        "STROKE       3    0.0\n",
        encoding="utf-8",
    )
    param_table = [
        ["pressure_roll_speed_lower", "pressure_roll_speed_upper"],
        ["pressure_roll", "pressure_roll"],
        ["0.5", "1.5"],
    ]
    generated = keyfile.generate_key_files(str(template), param_table, str(tmp_path / "out"))
    lines = open(generated[0], encoding="utf-8").read().splitlines()
    # 头行的控制点行数 3 未被误改
    assert lines[0].split()[-1] == "3"
    absv = [abs(float(lines[i].split()[1])) for i in (1, 2, 3)]
    # 缩放后绝对值应恰好铺满目标区间 [0.5, 1.5]
    assert min(absv) == pytest.approx(0.5)
    assert max(absv) == pytest.approx(1.5)
    # 负数保号
    assert float(lines[2].split()[1]) < 0


def test_apply_parameters_routes_custom_atomic_capability(monkeypatch):
    registry = ReplacementRegistry()
    calls = []

    def replace_custom(line, binding):
        calls.append((line, binding.name, binding.object_id, binding.value))
        if not line.startswith("CUSTOM"):
            return LineReplacement(line, False)
        return LineReplacement(f"CUSTOM {binding.object_id} {binding.value}\n", True)

    registry.register_fn("custom", replace_custom, kind="line")
    monkeypatch.setattr(keyfile, "replacement_registry", registry)

    result = keyfile.apply_parameters(
        ["CUSTOM 1 old\n", "UNCHANGED\n"],
        ["custom"], ["workpiece"], ["new"],
    )

    assert result == ["CUSTOM 1 new\n", "UNCHANGED\n"]
    assert calls[0][1:] == ("custom", "1", "new")

