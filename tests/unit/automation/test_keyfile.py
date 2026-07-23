"""KEY 文件文本处理 (:mod:`mobo.automation.keyfile`) 测试。

使用合成模板验证 :func:`generate_key_files` 的核心语义：param_table 前两行为
表头（参数名 / 对象名），从第 2 行起每行是一个样本，为每个样本生成一个 KEY，
并按 (关键字, 对象ID) 定位替换目标行末尾的数值。
"""

import os

from mobo.automation import keyfile


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


def test_read_key_frames(tmp_path):
    f1 = tmp_path / "a.KEY"
    f2 = tmp_path / "b.KEY"
    f1.write_text("line1\nline2\n", encoding="utf-8")
    f2.write_text("only\n", encoding="utf-8")
    frames = keyfile.read_key_frames([str(f1), str(f2)])
    assert len(frames) == 2
    assert frames[0] == ["line1\n", "line2\n"]
    assert frames[1] == ["only\n"]
