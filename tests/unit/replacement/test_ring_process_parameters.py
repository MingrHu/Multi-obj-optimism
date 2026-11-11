"""环轧六参数只修改目标字段，不改变网格、材料和参考温度 严格跨过行程阈值。"""

from decimal import Decimal

import pytest

from mobo.automation.keyfile import apply_parameters, generate_key_files
from mobo.automation.task_collection import GH4169_RING_SINGLE_TASK_1
from mobo.utils.sample_recovery import extract_sample_from_key
from mobo.automation.config import DeformConfig


def ring_key(newline="\n"):
    rows = ["TMAX 65", "SMAX 0 14.5 0", "PDIE 3 0", "NSTEP 10000", "DTMAX 0.01 0",
            "STROKE 3 0 0 0", "* material and geometry unchanged"]
    for obj in range(1, 6):
        rows += [f"REFTMP {obj} 150", f"NDTMP {obj} 0 150"]
    rows += ["ANGMOV 2 1 0 1.2", "ANGMOV 4 1 0 0", "MOVCTL 3 1 2 0 1 0 5",
             "0 0", "10 0.4", "36.25 0.4", "46.25 0", "65 0"]
    for obj in range(2, 6):
        rows += [f"FRCFAC 1 {obj} 1 0 0.7", f"IHTCOF 1 {obj} 0 5"]
    rows += ["FRCFAC 2 3 2 0 0.2", "IHTCOF 2 3 0 8"]
    return [row + newline for row in rows]


def render(lines, values=("1125.25", "300.50", "1.25", "1.50", "0.50", "4.50")):
    params = GH4169_RING_SINGLE_TASK_1.parameters
    return apply_parameters(lines, [p["name"] for p in params],
                            [p["object"] for p in params], values)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_six_parameters_only_change_allowed_numeric_fields(tmp_path, newline):
    lines = ring_key(newline)
    template = tmp_path / "ring.KEY"
    template.write_bytes("".join(lines).encode())
    params = GH4169_RING_SINGLE_TASK_1.parameters
    values = ["1125.25", "300.50", "1.25", "1.50", "0.50", "4.50"]
    generated = generate_key_files(str(template), [
        [p["name"] for p in params], [p["object"] for p in params], values,
    ], str(tmp_path / "keys"))
    from pathlib import Path
    output = Path(generated[0]).read_bytes().decode()
    actual = output.splitlines(keepends=True)
    assert len(actual) == len(lines) - 1
    start = next(i for i, row in enumerate(lines) if row.startswith("MOVCTL 3 "))
    assert actual[start].split()[-1] == "4"
    assert all(row.endswith(newline) for row in actual)
    # 删除对应控制块后逐行比较 禁止误改非目标材料和几何字段
    before_rows = lines[:start] + lines[start + 6:]
    after_rows = actual[:start] + actual[start + 5:]
    for before, after in zip(before_rows, after_rows, strict=True):
        if before != after:
            assert before.split()[0] in {"NDTMP", "ANGMOV", "FRCFAC", "IHTCOF", "TMAX", "SMAX"}
    assert float(next(row for row in actual if row.startswith("SMAX ")).split()[2]) == 14.5001
    assert template.read_bytes() == "".join(lines).encode()
    recovered = extract_sample_from_key(generated[0], params, DeformConfig.OBJ_DEF)
    assert recovered == [Decimal(value) for value in values]
    for keyword in ("REFTMP", "PDIE", "STROKE", "DTMAX", "NSTEP"):
        assert [row for row in actual if row.startswith(keyword + " ")] == [
            row for row in lines if row.startswith(keyword + " ")
        ]


def test_explicit_nodal_temperatures_change_without_changing_count_or_other_columns():
    source = ["REFTMP 1 20\r\n", "NDTMP 1 2 150 1\r\n",
              "   1    160\r\n", "   2    170\r\n", "RZ 1 2\r\n"]
    result = apply_parameters(source, ["workpiece_temperature"], ["workpiece"], ["1125"])
    assert result[0] == source[0] and result[-1] == source[-1]
    assert result[1].split()[2:] == ["2", "1.1250000000E+003", "1"]
    assert all(float(row.split()[1]) == 1125 for row in result[2:4])


@pytest.mark.parametrize("old,new", [
    ("FRCFAC 1 3 1 0 0.7", "FRCFAC 1 3 2 0 0.7"),
    ("IHTCOF 1 3 0 5", "IHTCOF 1 3 1 5"),
    ("ANGMOV 2 1 0 1.2", "ANGMOV 2 0 0 1.2"),
    ("NDTMP 4 0 150\n", ""),
    ("FRCFAC 1 4 1 0 0.7\n", ""),
    ("MOVCTL 3 1 2 0 1 0 5", "MOVCTL 3 1 0 0 1 0 5"),
    ("10 0.4\n", "10 0\n"),
])
def test_unsupported_or_missing_target_fails_before_writing(old, new):
    lines = "".join(ring_key()).replace(old, new).splitlines(keepends=True)
    with pytest.raises(ValueError):
        render(lines)


def test_old_three_column_samples_are_rejected():
    with pytest.raises(ValueError, match="样本列数"):
        render(ring_key(), ("1125", "300", "1.25"))


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_parameters_are_rejected(value):
    with pytest.raises(ValueError, match="有限"):
        render(ring_key(), (value, "300", "1.25", "1.5", "0.5", "4.5"))


@pytest.mark.parametrize("speed", [value / 100 for value in range(10, 251)])
def test_every_two_decimal_speed_has_single_trapezoid_area_and_stop_margin(speed):
    lines = ring_key()
    result = apply_parameters(lines, ["pressure_roll_profile_peak_speed"], ["pressure_roll"], [str(speed)])
    start = next(i for i, line in enumerate(result) if line.startswith("MOVCTL 3 "))
    points = [tuple(map(float, row.split())) for row in result[start + 1:start + 5]]
    main_stroke = sum((a[1] + b[1]) * (b[0] - a[0]) / 2
                      for a, b in zip(points[:4], points[1:4], strict=False))
    total_stroke = sum((a[1] + b[1]) * (b[0] - a[0]) / 2
                       for a, b in zip(points, points[1:], strict=False))
    assert main_stroke == pytest.approx(14.513, abs=1e-8)
    assert total_stroke == pytest.approx(14.513, abs=1e-8)
    assert [point[1] for point in points] == pytest.approx([0, speed, speed, 0])
    assert [point[0] for point in points] == pytest.approx(
        [distance / speed for distance in [0, 4, 14.513, 18.513]], rel=1e-9,
    )
    tmax = float(next(row for row in result if row.startswith("TMAX ")).split()[1])
    nstep = int(next(row for row in result if row.startswith("NSTEP ")).split()[1])
    assert tmax == points[-1][0]
    assert nstep >= tmax / 0.01
    assert float(next(row for row in result if row.startswith("SMAX ")).split()[2]) == 14.5001
    assert result[start + 5:] == lines[start + 6:]
    assert all(before == after for index, (before, after) in enumerate(zip(lines[:start], result[:start], strict=True))
               if index not in {0, 1, 3})
    # 同参数重复渲染不得继续增加补进点或步数
    repeated = apply_parameters(result, ["pressure_roll_profile_peak_speed"], ["pressure_roll"], [str(speed)])
    assert repeated == result
    # 停止阈值在减速段内达到 仍有正速度 且先于保护时间
    import math
    stop_time = (18.513 - math.sqrt(8 * (14.513 - 14.5001))) / speed
    assert points[2][0] < stop_time < tmax
    assert (tmax - stop_time) / 0.01 > 12


@pytest.mark.parametrize("old,new", [
    ("PDIE 3 0", "PDIE 2 0"), ("SMAX 0 14.5 0", "SMAX 0 15 0"),
    ("STROKE 3 0 0 0", "STROKE 3 0 1 0"), ("NSTEP 10000", "NSTEP 0"),
    ("TMAX 65", "TMAX 0"), ("36.25 0.4", "9 0.4"),
])
def test_incompatible_stroke_control_is_rejected(old, new):
    with pytest.raises(ValueError):
        render("".join(ring_key()).replace(old, new).splitlines(keepends=True))


@pytest.mark.parametrize("speed", ["0", "-0.1"])
def test_nonpositive_feed_cannot_complete_stroke(speed):
    with pytest.raises(ValueError, match="正数"):
        render(ring_key(), ("1125", "300", speed, "1.5", "0.5", "4.5"))


@pytest.mark.parametrize("speed", ["-0.4", "nan", "inf"])
def test_invalid_template_speed_is_rejected(speed):
    lines = "".join(ring_key()).replace("10 0.4", f"10 {speed}").splitlines(keepends=True)
    with pytest.raises(ValueError):
        render(lines)


def test_existing_old_key_is_not_reused_or_overwritten(tmp_path):
    template = tmp_path / "ring.KEY"
    template.write_text("".join(ring_key()))
    keys = tmp_path / "keys"
    keys.mkdir()
    old = keys / "ring0.KEY"
    old.write_bytes(template.read_bytes())
    db = keys / "ring0.DB"
    db.write_bytes(b"old simulation")
    with pytest.raises(ValueError, match="新.*工作区"):
        generate_key_files(str(template), [["pressure_roll_profile_peak_speed"], ["pressure_roll"], ["0.4"]], str(keys))
    assert old.read_bytes() == template.read_bytes()
    assert db.read_bytes() == b"old simulation"


def test_new_profile_can_be_used_for_another_speed():
    first = render(ring_key(), ("1125", "300", "0.4", "1.5", "0.5", "4.5"))
    second = render(first, ("1125", "300", "0.1", "1.5", "0.5", "4.5"))
    assert float(next(row for row in second if row.startswith("TMAX ")).split()[1]) == pytest.approx(185.13)
    assert int(next(row for row in second if row.startswith("NSTEP ")).split()[1]) == 40000


def test_legacy_six_point_input_loses_recovery_segment():
    source = "".join(ring_key()).replace("SMAX 0 14.5 0", "SMAX 0 14.5001 0")
    source = source.replace("MOVCTL 3 1 2 0 1 0 5", "MOVCTL 3 1 2 0 1 0 6")
    source = source.replace("65 0", "46.5 0.008\n48 0.008")
    result = render(source.splitlines(keepends=True), ("1125", "300", "0.4", "1.5", "0.5", "4.5"))
    start = next(i for i, row in enumerate(result) if row.startswith("MOVCTL 3 "))
    assert result[start].split()[-1] == "4"
    assert [float(row.split()[1]) for row in result[start + 1:start + 5]] == [0, .4, .4, 0]
    assert result[start + 5].startswith("FRCFAC")


def test_current_four_point_input_with_wrong_area_is_rejected():
    lines = render(ring_key())
    start = next(i for i, row in enumerate(lines) if row.startswith("MOVCTL 3 "))
    lines[start + 3] = "11.6 1.25\n"
    with pytest.raises(ValueError, match="面积14.513"):
        render(lines)


def test_existing_six_point_workspace_is_rejected(tmp_path):
    template = tmp_path / "ring.KEY"
    template.write_text("".join(ring_key()))
    keys = tmp_path / "keys"
    keys.mkdir()
    old = keys / "ring0.KEY"
    old.write_text("".join(ring_key()).replace("MOVCTL 3 1 2 0 1 0 5", "MOVCTL 3 1 2 0 1 0 6")
                   .replace("65 0", "46.5 0.008\n48 0.008").replace("SMAX 0 14.5 0", "SMAX 0 14.5001 0"))
    before = old.read_bytes()
    with pytest.raises(ValueError, match="新的工作区"):
        generate_key_files(str(template), [["pressure_roll_profile_peak_speed"], ["pressure_roll"], ["0.4"]], str(keys))
    assert old.read_bytes() == before
