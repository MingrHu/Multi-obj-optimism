"""环轧六参数只修改目标字段，不改变网格、材料、参考温度和终止条件。"""

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
             "0 0", "10 -0.6", "36.25 -0.6", "46.25 0", "65 0"]
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
    assert len(actual) == len(lines)
    changes = [(a, b) for a, b in zip(lines, actual, strict=True) if a != b]
    assert len(changes) == 19  # 原16行 + TMAX + 两个零速阶段的非零时间节点
    for before, after in changes:
        if before.split()[0] in {"10", "36.25", "46.25", "65"}:
            assert float(after.split()[0]) == pytest.approx(float(before.split()[0]) * 0.32)
            assert float(after.split()[1]) == (-1.25 if float(before.split()[1]) else 0)
        else:
            assert before.rsplit(" ", 1)[0] == after.rsplit(" ", 1)[0]
        assert before.endswith(newline) and after.endswith(newline)
        assert before.split()[0] in {"NDTMP", "ANGMOV", "FRCFAC", "IHTCOF", "TMAX", "10", "36.25", "46.25", "65"}
    assert template.read_bytes() == "".join(lines).encode()
    recovered = extract_sample_from_key(generated[0], params, DeformConfig.OBJ_DEF)
    assert recovered == [Decimal(value) for value in values]
    for keyword in ("REFTMP", "SMAX", "PDIE", "STROKE", "DTMAX", "NSTEP"):
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
    ("10 -0.6\n", "10 0\n"),
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
def test_every_two_decimal_speed_preserves_14_5_mm_stroke(speed):
    lines = ring_key()
    result = apply_parameters(lines, ["pressure_roll_profile_peak_speed"], ["pressure_roll"], [str(speed)])
    start = next(i for i, line in enumerate(result) if line.startswith("MOVCTL 3 "))
    points = [tuple(map(float, row.split())) for row in result[start + 1:start + 6]]
    stroke = sum((abs(a[1]) + abs(b[1])) * (b[0] - a[0]) / 2
                 for a, b in zip(points, points[1:], strict=False))
    assert stroke == pytest.approx(14.5, abs=1e-8)
    assert [abs(point[1]) for point in points] == [0, speed, speed, 0, 0]
    assert [point[0] for point in points] == pytest.approx(
        [time * 0.4 / speed for time in [0, 10, 36.25, 46.25, 65]], rel=1e-9,
    )
    tmax = float(next(row for row in result if row.startswith("TMAX ")).split()[1])
    nstep = int(next(row for row in result if row.startswith("NSTEP ")).split()[1])
    assert tmax >= points[3][0]
    assert nstep >= tmax / 0.01
    # 允许变化仅限四个非零时间节点、两个速度值、TMAX和NSTEP。
    allowed = {0, 3, *range(start + 1, start + 6)}
    assert all(before == after for index, (before, after) in enumerate(zip(lines, result, strict=True))
               if index not in allowed)


@pytest.mark.parametrize("old,new", [
    ("PDIE 3 0", "PDIE 2 0"), ("SMAX 0 14.5 0", "SMAX 0 15 0"),
    ("STROKE 3 0 0 0", "STROKE 3 0 1 0"), ("NSTEP 10000", "NSTEP 0"),
    ("TMAX 65", "TMAX 0"), ("36.25 -0.6", "9 -0.6"),
])
def test_incompatible_stroke_control_is_rejected(old, new):
    with pytest.raises(ValueError):
        render("".join(ring_key()).replace(old, new).splitlines(keepends=True))


@pytest.mark.parametrize("speed", ["0", "-0.1"])
def test_nonpositive_feed_cannot_complete_stroke(speed):
    with pytest.raises(ValueError, match="正数"):
        render(ring_key(), ("1125", "300", speed, "1.5", "0.5", "4.5"))
