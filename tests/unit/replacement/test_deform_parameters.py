import pytest

from mobo.replacement.base import ParameterBinding
from mobo.replacement.deform_parameters import (
    collect_speed_scale_specs,
    replace_keyword_last_value,
    replace_movctl_profile_peak_speed,
    rescale_movctl_speed_profile,
)


def _binding(name, object_id, value, object_name="pressure_roll"):
    return ParameterBinding(name, object_name, object_id, value)


def test_keyword_replacer_matches_keyword_and_object():
    line = "REFTMP       1    9.0000000000E+002\n"
    result = replace_keyword_last_value(
        line, _binding("roll_tmp", "1", "950", "workpiece"), keyword="REFTMP"
    )

    assert result.matched is True
    assert float(result.text.split()[-1]) == pytest.approx(950.0)


def test_keyword_replacer_reports_non_match_without_editing():
    line = "REFTMP       2    2.0000000000E+002\n"
    result = replace_keyword_last_value(
        line, _binding("roll_tmp", "1", "950", "workpiece"), keyword="REFTMP"
    )

    assert result.matched is False
    assert result.text == line


def test_speed_profile_requires_paired_bounds_for_same_object():
    bindings = [
        _binding("pressure_roll_speed_lower", "3", "0.5"),
        _binding("pressure_roll_speed_upper", "3", "1.5"),
        _binding("pressure_roll_speed_upper", "2", "9.0", "driving_roll"),
    ]

    assert collect_speed_scale_specs(bindings) == {"3": (0.5, 1.5)}


def test_speed_profile_replacer_handles_whole_movctl_block():
    lines = [
        "MOVCTL       3       1       2    0.0    1.0    0.0       2\n",
        "    1.0000000000E+000    2.0000000000E-001\n",
        "    2.0000000000E+000    2.3000000000E+000\n",
    ]
    bindings = [
        _binding("pressure_roll_speed_lower", "3", "0.5"),
        _binding("pressure_roll_speed_upper", "3", "1.5"),
    ]

    result = rescale_movctl_speed_profile(lines, bindings)

    assert float(result[1].split()[-1]) == pytest.approx(0.5)
    assert float(result[2].split()[-1]) == pytest.approx(1.5)


def test_profile_peak_speed_preserves_zero_speed_control_points():
    lines = [
        "MOVCTL       3       1       2    0.0    1.0    0.0       5\n",
        "    0.0000000000E+000    0.0000000000E+000\n",
        "    1.0000000000E+001    4.0000000000E-001\n",
        "    3.6250000000E+001    4.0000000000E-001\n",
        "    4.6250000000E+001    0.0000000000E+000\n",
        "    6.5000000000E+001    0.0000000000E+000\n",
    ]
    bindings = [_binding("pressure_roll_profile_peak_speed", "3", "2.5")]

    result = replace_movctl_profile_peak_speed(lines, bindings)

    assert [float(line.split()[-1]) for line in result[1:]] == [
        0.0, 2.5, 2.5, 0.0, 0.0,
    ]
