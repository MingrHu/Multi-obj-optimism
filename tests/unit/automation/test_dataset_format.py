"""数据集数值格式测试"""

import pytest

from mobo.automation.dataset_format import format_dataset_row, format_dataset_value


def test_dataset_value_truncates_to_two_decimal_places():
    assert format_dataset_value("5.389") == "5.38"
    assert format_dataset_value("-5.389") == "-5.38"
    assert format_dataset_value(5) == "5.00"
    assert format_dataset_value("0.009") == "0.00"


def test_dataset_row_preserves_missing_and_text_values():
    fields = format_dataset_row(["1.239", "nan", "label"])
    assert [field.strip() for field in fields] == ["1.239000", "nan", "label"]
    assert all(len(field) == 20 for field in fields)


def test_fixed_width_preserves_small_targets_and_tsv_column_positions():
    rows = [format_dataset_row(["320", "4440000.96", "0.001224"]),
            format_dataset_row(["450", "858012.06", "0.477424"])]
    assert rows[0][-1].strip() == "0.001224"
    assert len("\t".join(rows[0])) == len("\t".join(rows[1])) == 62
    assert [float(value) for value in "\t".join(rows[0]).split("\t")] == [320, 4440000.96, 0.001224]


def test_fixed_width_rejects_overflow_without_truncating_integer_digits():
    with pytest.raises(ValueError, match="固定宽度"):
        format_dataset_row(["123456789012345678901"])


def test_six_decimal_truncation_and_negative_zero():
    assert format_dataset_row(["-0.0000009", "-1.23456789"])[0].strip() == "0.000000"
    assert format_dataset_row(["-1.23456789"])[0].strip() == "-1.234567"
    with pytest.raises(ValueError, match="非负"):
        format_dataset_value(1, decimal_places=-1)
