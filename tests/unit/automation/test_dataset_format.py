"""数据集数值格式测试"""

from mobo.automation.dataset_format import format_dataset_row, format_dataset_value


def test_dataset_value_truncates_to_two_decimal_places():
    assert format_dataset_value("5.389") == "5.38"
    assert format_dataset_value("-5.389") == "-5.38"
    assert format_dataset_value(5) == "5.00"
    assert format_dataset_value("0.009") == "0.00"


def test_dataset_row_preserves_missing_and_text_values():
    assert format_dataset_row(["1.239", "nan", "label"]) == [
        "1.23", "nan", "label",
    ]
