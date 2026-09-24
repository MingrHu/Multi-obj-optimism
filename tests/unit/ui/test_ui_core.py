from __future__ import annotations

import json
import math
from datetime import timedelta, timezone
from pathlib import Path

import pytest

from work_platform.mobo_ui import core


class _Response:
    def __init__(self, value):
        self._body = json.dumps(value).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self._body


def test_rows_by_fields_transposes_columns():
    assert core.rows_by_fields({"x": [1, 2], "y": [3, 4]}, ["x", "y"]) == [[1, 3], [2, 4]]


def test_rows_by_fields_rejects_mismatched_columns():
    with pytest.raises(ValueError, match="长度不一致"):
        core.rows_by_fields({"x": [1], "y": [2, 3]}, ["x", "y"])


def test_read_tabular_file_supports_header_and_no_header(tmp_path: Path):
    header_file = tmp_path / "header.csv"
    header_file.write_text("temperature,load\n900,12.5\n", encoding="utf-8")
    assert core.read_tabular_file(header_file) == (["temperature", "load"], [[900.0, 12.5]])

    plain_file = tmp_path / "plain.tsv"
    plain_file.write_text("1\t2\n3\t4\n", encoding="utf-8")
    assert core.read_tabular_file(plain_file) == (["字段 1", "字段 2"], [[1.0, 2.0], [3.0, 4.0]])


def test_api_client_builds_repeated_fields_query(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return _Response({"code": 0, "message": "ok", "data": {"row_count": 1}})

    monkeypatch.setattr(core, "urlopen", fake_urlopen)
    result = core.ApiClient("http://localhost:5000", timeout=3).get_data(
        "doe-1", "tos-123", ["temperature", "load"]
    )
    assert result == {"row_count": 1}
    assert "fields=temperature" in captured["url"]
    assert "fields=load" in captured["url"]
    assert captured["timeout"] == 3


def test_api_client_surfaces_business_error(monkeypatch):
    monkeypatch.setattr(
        core,
        "urlopen",
        lambda *_args, **_kwargs: _Response({"code": 1, "message": "参数不合法", "data": {}}),
    )
    with pytest.raises(core.ApiError, match="参数不合法"):
        core.ApiClient().health()


def test_api_client_generates_training_dataset(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Response({"code": 0, "message": "ok", "data": {"resource_id": "tos-demo"}})

    monkeypatch.setattr(core, "urlopen", fake_urlopen)
    payload = {"id": "demo", "n_samples": 80}
    result = core.ApiClient().generate_training_dataset(payload)
    assert captured["url"].endswith("/api/v1/hust/doe/dataset/generate")
    assert captured["body"] == payload
    assert result["resource_id"] == "tos-demo"


def test_api_client_saves_training_dataset_definition(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Response({"code": 0, "message": "ok", "data": {"sample_count": 2}})

    monkeypatch.setattr(core, "urlopen", fake_urlopen)
    payload = {"id": "demo", "data_source": {"input_data": {}, "output_data": {}}}
    result = core.ApiClient().save_training_dataset(payload)
    assert captured["url"].endswith("/api/v1/hust/doe/dataset/save")
    assert captured["body"] == payload
    assert result["sample_count"] == 2


def test_api_client_deletes_doe(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Response({"code": 0, "message": "ok", "data": "doe-1"})

    monkeypatch.setattr(core, "urlopen", fake_urlopen)

    result = core.ApiClient().delete_doe("doe-1")

    assert captured["url"].endswith("/api/v1/doe/delete")
    assert captured["body"] == {"id": "doe-1"}
    assert result == {"value": "doe-1"}


def test_status_view_has_safe_fallback():
    assert core.status_view("finished").tone == "success"
    assert core.status_view("future_state").text == "future_state"
    assert core.stage_text("not_started") == "未开始"
    assert core.stage_text("evaluating") == "交叉验证"


def test_format_timestamp_converts_aware_values_and_preserves_naive_wall_time():
    china = timezone(timedelta(hours=8))

    assert core.format_timestamp("2026-09-10T06:19:03+00:00", china) == "2026-09-10 14:19:03"
    assert core.format_timestamp("2026-09-10 14:19:03", china) == "2026-09-10 14:19:03"
    assert core.format_timestamp(None, china) == "—"


def test_prepare_numeric_table_drops_all_nan_column_and_incomplete_rows():
    cleaned = core.prepare_numeric_table(
        ["temperature", "load", "empty_target"],
        [
            [900, 100, math.nan],
            [950, "missing", math.nan],
            [1000, 120, math.inf],
        ],
    )

    assert cleaned.headers == ["temperature", "load"]
    assert cleaned.rows == [[900.0, 100.0], [1000.0, 120.0]]
    assert cleaned.dropped_columns == ["empty_target"]
    assert cleaned.dropped_row_count == 1


def test_numeric_fields_exclude_non_finite_only_columns():
    assert core.numeric_field_names(
        ["x", "nan_column", "flag"],
        [[1, math.nan, "true"], [2, math.inf, "false"]],
    ) == ["x", "flag"]
