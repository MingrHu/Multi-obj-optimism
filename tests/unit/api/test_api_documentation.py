"""对接口文档中的示例和轮次字段执行轻量回归检查"""

import json
import re
from pathlib import Path

from mobo.api import service, store
from mobo.api.app import create_app
from mobo.common.paths import PROJECT_DIR


DOCUMENT = Path(PROJECT_DIR) / "docs" / "api" / "DOE_HTTP_API.md"


def _examples(section):
    return [json.loads(block) for block in re.findall(r"```json\s*\n(.*?)\n```", section, re.S)]


def _section(number):
    text = DOCUMENT.read_text(encoding="utf-8")
    return re.split(r"\n## ", text.split(f"## {number} ", 1)[1], maxsplit=1)[0]


def test_document_json_and_all_routes():
    document = DOCUMENT.read_text(encoding="utf-8")
    assert _examples(document)
    for rule in create_app({"TESTING": True}).url_map.iter_rules():
        if rule.endpoint == "static":
            continue
        for method in rule.methods - {"HEAD", "OPTIONS"}:
            assert f"{method} {rule.rule}" in document


def test_documented_optimization_submission_matches_service(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DOE_TASKS_DIR", tmp_path / "doe_tasks")
    store.create({"id": "document_check"})
    example = next(item for item in _examples(_section(9)) if item.get("message") == "优化任务已提交")
    expected = example["data"]
    monkeypatch.setattr(service, "_normalize_optimization", lambda payload, state: {
        "requested_mode": expected["mode"], "optimizer": "rl",
        "model_id": expected["model_id"], "objective_names": expected["objectives"],
    })
    monkeypatch.setattr(service.registry, "running", lambda *args: False)
    monkeypatch.setattr(service.registry, "start", lambda *args: None)
    response = service.start_optimization({"id": "document_check"})
    assert response.keys() == expected.keys()
    assert response["run_id"].startswith("run_")
    for request in (item for item in _examples(_section(9)) if "mode" in item):
        assert request["training_run_id"].startswith("train_")


def test_documented_normalized_history_matches_objective_contract():
    history = next(item for item in _examples(_section(11)) if "run_id" in item)
    request = history["request"]
    objectives = [{"name": name, "direction": "min"} for name in request["objective_names"]]
    _, actual = service._normalize_optimization_objectives(
        objectives, request["objective_names"], "multi",
    )
    assert request["objective_config"] == actual


def test_documented_curves_are_single_aligned_series():
    examples = _examples(_section(12))
    curves = [item["data"] if item.get("code") == 0 else item for item in examples]
    curves = [item for item in curves if "x" in item]
    assert {item["mode"] for item in curves} == {"single", "multi", "reinforcement_learning"}
    for data in curves:
        assert len(data["x"]) == len(data["y"])
        if "point_count" in data:
            assert len(data["x"]) == data["point_count"]
        assert all(value is None or isinstance(value, (int, float)) for value in data["y"])
        assert data["direction"] in {"min", "max", None}


def test_documented_dataset_responses_match_service(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DOE_TASKS_DIR", tmp_path / "doe_tasks")
    for number, action, message in (
        (3, service.save_training_dataset, "DOE 配置与训练数据已保存"),
        (16, service.generate_training_dataset, "训练数据集生成完成"),
    ):
        examples = _examples(_section(number))
        payload = next(item for item in examples if "id" in item)
        expected = next(item["data"] for item in examples if item.get("message") == message)
        if not store.list_all():
            store.create({"id": payload["id"]})
        actual = action(payload)
        assert actual.keys() == expected.keys()
        for field in expected.keys() - {"resource_id"}:
            assert actual[field] == expected[field]


def test_documented_resource_row_count_matches_values():
    response = next(item for item in _examples(_section(2)) if item.get("code") == 0)
    data = response["data"]
    assert all(len(values) == data["row_count"] for values in data["values"].values())
