from dataclasses import replace
from pathlib import Path

import pytest

import mobo.automation.template_store as template_store
from mobo.automation.task_collection import (
    RING_7050_SINGLE_TASK_1,
    TC4_RING_MULTI_TASK_1,
)


def _key(path: Path, *, transition: bool = False) -> str:
    text = "REFTMP 1 300\n"
    if transition:
        text = (
            "* Data for Object # 1\n"
            "CNTRAX 1 0 0 0 0 0 1 0\n"
            "DIEGEO 1 1\n"
            "* Inter-Object Data\n"
        )
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_single_template_round_trip_and_delete(monkeypatch, tmp_path):
    storage = tmp_path / "templates"
    monkeypatch.setattr(template_store, "AUTOMATION_TEMPLATES_DIR", storage)
    definition = replace(
        RING_7050_SINGLE_TASK_1,
        task_id="custom-single",
        name="自定义单工步",
        workspace=tmp_path / "workspace",
        template_key=_key(tmp_path / "single.KEY"),
    )

    saved = template_store.save_template(definition)
    loaded = template_store.load_template("custom-single")

    assert saved == str(storage / "custom-single.json")
    assert loaded == definition
    assert template_store.list_templates(multi=False) == [definition]
    assert template_store.list_templates(multi=True) == []

    template_store.delete_template("custom-single")
    assert template_store.list_templates() == []


def test_multi_template_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(template_store, "AUTOMATION_TEMPLATES_DIR", tmp_path / "templates")
    operations = TC4_RING_MULTI_TASK_1.operation_configs()
    for index, operation in enumerate(operations, 1):
        operation["template_key"] = _key(
            tmp_path / f"{index}.KEY", transition=index > 1
        )
    definition = replace(
        TC4_RING_MULTI_TASK_1,
        task_id="custom-multi",
        name="自定义多工步",
        workspace=tmp_path / "workspace",
        operations=tuple(operations),
    )

    template_store.save_template(definition)

    assert template_store.load_template("custom-multi") == definition
    assert template_store.list_templates(multi=True) == [definition]


def test_template_ids_are_unique_and_builtins_are_read_only(monkeypatch, tmp_path):
    monkeypatch.setattr(template_store, "AUTOMATION_TEMPLATES_DIR", tmp_path / "templates")
    custom = replace(
        RING_7050_SINGLE_TASK_1,
        task_id="unique-task",
        name="唯一模板",
        workspace=tmp_path / "workspace",
        template_key=_key(tmp_path / "single.KEY"),
    )
    template_store.save_template(custom)

    with pytest.raises(FileExistsError, match="已存在"):
        template_store.save_template(custom)
    template_store.save_template(replace(custom, name="更新名称"), overwrite=True)
    assert template_store.load_template("unique-task").name == "更新名称"

    builtin = replace(
        RING_7050_SINGLE_TASK_1,
        template_key=_key(tmp_path / "builtin.KEY"),
    )
    with pytest.raises(ValueError, match="内置"):
        template_store.save_template(builtin)
    with pytest.raises(ValueError, match="内置"):
        template_store.delete_template(builtin.task_id)


def test_template_validation_rejects_unknown_objects_and_invalid_transition_key(tmp_path):
    parameter = dict(RING_7050_SINGLE_TASK_1.parameters[0])
    parameter["object"] = "unknown_object"
    invalid_single = replace(
        RING_7050_SINGLE_TASK_1,
        task_id="invalid-single",
        template_key=_key(tmp_path / "single.KEY"),
        parameters=(parameter,),
    )
    with pytest.raises(ValueError, match="未知对象"):
        template_store.validate_template(invalid_single)

    operations = TC4_RING_MULTI_TASK_1.operation_configs()
    for index, operation in enumerate(operations, 1):
        operation["template_key"] = _key(tmp_path / f"invalid-{index}.KEY")
    invalid_multi = replace(
        TC4_RING_MULTI_TASK_1,
        task_id="invalid-multi",
        operations=tuple(operations),
    )
    with pytest.raises(ValueError, match="换模所需"):
        template_store.validate_template(invalid_multi)


def test_template_listing_skips_a_corrupt_file(monkeypatch, tmp_path):
    storage = tmp_path / "templates"
    storage.mkdir()
    (storage / "broken.json").write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(template_store, "AUTOMATION_TEMPLATES_DIR", storage)

    assert template_store.list_templates() == []
