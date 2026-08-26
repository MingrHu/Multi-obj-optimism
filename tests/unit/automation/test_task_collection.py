"""可导入多工步任务集合测试。"""

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import mobo.automation.task_collection as task_collection
from mobo.automation.task_collection import (
    GH4169_RING_SINGLE_TASK_1,
    RING_7050_SINGLE_TASK_1,
    TC4_RING_MULTI_TASK_1,
    get_multi_operation_task_definition,
    get_single_operation_task_definition,
)


def _small_template(path: Path) -> str:
    path.write_text(
        "* Data for Object # 1\n"
        "REFTMP 1 950\n"
        "CNTRAX 1 0 0 0 0 0 1 0\n"
        "DIEGEO 1 1\n"
        "1 2 3\n"
        "NDTMP 1 0 950\n"
        "* Data for Object # 2\n"
        "REFTMP 2 250\n"
        "DIEGEO 2 1\n"
        "1 2 3\n"
        "* Data for Object # 3\n"
        "REFTMP 3 250\n"
        "MOVCTL 3 1 0 0 1 0 1.0\n"
        "DIEGEO 3 1\n"
        "1 2 3\n"
        "* Data for Object # 4\n"
        "REFTMP 4 250\n"
        "DIEGEO 4 1\n"
        "1 2 3\n"
        "* Data for Object # 5\n"
        "REFTMP 5 250\n"
        "DIEGEO 5 1\n"
        "1 2 3\n"
        "* Inter-Object Data\n",
        encoding="utf-8",
    )
    return str(path)


def test_tc4_task_is_registered_and_has_fixed_first_operation():
    task = get_multi_operation_task_definition("tc4-ring-multi-task-1")
    assert task is TC4_RING_MULTI_TASK_1
    assert task.name == "TC4碾环多工步任务1"
    assert task.operations[0]["parameters"] == []
    assert [len(operation["parameters"]) for operation in task.operations] == [0, 3, 3]
    assert all(operation.get("inherit_materials") is True
               for operation in task.operations[1:])
    assert all(operation.get("enable_grain") is True
               for operation in task.operations[1:])
    assert task.workspace.parts[-2:] == ("mult", "tc4_ring_multi_task_1")
    grain_target = next(
        target for target in task.targets
        if target.output_name == "average_grain_size"
    )
    assert grain_target.target_name == "grain_morph"
    assert grain_target.object_name == "workpiece"
    assert grain_target.operation_indices == (3,)
    assert grain_target.select_component == 1
    load_target = next(
        target for target in task.targets if target.output_name == "die_load_y"
    )
    assert load_target.operation_indices == (1, 2, 3)
    assert load_target.select_component == 1
    assert load_target.in_progress is True


def test_gh4169_single_task_is_registered_with_requested_design_space():
    task = get_single_operation_task_definition("gh4169-ring-single-task-1")
    assert task is GH4169_RING_SINGLE_TASK_1
    assert task.workspace.parts[-2:] == ("single", "gh4169_ring_single_task_1")
    assert [(item["name"], item["range"]) for item in task.parameters] == [
        ("workpiece_temperature", [1100.0, 1150.0]),
        ("ring_die_temperature", [250.0, 350.0]),
        ("pressure_roll_constant_speed", [0.1, 2.5]),
    ]
    assert [target.output_name for target in task.targets] == [
        "roundness_inner", "roundness_outer", "die_load_y",
        "effective_strain_std", "average_grain_size", "material_fill",
    ]
    load_target = next(
        target for target in task.targets if target.output_name == "die_load_y"
    )
    assert load_target.target_name == "load"
    assert load_target.object_name == "driving_roll"
    assert load_target.operation_indices == (1,)
    assert load_target.select_component == 1
    assert load_target.in_progress is True
    assert task.targets[-1].verified is False


def test_7050_single_task_remains_registered():
    task = get_single_operation_task_definition("7050-ring-single-task-1")
    assert task is RING_7050_SINGLE_TASK_1
    assert [(item["name"], item["range"]) for item in task.parameters] == [
        ("workpiece_temperature", [320.0, 450.0]),
        ("ring_die_temperature", [150.0, 250.0]),
        ("pressure_roll_constant_speed", [0.2, 2.0]),
    ]


def test_typed_task_getters_reject_the_wrong_task_kind():
    assert get_multi_operation_task_definition(
        "tc4-ring-multi-task-1"
    ) is TC4_RING_MULTI_TASK_1
    assert get_single_operation_task_definition(
        "gh4169-ring-single-task-1"
    ) is GH4169_RING_SINGLE_TASK_1
    with pytest.raises(TypeError, match="不是多工步任务"):
        get_multi_operation_task_definition("gh4169-ring-single-task-1")
    with pytest.raises(TypeError, match="不是单工步任务"):
        get_single_operation_task_definition("tc4-ring-multi-task-1")


@pytest.mark.integration
def test_real_7050_template_has_grain_and_generates_parameterized_key(tmp_path):
    task = RING_7050_SINGLE_TASK_1
    task.validate()
    template = Path(task.template_key)
    text = template.read_text(encoding="utf-8")
    assert "TRANS        1       1       0       0       1" in text
    assert "GRAIN        1   23400      16" in text
    grain_start = text.index("GRAIN        1   23400      16")
    assert text[grain_start:].splitlines()[1].split()[2:4] == [
        "5.0000000000E+001", "5.0000000000E+001",
    ]

    sample = tmp_path / "sample.txt"
    sample.write_text("320\t150\t0.2\n", encoding="utf-8")
    generated = Path(task.prepare_keys(sample, workspace=tmp_path / "run")[0])
    rendered = generated.read_text(encoding="utf-8")
    assert "REFTMP       1    3.2000000000E+002" in rendered
    assert "NDTMP        1       0    3.2000000000E+002" in rendered
    for object_id in (2, 3, 4, 5):
        assert f"REFTMP       {object_id}    1.5000000000E+002" in rendered
    assert (
        "MOVCTL       3       1       0    0.0000000000E+000    "
        "1.0000000000E+000    0.0000000000E+000    2.0000000000E-001"
    ) in rendered


@pytest.mark.integration
def test_real_gh4169_template_has_grain_and_generates_parameterized_key(tmp_path):
    task = GH4169_RING_SINGLE_TASK_1
    task.validate()
    template = Path(task.template_key)
    text = template.read_text(encoding="utf-8")
    assert "TRANS        1       1       0       0       1" in text
    assert "4.6590000000E-003" in text
    assert "GRAIN        1   14400      16" in text
    grain_start = text.index("GRAIN        1   14400      16")
    assert text[grain_start:].splitlines()[1].split()[2:4] == [
        "5.0000000000E+001", "5.0000000000E+001",
    ]
    assert text[grain_start:].splitlines()[1 + (14400 - 1) * 4].split()[2:4] == [
        "5.0000000000E+001", "5.0000000000E+001",
    ]

    sample = tmp_path / "sample.txt"
    sample.write_text("1100\t250\t0.1\n", encoding="utf-8")
    generated = Path(task.prepare_keys(sample, workspace=tmp_path / "run")[0])
    rendered = generated.read_text(encoding="utf-8")
    assert "REFTMP       1    1.1000000000E+003" in rendered
    assert "NDTMP        1       0    1.1000000000E+003" in rendered
    for object_id in (2, 3, 4, 5):
        assert f"REFTMP       {object_id}    2.5000000000E+002" in rendered
    assert (
        "MOVCTL       3       1       0    0.0000000000E+000    "
        "1.0000000000E+000    0.0000000000E+000    1.0000000000E-001"
    ) in rendered


def test_task_prepare_uses_temp_workspace_and_never_calls_solver(monkeypatch, tmp_path):
    monkeypatch.setattr(
        task_collection, "task_dir", lambda task_id: tmp_path / "tasks" / task_id
    )
    templates = tuple(_small_template(tmp_path / f"{index}.KEY") for index in (1, 2, 3))
    operations = TC4_RING_MULTI_TASK_1.operation_configs()
    for operation, template in zip(operations, templates, strict=True):
        operation["template_key"] = template
    task = replace(
        TC4_RING_MULTI_TASK_1,
        workspace=tmp_path / "workspace",
        operations=tuple(operations),
    )
    sample = tmp_path / "sample.txt"
    sample.write_text("850\t250\t0.5\t900\t300\t1.5\n", encoding="utf-8")
    paths = task.prepare_keys(sample, work_dir=tmp_path / "prepared")
    assert len(paths) == 3
    assert Path(paths[0]).read_bytes() == Path(templates[0]).read_bytes()
    assert [Path(path).name for path in paths] == [
        "1_parameterized.KEY", "2_parameterized.KEY", "3_parameterized.KEY",
    ]
    assert Path(paths[0]).parts[-3:-1] == ("0", "op1")
    assert (tmp_path / "tasks" / task.task_id / "multi_operation_state.json").exists()


def test_multi_operation_dataset_has_no_header(monkeypatch, tmp_path):
    monkeypatch.setattr(
        type(TC4_RING_MULTI_TASK_1),
        "extract_targets",
        lambda self, files: {"load": "12.30", "grain": "50.10"},
    )
    monkeypatch.setattr(
        type(TC4_RING_MULTI_TASK_1),
        "_completed_key_files",
        lambda self, task, sample_index: {
            index: [str(tmp_path / f"{index}.KEY")] for index in range(1, 4)
        },
    )
    operations = {
        str(index): {"terminal_key": str(tmp_path / f"{index}.KEY")}
        for index in range(1, 4)
    }
    task = SimpleNamespace(
        samples=[["800", "200", "0.1", "900", "300", "1.0"]],
        state={"samples": {"0": {"status": "completed", "operations": operations}}},
    )
    output = TC4_RING_MULTI_TASK_1.extract_dataset(task, result_dir=tmp_path)
    assert Path(output).read_text(encoding="utf-8") == (
        "800\t200\t0.1\t900\t300\t1.0\t12.30\t50.10\n"
    )


def test_multi_operation_progress_targets_export_checkpoint_saved_steps(
        monkeypatch, tmp_path):
    operations = {}
    for index in range(1, 4):
        checkpoint = tmp_path / f"checkpoint_{index}.DB"
        checkpoint.touch()
        operations[str(index)] = {
            "terminal_key": str(tmp_path / f"terminal_{index}.KEY"),
            "checkpoint": str(checkpoint),
        }
    task = SimpleNamespace(state={"samples": {"0": {"operations": operations}}})
    calls = []

    def fake_export(db_path, output_dir):
        calls.append((db_path, output_dir))
        return [f"{output_dir}/step_1.KEY", f"{output_dir}/step_60.KEY"]

    monkeypatch.setattr(task_collection, "export_saved_step_keys", fake_export)
    result = TC4_RING_MULTI_TASK_1._completed_key_files(task, 0)

    assert len(calls) == 3
    assert all(len(result[index]) == 2 for index in range(1, 4))


@pytest.mark.integration
def test_real_tc4_templates_can_generate_one_parameterized_sample(monkeypatch, tmp_path):
    """使用真实三工步模板验证 KEY 生成，产物仅写入 tmp_path。"""
    monkeypatch.setattr(
        task_collection, "task_dir", lambda task_id: tmp_path / "tasks" / task_id
    )
    TC4_RING_MULTI_TASK_1.validate()
    sample = tmp_path / "sample.txt"
    sample.write_text("800\t200\t0.1\t960\t350\t2.2\n", encoding="utf-8")
    paths = [Path(path) for path in TC4_RING_MULTI_TASK_1.prepare_keys(
        sample, work_dir=tmp_path / "prepared"
    )]
    first_source = Path(TC4_RING_MULTI_TASK_1.operations[0]["template_key"])
    assert hashlib.sha256(paths[0].read_bytes()).digest() == hashlib.sha256(
        first_source.read_bytes()
    ).digest()
    stage2 = paths[1].read_text(encoding="utf-8")
    assert "REFTMP       1    8.0000000000E+002" in stage2
    assert "NDTMP        1       0    8.0000000000E+002" in stage2
    for object_id in (2, 3, 4, 5):
        assert f"REFTMP       {object_id}    2.0000000000E+002" in stage2
    assert (
        "MOVCTL       3       1       0    0.0000000000E+000    "
        "1.0000000000E+000    0.0000000000E+000    1.0000000000E-001"
    ) in stage2
    stage3 = paths[2].read_text(encoding="utf-8")
    assert "REFTMP       1    9.6000000000E+002" in stage3
    assert "NDTMP        1       0    9.6000000000E+002" in stage3
    for object_id in (2, 3, 4, 5):
        assert f"REFTMP       {object_id}    3.5000000000E+002" in stage3
    assert (
        "MOVCTL       3       1       0    0.0000000000E+000    "
        "1.0000000000E+000    0.0000000000E+000    2.2000000000E+000"
    ) in stage3
