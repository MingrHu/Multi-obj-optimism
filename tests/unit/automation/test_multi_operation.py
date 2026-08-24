"""多工步采样、KEY 拆分和磁盘续跑测试。"""

import hashlib
from pathlib import Path

import mobo.automation.multi_operation as multi_operation_module
import pytest
from mobo.automation.multi_operation import (
    MultiOperationTask,
    _has_grain_state,
    _prepare_transition_simulation,
    generate_multi_operation_samples,
    split_operation_key,
)


def _template(path: Path, center=(0, 0, 0)) -> Path:
    path.write_text(
        "* Data for Object # 1\n"
        "REFTMP 1 900\n"
        f"CNTRAX 1 {center[0]} {center[1]} {center[2]} 0 0 1 0\n"
        "MOVCTL 1 0\n"
        "DIEGEO 1 1\n"
        "1 2 3\n"
        "NDTMP 1 0 900\n"
        "* Data for Object # 2\n"
        "REFTMP 2 200\n"
        "DIEGEO 2 1\n"
        "4 5 6\n"
        "* Inter-Object Data\n"
        "IINUSR 1 2\n",
        encoding="utf-8",
    )
    return path


def _operations(tmp_path):
    first = _template(tmp_path / "1.KEY")
    second = _template(tmp_path / "2.KEY", (1, 2, 3))
    return [
        {"name": "first", "template_key": str(first), "parameters": [
            {"name": "temp", "object": "workpiece", "range": [900, 920]},
        ]},
        {"name": "second", "template_key": str(second), "parameters": [
            {"name": "temp", "object": "workpiece", "range": [800, 820]},
        ]},
    ]


def test_generate_multi_operation_samples_writes_only_sample_txt(tmp_path):
    operations = _operations(tmp_path)
    path = generate_multi_operation_samples(
        "multi", operations, str(tmp_path), method="full", level_nums=[2, 2]
    )
    rows = Path(path).read_text(encoding="utf-8").splitlines()
    assert len(rows) == 4
    assert Path(path).name == "multi-fullfactorial.txt"
    assert not Path(path + ".json").exists()


def test_shared_parameter_has_one_sample_column_and_multiple_objects(tmp_path):
    operations = _operations(tmp_path)
    operations[0]["parameters"] = []
    operations[1]["parameters"] = [
        {"name": "roll_tmp", "objects": ["driving_roll", "pressure_roll"],
         "range": [200, 350]},
    ]
    path = generate_multi_operation_samples(
        "shared", operations, str(tmp_path), method="full", level_nums=[2]
    )
    assert len(Path(path).read_text(encoding="utf-8").splitlines()) == 2
    assert not Path(path + ".json").exists()


def test_split_operation_key_keeps_source_and_excludes_workpiece_geometry(tmp_path):
    source = _template(tmp_path / "source.KEY")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    parts = split_operation_key(str(source), str(tmp_path / "parts"))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert "DIEGEO 1" not in Path(parts["object1_control"]).read_text(encoding="utf-8")
    assert "NDTMP 1 0 900" in Path(parts["object1_temperature"]).read_text(encoding="utf-8")
    assert "DIEGEO 2" in Path(parts["object2"]).read_text(encoding="utf-8")


def test_transition_simulation_inherits_materials_and_enables_grain(tmp_path):
    simulation = tmp_path / "simulation.KEY"
    simulation.write_text(
        "TRANS        1       1       0       0       0\n"
        "NSTEP       60\n"
        "*\n"
        "*  Property Data of Material     1\n"
        "*\n"
        "MTNAME       1\n"
        "GRNDAT       1       1\n",
        encoding="utf-8",
    )
    _prepare_transition_simulation(
        str(simulation), inherit_materials=True, enable_grain=True
    )
    text = simulation.read_text(encoding="utf-8")
    assert "TRANS        1       1       0       0       1" in text
    assert "NSTEP       60" in text
    assert "MTNAME" not in text
    assert "GRNDAT" not in text


def test_has_grain_state_requires_nonempty_workpiece_block(tmp_path):
    key = tmp_path / "terminal.KEY"
    key.write_text("GRAIN 1 20 16 0\n", encoding="utf-8")
    assert _has_grain_state(str(key)) is True
    key.write_text("GRAIN 1 0 16 0\n", encoding="utf-8")
    assert _has_grain_state(str(key)) is False


def test_prepare_parameterized_keys_keeps_fixed_stage_and_expands_shared_value(tmp_path):
    first = _template(tmp_path / "1.KEY")
    second = _template(tmp_path / "2.KEY", (1, 2, 3))
    second.write_text(
        second.read_text(encoding="utf-8").replace(
            "REFTMP 2 200\n",
            "REFTMP 2 200\nREFTMP 3 200\nMOVCTL 3 1 0 0 1 0 1.0\n",
        ),
        encoding="utf-8",
    )
    operations = [
        {"name": "fixed", "template_key": str(first), "parameters": []},
        {"name": "variable", "template_key": str(second), "parameters": [
            {"name": "workpiece_temperature", "object": "workpiece", "range": [800, 960]},
            {"name": "roll_tmp", "objects": ["driving_roll", "pressure_roll"],
             "range": [200, 350]},
            {"name": "pressure_roll_constant_speed", "object": "pressure_roll",
             "range": [0.1, 2.2]},
        ]},
    ]
    samples = tmp_path / "samples.txt"
    samples.write_text("880\t275\t1.5\n", encoding="utf-8")
    task = MultiOperationTask(
        "prepare", str(samples), operations, str(tmp_path / "runs"), dry_run=True
    )
    generated = [Path(path) for path in task.prepare_parameterized_keys()]
    assert generated[0].read_bytes() == first.read_bytes()
    text = generated[1].read_text(encoding="utf-8")
    assert "REFTMP 1 8.8000000000E+002" in text
    assert "NDTMP 1 0 8.8000000000E+002" in text
    assert "REFTMP 2 2.7500000000E+002" in text
    assert "REFTMP 3 2.7500000000E+002" in text
    assert "MOVCTL 3 1 0 0 1 0 1.5000000000E+000" in text


def test_prepare_parameterized_keys_reuses_existing_outputs(tmp_path):
    operations = _operations(tmp_path)
    samples = tmp_path / "samples.txt"
    samples.write_text("910\t810\n", encoding="utf-8")
    task = MultiOperationTask(
        "multi", str(samples), operations, str(tmp_path / "runs"), dry_run=True
    )
    generated = task.prepare_parameterized_keys()
    Path(generated[1]).write_text("existing-key", encoding="utf-8")

    repeated = task.prepare_parameterized_keys()

    assert repeated == generated
    assert Path(generated[1]).read_text(encoding="utf-8") == "existing-key"


def test_prepare_initial_db_reuses_existing_keys_and_db(tmp_path, monkeypatch):
    operations = _operations(tmp_path)
    samples = tmp_path / "samples.txt"
    samples.write_text("910\t810\n", encoding="utf-8")
    task = MultiOperationTask(
        "multi", str(samples), operations, str(tmp_path / "runs"), dry_run=True
    )
    task.prepare_parameterized_keys()
    db_path = tmp_path / "runs" / "0" / "result.DB"
    db_path.touch()
    monkeypatch.setattr(
        task, "prepare_parameterized_keys",
        lambda: (_ for _ in ()).throw(AssertionError("不应重新生成 KEY")),
    )

    assert task.prepare_initial_db_files() == [str(db_path)]


def test_dry_run_completes_and_rebuild_skips_completed_operations(tmp_path):
    operations = _operations(tmp_path)
    samples = tmp_path / "samples.txt"
    samples.write_text("910\t810\n", encoding="utf-8")
    work_dir = tmp_path / "runs"
    task = MultiOperationTask("multi", str(samples), operations, str(work_dir), dry_run=True)
    result = task.run()
    assert result["status"] == "completed"
    assert result["total"] == 1
    assert result["completed"] == 1
    assert result["running"] == 0
    assert result["failed"] == 0
    assert result["pending"] == 0
    assert all(item["status"] == "completed"
               for item in result["samples"]["0"]["operations"].values())
    transition = work_dir / "0" / "op2" / "transition.KEY"
    text = transition.read_text(encoding="utf-8")
    assert text.index("OBJPOS") < text.index("object1_control.KEY")
    assert "DIEGEO 1" not in text

    rebuilt = MultiOperationTask("multi", str(samples), operations, str(work_dir), dry_run=True)
    assert rebuilt.run()["status"] == "completed"


def test_resume_reprepares_solving_operation_when_result_db_is_missing(tmp_path):
    operations = _operations(tmp_path)
    samples = tmp_path / "samples.txt"
    samples.write_text("910\t810\n", encoding="utf-8")
    work_dir = tmp_path / "runs"
    task = MultiOperationTask("multi", str(samples), operations, str(work_dir), dry_run=True)
    sample_state = task.state["samples"]["0"]
    sample_state["status"] = "failed"
    sample_state["operations"]["1"].update({
        "status": "failed",
        "phase": "failed",
        "failed_phase": "solving",
    })
    task._save()

    assert not (work_dir / "0" / "result.DB").exists()
    result = task.run()

    assert result["status"] == "completed"
    assert (work_dir / "0" / "result.DB").exists()
    assert result["samples"]["0"]["operations"]["1"]["attempts"] == 1


def test_initial_db_failure_updates_sample_and_summary(tmp_path, monkeypatch):
    operations = _operations(tmp_path)
    samples = tmp_path / "samples.txt"
    samples.write_text("910\t810\n", encoding="utf-8")
    task = MultiOperationTask(
        "multi", str(samples), operations, str(tmp_path / "runs"), dry_run=False
    )
    monkeypatch.setattr(multi_operation_module, "key_to_db", lambda *_: None)

    with pytest.raises(FileNotFoundError, match="KEY 转 DB 后未生成"):
        task.prepare_initial_db_files()

    operation = task.state["samples"]["0"]["operations"]["1"]
    assert operation["status"] == "failed"
    assert operation["failed_phase"] == "preparing"
    assert task.state["failed"] == 1
    assert task.state["completed"] == 0


def test_sample_callback_runs_only_after_all_operations_complete(tmp_path):
    operations = _operations(tmp_path)
    samples = tmp_path / "samples.txt"
    samples.write_text("910\t810\n", encoding="utf-8")
    callback_states = []
    task = None

    def on_completed(sample_index):
        sample_state = task.state["samples"][str(sample_index)]
        callback_states.append({
            "sample_status": sample_state["status"],
            "operation_statuses": [
                item["status"] for item in sample_state["operations"].values()
            ],
        })

    task = MultiOperationTask(
        "multi", str(samples), operations, str(tmp_path / "runs"),
        dry_run=True, on_sample_completed=on_completed,
    )
    task.run()

    assert callback_states == [{
        "sample_status": "completed",
        "operation_statuses": ["completed", "completed"],
    }]


def test_explicit_state_file_can_live_outside_run_directory(tmp_path):
    operations = _operations(tmp_path)
    samples = tmp_path / "samples.txt"
    samples.write_text("910\t810\n", encoding="utf-8")
    state_file = tmp_path / "tasks" / "multi" / "multi_operation_state.json"
    task = MultiOperationTask(
        "multi", str(samples), operations, str(tmp_path / "runs"),
        dry_run=True, state_file=str(state_file),
    )
    assert Path(task.state_file) == state_file
    assert state_file.exists()
    assert not (tmp_path / "runs" / "multi_operation_state.json").exists()


def test_solver_branch_uses_current_operation_for_grain_validation(
        tmp_path, monkeypatch):
    """覆盖真实求解分支，防止当前工步配置再次出现未定义变量。"""
    operations = _operations(tmp_path)
    operations[1]["enable_grain"] = True
    second_template = Path(operations[1]["template_key"])
    second_template.write_text(
        "TRANS 1 1 0 0 0\n" + second_template.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    samples = tmp_path / "samples.txt"
    samples.write_text("910\t810\n", encoding="utf-8")

    def fake_key_to_db(_key_path, db_path):
        Path(db_path).touch()

    def fake_db_to_key(_db_path, key_path, _step):
        Path(key_path).write_text(
            _template(tmp_path / "terminal-template.KEY").read_text(encoding="utf-8")
            + "GRAIN 1 1 16 0\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(multi_operation_module, "key_to_db", fake_key_to_db)
    monkeypatch.setattr(multi_operation_module, "solve_db_sync", lambda _path: None)
    monkeypatch.setattr(multi_operation_module, "db_to_key", fake_db_to_key)
    monkeypatch.setattr(multi_operation_module, "run_key_actions", lambda _path: None)

    task = MultiOperationTask(
        "multi", str(samples), operations, str(tmp_path / "runs"), dry_run=False
    )
    assert task.run()["status"] == "completed"
