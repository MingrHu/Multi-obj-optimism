"""单工步范围选择保持全局样本号及参数/DB/提取结果的一致对应。"""

import json
from pathlib import Path

import pytest

from mobo.automation import extract, pipeline, task_collection
from mobo.automation.pipeline import TaskStatus
from mobo.automation.solver import DeformSolver
from mobo.automation.task_collection import SingleOperationTaskDefinition, TargetDefinition


@pytest.fixture
def setup_task(tmp_path, monkeypatch):
    monkeypatch.setattr(task_collection, "task_dir", lambda task_id: tmp_path / "state" / task_id)
    template = tmp_path / "ring.KEY"
    template.write_text("NDTMP 1 0 1000\nMOVCTL 2 1 0 0 1 0 1\n")
    sample = tmp_path / "300.tsv"
    sample.write_text("".join(f"{1000 + i / 100:.2f}\t{i / 100:.2f}\n" for i in range(300)))
    definition = SingleOperationTaskDefinition(
        task_id="range-test", name="range", workspace=tmp_path / "run",
        template_key=str(template),
        parameters=({"name": "temp", "object": "workpiece", "range": [1000, 1100]},
                    {"name": "speed", "object": "topdie", "range": [0, 3]}),
        targets=(TargetDefinition("stress", "stress", "workpiece", (1,)),),
    )
    return definition, sample


@pytest.mark.parametrize("start,end", [(0, 100), (100, 200), (200, 300)])
def test_only_selected_hundred_keys_and_db_paths_are_used(setup_task, start, end):
    definition, sample = setup_task
    original = sample.read_bytes()
    task = definition.build(sample, sample_start=start, sample_end=end)
    task.generate_keys()
    assert task.status == TaskStatus.DONE
    assert task.sample_indices == list(range(start, end))
    assert len(task.key_files) == 100
    assert Path(task.key_files[0]).name == f"ring{start}.KEY"
    assert Path(task.key_files[-1]).name == f"ring{end - 1}.KEY"
    assert float(Path(task.key_files[0]).read_text().splitlines()[0].split()[-1]) == 1000 + start / 100
    pending = task.prepare_db_files()
    assert len(pending) == 100
    assert {Path(db).parent.name for _, db in pending} == {str(i) for i in range(start, end)}
    assert sample.read_bytes() == original


@pytest.mark.parametrize("start,end", [(-1, 100), (100, 100), (200, 100), (0, 301), (300, None)])
def test_invalid_range_fails_without_creating_key_files(setup_task, start, end):
    definition, sample = setup_task
    with pytest.raises(ValueError, match="样本范围"):
        definition.prepare_keys(sample, sample_start=start, sample_end=end)
    assert not (definition.workspace / "input_keys").exists()


def test_rebuild_and_different_ranges_keep_independent_state(setup_task, tmp_path):
    definition, sample = setup_task
    task = definition.build(sample, sample_start=100, sample_end=200)
    task.load_samples_into_table()
    task.prepare_db_files()
    progress = DeformSolver(process_info_file=task.process_info_file)
    progress._init_progress(task.db_files)
    progress._mark_done(task.db_files[0])
    resumed = definition.build(sample, sample_start=100, sample_end=200)
    resumed.load_samples_into_table()
    resumed.prepare_db_files()
    assert resumed.process_info_file == task.process_info_file
    assert len(progress.pending_db_files(resumed.db_files)) == 99
    assert resumed.db_files[0] == task.db_files[0]
    other = definition.build(sample, sample_start=200, sample_end=300)
    other_workspace = definition.build(sample, workspace=tmp_path / "another",
                                       sample_start=100, sample_end=200)
    assert other.process_info_file != task.process_info_file
    assert other.incremental_state_file != task.incremental_state_file
    assert other.result_txt_dir != task.result_txt_dir
    assert other_workspace.incremental_state_file != task.incremental_state_file


def test_incremental_extraction_uses_global_id_and_correct_parameter_row(setup_task, monkeypatch):
    definition, sample = setup_task
    task = definition.build(sample, sample_start=100, sample_end=102, incremental=True)
    task.generate_keys()
    captured = []
    monkeypatch.setattr(pipeline, "key_to_db_batch", lambda keys, dbs: captured.extend(dbs))
    monkeypatch.setattr(pipeline, "extract_dataset_row",
                        lambda db, index, directory, params, *args: list(params) + [str(index)])

    class ImmediateSolver:
        def __init__(self, *, on_completed, **kwargs):
            self.completed = on_completed

        def run_all(self, dbs):
            for db in reversed(dbs):
                self.completed(db)

    monkeypatch.setattr(pipeline, "DeformSolver", ImmediateSolver)
    task.run_solver()
    assert task.status == TaskStatus.DONE
    assert [Path(path).parent.name for path in captured] == ["100", "101"]
    state = json.loads(Path(task.incremental_state_file).read_text())
    assert set(state["samples"]) == {"100", "101"}
    rows = [row.split() for row in Path(task.incremental_output_file).read_text().splitlines()]
    assert rows == [["1001.000000", "1.000000", "100.000000"],
                    ["1001.010000", "1.010000", "101.000000"]]


def test_extract_can_rebuild_selected_db_list_without_generate(setup_task, monkeypatch):
    definition, sample = setup_task
    task = definition.build(sample, sample_start=100, sample_end=102)
    task.load_samples_into_table()
    task.prepare_db_files()
    directories = []

    def fake_values(db, directory, *args):
        directories.append(Path(directory).name)
        return [Path(db).parent.name]

    monkeypatch.setattr(extract, "_extract_values", fake_values)
    task.extract()
    assert task.status == TaskStatus.DONE
    assert directories == ["100", "101"]
    output = next(Path(task.result_txt_dir).glob("*.txt"))
    assert [row.split() for row in output.read_text().splitlines()] == [
        ["1001.000000", "1.000000", "100.000000"],
        ["1001.010000", "1.010000", "101.000000"],
    ]
