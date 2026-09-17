from pathlib import Path

from mobo.automation import multi_operation_service as service


def test_range_suffix_keeps_full_name_and_isolates_shards():
    assert service._task_state_name() == "state.json"
    assert service._task_state_name(0, 100) == "state_0_100.json"
    assert service._task_state_name(100, 200) == "state_100_200.json"
    assert service._task_state_name(200, 260) == "state_200_260.json"
    assert service._workflow_state_file("multi").endswith(
        "multi_operation_state.json"
    )
    assert service._workflow_state_file("multi", 0, 100).endswith(
        "multi_operation_state_0_100.json"
    )
    assert service._workflow_state_file("multi", 100, 200).endswith(
        "multi_operation_state_100_200.json"
    )
    assert service._workflow_state_file("multi", 200, 260).endswith(
        "multi_operation_state_200_260.json"
    )
    assert service._incremental_state_file("multi", 0, 100).endswith(
        "incremental_dataset_0_100.json"
    )
    assert service._incremental_state_file("multi", 100, 200).endswith(
        "incremental_dataset_100_200.json"
    )
    assert service._incremental_state_file("multi", 200, 260).endswith(
        "incremental_dataset_200_260.json"
    )


def test_init_does_not_pass_incremental_option_to_task(monkeypatch, tmp_path):
    captured = {}

    class FakeTask:
        state_file = "workflow.json"

        def __init__(self, task_id, sample_file, operations, work_dir,
                     max_parallel_samples, keep_checkpoints, dry_run, state_file,
                     sample_start, sample_end):
            captured.update(locals())

        def prepare_parameterized_keys(self):
            return ["op1.KEY", "op2.KEY"]

    monkeypatch.setattr(service, "MultiOperationTask", FakeTask)
    monkeypatch.setattr(service.task_store, "init_state", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        service.task_store, "update", lambda task_id, **kwargs: {"task_id": task_id, **kwargs}
    )

    result = service.init_multi_operation_task(
        "multi", str(tmp_path / "samples.txt"), [{"name": "op"}],
        str(tmp_path / "runs"), incremental=True,
        sample_start=200, sample_end=300,
    )

    assert result["stage"] == "initialized"
    assert result["data"]["key_file_count"] == 2
    assert "incremental" not in captured
    assert captured["sample_start"] == 200
    assert captured["sample_end"] == 300
    assert captured["state_file"].endswith("multi_operation_state_200_300.json")


def test_shards_keep_independent_task_states(monkeypatch, tmp_path):
    instances = []

    class FakeTask:
        def __init__(self, **kwargs):
            self.state_file = kwargs["state_file"]
            self.sample_start = kwargs["sample_start"]
            self.sample_end = kwargs["sample_end"]
            instances.append(self)

        def prepare_parameterized_keys(self):
            return ["op1.KEY"]

    def fake_task_dir(task_id):
        return Path(tmp_path) / task_id

    monkeypatch.setattr(service, "MultiOperationTask", FakeTask)
    monkeypatch.setattr(service, "task_dir", fake_task_dir)
    monkeypatch.setattr(service.task_store, "task_dir", fake_task_dir)

    common = {
        "task_id": "multi",
        "sample_file": str(tmp_path / "samples.txt"),
        "operations": [{"name": "op"}],
        "work_dir": str(tmp_path / "runs"),
    }
    service.init_multi_operation_task(
        **common, sample_start=0, sample_end=100
    )
    service.init_multi_operation_task(
        **common, sample_start=200, sample_end=260
    )

    first = service.task_store.load("multi", "state_0_100.json")
    second = service.task_store.load("multi", "state_200_260.json")
    assert first["req"]["sample_start"] == 0
    assert first["req"]["sample_end"] == 100
    assert second["req"]["sample_start"] == 200
    assert second["req"]["sample_end"] == 260
    assert not (fake_task_dir("multi") / "state.json").exists()

    service._rebuild("multi", 0, 100)
    assert instances[-1].sample_start == 0
    assert instances[-1].sample_end == 100
    service._rebuild("multi", 200, 260)
    assert instances[-1].sample_start == 200
    assert instances[-1].sample_end == 260
