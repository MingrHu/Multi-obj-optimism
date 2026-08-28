from mobo.automation import multi_operation_service as service


def test_range_suffix_keeps_full_name_and_isolates_shards():
    assert service._workflow_state_file("multi").endswith(
        "multi_operation_state.json"
    )
    assert service._workflow_state_file("multi", 0, 200).endswith(
        "multi_operation_state_0_200.json"
    )
    assert service._workflow_state_file("multi", 200, 300).endswith(
        "multi_operation_state_200_300.json"
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
