from mobo.automation import multi_operation_service as service


def test_init_does_not_pass_service_options_to_task(monkeypatch, tmp_path):
    captured = {}

    class FakeTask:
        state_file = "workflow.json"

        def __init__(self, task_id, sample_file, operations, work_dir,
                     max_parallel_samples, keep_checkpoints, dry_run, state_file):
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
        incremental_result_dir=str(tmp_path / "results"),
    )

    assert result["stage"] == "initialized"
    assert result["data"]["key_file_count"] == 2
    assert "incremental" not in captured
    assert "incremental_result_dir" not in captured
