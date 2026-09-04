from types import SimpleNamespace

from mobo.cli import sample_recovery_demo


def test_demo_can_recover_with_task_id_only(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        sample_recovery_demo,
        "get_single_operation_task_definition",
        lambda task_id: SimpleNamespace(workspace=tmp_path / task_id),
    )

    def fake_recover(task_id, key_dir, output_file, *, overwrite):
        captured.update({
            "task_id": task_id,
            "key_dir": key_dir,
            "output_file": output_file,
            "overwrite": overwrite,
        })
        return output_file

    monkeypatch.setattr(sample_recovery_demo, "recover_samples", fake_recover)

    result = sample_recovery_demo.recover_sample_test("gh4169-ring-single-task-1")

    workspace = tmp_path / "gh4169-ring-single-task-1"
    assert captured == {
        "task_id": "gh4169-ring-single-task-1",
        "key_dir": workspace / "input_keys",
        "output_file": workspace / "samples" / "gh4169-ring-single-task-1-recovered.txt",
        "overwrite": False,
    }
    assert result == str(captured["output_file"])
