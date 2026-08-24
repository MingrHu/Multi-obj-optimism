import json
from concurrent.futures import ThreadPoolExecutor

from mobo.automation.incremental import IncrementalDataset


def test_incremental_dataset_is_ordered_and_idempotent(tmp_path):
    state_file = tmp_path / "incremental.json"
    output_file = tmp_path / "result.txt"
    dataset = IncrementalDataset(str(state_file), str(output_file))

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(lambda item: dataset.commit(*item), [
            (2, ["p2", "y2"]),
            (0, ["p0", "y0"]),
            (1, ["p1", "y1"]),
        ]))
    dataset.commit(1, ["p1", "updated"])

    assert output_file.read_text(encoding="utf-8").splitlines() == [
        "p0\ty0", "p1\tupdated", "p2\ty2",
    ]
    assert dataset.is_completed(1)
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert len(state["samples"]) == 3


def test_incremental_dataset_failed_sample_can_resume(tmp_path):
    state_file = str(tmp_path / "incremental.json")
    output_file = str(tmp_path / "result.txt")
    dataset = IncrementalDataset(state_file, output_file)
    dataset.mark_started(4)
    dataset.mark_failed(4, "power loss")
    assert not dataset.is_completed(4)

    resumed = IncrementalDataset(state_file, output_file)
    resumed.commit(4, ["x", "y"])
    assert resumed.is_completed(4)
    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "x\ty\n"
