import json
from concurrent.futures import ThreadPoolExecutor

from mobo.automation.incremental import IncrementalDataset
from mobo.automation.dataset_format import format_dataset_row


def _line(values):
    return "\t".join(format_dataset_row(values))


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
        _line(["p0", "y0"]), _line(["p1", "updated"]), _line(["p2", "y2"]),
    ]
    assert dataset.is_completed(1)
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert len(state["samples"]) == 3


def test_incremental_dataset_failed_sample_can_resume(tmp_path):
    state_file = str(tmp_path / "incremental.json")
    output_file = str(tmp_path / "result.txt")
    dataset = IncrementalDataset(state_file, output_file)
    dataset.mark_started(4)
    dataset.mark_failed(
        4,
        "power loss",
        error_type="RuntimeError",
        traceback_text="trace line",
    )
    assert not dataset.is_completed(4)
    failed = json.loads(open(state_file, encoding="utf-8").read())["samples"]["4"]
    assert failed["attempts"] == 1
    assert failed["error_type"] == "RuntimeError"
    assert failed["traceback"] == "trace line"
    assert failed["failed_at"]

    resumed = IncrementalDataset(state_file, output_file)
    resumed.commit(4, ["x", "y"])
    assert resumed.is_completed(4)
    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == _line(["x", "y"]) + "\n"


def test_missing_output_is_rebuilt_from_state(tmp_path):
    state_file = str(tmp_path / "incremental.json")
    output = tmp_path / "result.txt"
    dataset = IncrementalDataset(state_file, str(output))
    dataset.commit(2, ["p2", "y2"])
    dataset.commit(0, ["p0", "y0"])
    output.unlink()

    IncrementalDataset(state_file, str(output))

    assert output.read_text(encoding="utf-8") == _line(["p0", "y0"]) + "\n" + _line(["p2", "y2"]) + "\n"


def test_incremental_dataset_saves_fixed_width_six_decimals(tmp_path):
    state_file = str(tmp_path / "incremental.json")
    output = tmp_path / "result.txt"
    dataset = IncrementalDataset(state_file, str(output))

    dataset.commit(0, [5, "1.239", "nan"])

    text = output.read_text(encoding="utf-8")
    assert text == _line([5, "1.239", "nan"]) + "\n"
    assert [cell.strip() for cell in text.rstrip("\n").split("\t")] == ["5.000000", "1.239000", "nan"]
    assert all(len(cell) == 20 for cell in text.rstrip("\n").split("\t"))
