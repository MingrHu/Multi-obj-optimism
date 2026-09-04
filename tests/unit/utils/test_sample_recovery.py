import pytest

from mobo.utils.sample_recovery import (
    extract_sample_from_key,
    recover_samples,
)


OBJECT_IDS = {
    "workpiece": "1",
    "driving_roll": "2",
    "pressure_roll": "3",
    "axial_roll_1": "4",
    "axial_roll_2": "5",
}


def _profile_key(workpiece: str, die: str, speed: str) -> str:
    return "".join([
        f"REFTMP 1 {workpiece}\n",
        f"REFTMP 2 {die}\n",
        f"REFTMP 3 {die}\n",
        "MOVCTL 3 1 2 0 1 0 5\n",
        "0 0\n",
        f"10 -{speed}\n",
        f"20 -{speed}\n",
        "30 0\n",
        "40 0\n",
        f"REFTMP 4 {die}\n",
        f"REFTMP 5 {die}\n",
    ])


def _constant_key(workpiece: str, die: str, speed: str) -> str:
    return "".join([
        f"REFTMP 1 {workpiece}\n",
        f"REFTMP 2 {die}\n",
        f"REFTMP 3 {die}\n",
        f"MOVCTL 3 1 0 0 1 0 {speed}\n",
        f"REFTMP 4 {die}\n",
        f"REFTMP 5 {die}\n",
    ])


def test_extract_sample_uses_task_parameter_column_order(tmp_path):
    key_path = tmp_path / "GH41690.KEY"
    key_path.write_text(_profile_key("1.125E+3", "3E+2", "1.25"))
    parameters = (
        {"name": "pressure_roll_profile_peak_speed", "object": "pressure_roll", "range": [0.1, 2.5]},
        {"name": "workpiece_temperature", "object": "workpiece", "range": [1100, 1150]},
        {"name": "ring_die_temperature", "object": "ring_dies", "range": [250, 350]},
    )

    values = extract_sample_from_key(key_path, parameters, OBJECT_IDS)

    assert [str(value) for value in values] == ["1.25", "1125", "3E+2"]


@pytest.mark.parametrize(
    "task_id,prefix,key_text,expected",
    [
        (
            "gh4169-ring-single-task-1",
            "GH4169",
            _profile_key("1125", "300", "1.25"),
            "1125.00\t300.00\t1.25\n",
        ),
        (
            "7050-ring-single-task-1",
            "7050",
            _constant_key("400", "200", "0.75"),
            "400.00\t200.00\t0.75\n",
        ),
    ],
)
def test_recover_registered_single_operation_task(
    tmp_path, task_id, prefix, key_text, expected
):
    key_dir = tmp_path / "keys"
    key_dir.mkdir()
    (key_dir / f"{prefix}0.KEY").write_text(key_text)
    output = tmp_path / "recovered.txt"

    recover_samples(task_id, key_dir, output)

    assert output.read_text() == expected


def test_recover_orders_by_numeric_key_suffix(tmp_path):
    key_dir = tmp_path / "keys"
    key_dir.mkdir()
    (key_dir / "GH41691.KEY").write_text(_profile_key("1110", "260", "0.2"))
    (key_dir / "GH41690.KEY").write_text(_profile_key("1100", "250", "0.1"))
    output = tmp_path / "recovered.txt"

    recover_samples("gh4169-ring-single-task-1", key_dir, output)

    assert output.read_text() == (
        "1100.00\t250.00\t0.10\n"
        "1110.00\t260.00\t0.20\n"
    )


def test_recover_rejects_missing_index(tmp_path):
    (tmp_path / "GH41691.KEY").write_text(_profile_key("1110", "260", "0.2"))

    with pytest.raises(ValueError, match="从 0 连续排列"):
        recover_samples(
            "gh4169-ring-single-task-1", tmp_path, tmp_path / "out.txt"
        )


def test_extract_rejects_inconsistent_shared_parameter(tmp_path):
    key_path = tmp_path / "GH41690.KEY"
    key_path.write_text(
        _profile_key("1125", "300", "1.25").replace("REFTMP 5 300", "REFTMP 5 301")
    )
    parameters = (
        {"name": "ring_die_temperature", "object": "ring_dies", "range": [250, 350]},
    )

    with pytest.raises(ValueError, match="多处值不一致"):
        extract_sample_from_key(key_path, parameters, OBJECT_IDS)


def test_recover_does_not_overwrite_by_default(tmp_path):
    (tmp_path / "GH41690.KEY").write_text(_profile_key("1125", "300", "1.25"))
    output = tmp_path / "out.txt"
    output.write_text("keep")

    with pytest.raises(FileExistsError, match="overwrite=True"):
        recover_samples(
            "gh4169-ring-single-task-1", tmp_path, output
        )

    assert output.read_text() == "keep"
