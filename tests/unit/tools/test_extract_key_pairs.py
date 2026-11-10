"""独立初态目录配对与重复运行补齐回归测试"""

import runpy
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "tools" / "extract_key_pairs.py"
extract_key_pairs = runpy.run_path(str(SCRIPT))["extract_key_pairs"]


@pytest.fixture
def sources(tmp_path):
    initial = tmp_path / "input_keys"
    terminal = tmp_path / "result_keys"
    output = tmp_path / "output"
    initial.mkdir()
    case = terminal / "100"
    case.mkdir(parents=True)
    (initial / "TC18100.KEY").write_bytes(b"correct initial")
    (case / "TC18100_terminal.KEY").write_bytes(b"terminal")
    (case / "TC18100_step_1.KEY").write_bytes(b"not initial")
    return initial, terminal, output


def test_pairs_use_input_key_and_preserve_existing_files(sources):
    initial, terminal, output = sources
    assert extract_key_pairs(*sources) == (1, 0)
    assert {p.name for p in (output / "100").iterdir()} == {
        "TC18100.KEY", "TC18100_terminal.KEY",
    }
    assert (output / "100/TC18100.KEY").read_bytes() == b"correct initial"
    (initial / "TC18100.KEY").write_bytes(b"changed source")
    assert extract_key_pairs(*sources) == (0, 1)
    assert (output / "100/TC18100.KEY").read_bytes() == b"correct initial"
    assert not list(output.glob(".key_pair_*"))


@pytest.mark.parametrize("missing", ["TC18100.KEY", "TC18100_terminal.KEY"])
def test_rerun_fills_either_missing_member(sources, missing):
    initial, terminal, output = sources
    extract_key_pairs(*sources)
    (output / "100" / missing).unlink()
    assert extract_key_pairs(*sources) == (1, 0)
    assert (output / "100/TC18100.KEY").read_bytes() == b"correct initial"
    assert (output / "100/TC18100_terminal.KEY").read_bytes() == b"terminal"


def test_old_step_file_does_not_replace_input_initial(sources):
    initial, terminal, output = sources
    destination = output / "100"
    destination.mkdir(parents=True)
    (destination / "TC18100_step_1.KEY").write_bytes(b"old step")
    (destination / "TC18100_terminal.KEY").write_bytes(b"existing terminal")
    assert extract_key_pairs(*sources) == (1, 0)
    assert (destination / "TC18100.KEY").read_bytes() == b"correct initial"
    assert (destination / "TC18100_terminal.KEY").read_bytes() == b"existing terminal"
    assert not (destination / "TC18100_step_1.KEY").exists()
    assert (terminal / "100/TC18100_step_1.KEY").read_bytes() == b"not initial"


def test_old_step_is_removed_even_when_pair_is_complete(sources):
    initial, terminal, output = sources
    extract_key_pairs(*sources)
    old_step = output / "100/TC18100_step_1.KEY"
    old_step.write_bytes(b"old step")
    assert extract_key_pairs(*sources) == (0, 1)
    assert not old_step.exists()
    assert (output / "100/TC18100.KEY").read_bytes() == b"correct initial"
    assert (output / "100/TC18100_terminal.KEY").read_bytes() == b"terminal"


def test_missing_source_can_be_repaired_on_next_run(sources):
    initial, terminal, output = sources
    (initial / "TC18100.KEY").unlink()
    assert extract_key_pairs(*sources) == (0, 1)
    assert not (output / "100").exists()
    (initial / "TC18100.KEY").write_bytes(b"restored")
    assert extract_key_pairs(*sources) == (1, 0)


def test_ambiguous_terminal_is_skipped(sources):
    initial, terminal, output = sources
    (terminal / "100/other100_terminal.KEY").write_bytes(b"other")
    assert extract_key_pairs(*sources) == (0, 1)
    assert not (output / "100").exists()


@pytest.mark.parametrize("source_index", [0, 1])
def test_output_cannot_overlap_sources(sources, source_index):
    with pytest.raises(ValueError, match="不能相同或互相包含"):
        extract_key_pairs(sources[0], sources[1], sources[source_index] / "output")
