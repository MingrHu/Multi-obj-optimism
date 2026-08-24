"""多工步采样、KEY 拆分和磁盘续跑测试。"""

import hashlib
import json
from pathlib import Path

from mobo.automation.multi_operation import (
    MultiOperationTask,
    generate_multi_operation_samples,
    split_operation_key,
)


def _template(path: Path, center=(0, 0, 0)) -> Path:
    path.write_text(
        "NDTMP 1 900\n"
        "* Data for Object # 1\n"
        f"CNTRAX 1 {center[0]} {center[1]} {center[2]} 0 0 1 0\n"
        "MOVCTL 1 0\n"
        "DIEGEO 1 1\n"
        "1 2 3\n"
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


def test_generate_multi_operation_samples_writes_column_manifest(tmp_path):
    operations = _operations(tmp_path)
    path = generate_multi_operation_samples(
        "multi", operations, str(tmp_path), method="full", level_nums=[2, 2]
    )
    rows = Path(path).read_text(encoding="utf-8").splitlines()
    manifest = json.loads(Path(path + ".json").read_text(encoding="utf-8"))
    assert len(rows) == 4
    assert [item["operation"] for item in manifest["columns"]] == [1, 2]


def test_split_operation_key_keeps_source_and_excludes_workpiece_geometry(tmp_path):
    source = _template(tmp_path / "source.KEY")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    parts = split_operation_key(str(source), str(tmp_path / "parts"))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert "DIEGEO 1" not in Path(parts["object1_control"]).read_text(encoding="utf-8")
    assert "DIEGEO 2" in Path(parts["object2"]).read_text(encoding="utf-8")


def test_dry_run_completes_and_rebuild_skips_completed_operations(tmp_path):
    operations = _operations(tmp_path)
    samples = tmp_path / "samples.txt"
    samples.write_text("910\t810\n", encoding="utf-8")
    work_dir = tmp_path / "runs"
    task = MultiOperationTask("multi", str(samples), operations, str(work_dir), dry_run=True)
    result = task.run()
    assert result["status"] == "completed"
    assert all(item["status"] == "completed"
               for item in result["samples"]["0"]["operations"].values())
    transition = work_dir / "sample_000000" / "operation_2" / "transition.KEY"
    text = transition.read_text(encoding="utf-8")
    assert text.index("OBJPOS") < text.index("object1_control.KEY")
    assert "DIEGEO 1" not in text

    rebuilt = MultiOperationTask("multi", str(samples), operations, str(work_dir), dry_run=True)
    assert rebuilt.run()["status"] == "completed"

