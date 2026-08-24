"""使用短步 KEY 实测当前机器与许可证的 DEFORM 稳定并行数。"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from mobo.automation.solver import key_to_db, solve_db_sync
from mobo.common.paths import KEY_FILE_DIR, LOGS_DIR


def _short_key(source: Path, target: Path) -> None:
    text = source.read_text(encoding="utf-8", errors="ignore")
    text, nstep_count = re.subn(
        r"^NSTEP\s+\S+.*$", "NSTEP        1", text, count=1, flags=re.MULTILINE
    )
    text, stpinc_count = re.subn(
        r"^STPINC\s+\S+.*$", "STPINC       1", text, count=1, flags=re.MULTILINE
    )
    if nstep_count != 1 or stpinc_count != 1:
        raise ValueError(f"无法在模板中唯一定位 NSTEP/STPINC: {source}")
    target.write_text(text, encoding="utf-8")


def _run_level(level: int, baseline_db: Path, root: Path) -> dict:
    level_dir = root / f"parallel_{level}"
    level_dir.mkdir()
    db_files = []
    for index in range(level):
        job_dir = level_dir / str(index)
        job_dir.mkdir()
        db_path = job_dir / "short.DB"
        shutil.copy2(baseline_db, db_path)
        db_files.append(db_path)

    started = time.perf_counter()
    errors = []
    with ThreadPoolExecutor(max_workers=level) as executor:
        futures = {executor.submit(solve_db_sync, str(path)): path for path in db_files}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                path = futures[future]
                log_path = path.with_suffix(".LOG")
                log_tail = ""
                if log_path.exists():
                    log_tail = log_path.read_text(
                        encoding="utf-8", errors="ignore"
                    )[-4000:]
                errors.append({
                    "db": str(path),
                    "error": str(exc),
                    "log_tail": log_tail,
                })
    elapsed = time.perf_counter() - started
    completed = level - len(errors)
    return {
        "parallel": level,
        "completed": completed,
        "failed": len(errors),
        "elapsed_seconds": round(elapsed, 3),
        "jobs_per_minute": round(completed * 60.0 / elapsed, 3) if elapsed else 0.0,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--key",
        type=Path,
        default=KEY_FILE_DIR / "tc4_ring_multi_task_1" / "1.KEY",
    )
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16]
    )
    args = parser.parse_args()
    levels = sorted(set(args.levels))
    if not levels or levels[0] < 1:
        parser.error("并行级别必须是正整数")

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_key": str(args.key.resolve()),
        "logical_cpus": os.cpu_count(),
        "levels": [],
        "max_stable_parallel": 0,
    }
    with tempfile.TemporaryDirectory(prefix="mobo_deform_parallel_") as temp:
        root = Path(temp)
        short_key = root / "short.KEY"
        baseline_db = root / "baseline.DB"
        _short_key(args.key, short_key)
        key_to_db(str(short_key), str(baseline_db))
        if not baseline_db.exists():
            raise RuntimeError("DEFORM 前处理器未生成基准 DB")

        for level in levels:
            result = _run_level(level, baseline_db, root)
            report["levels"].append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if result["failed"]:
                break
            report["max_stable_parallel"] = level

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    output = LOGS_DIR / (
        "deform_parallel_benchmark_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".json"
    )
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report={output}")
    return 0 if report["max_stable_parallel"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
