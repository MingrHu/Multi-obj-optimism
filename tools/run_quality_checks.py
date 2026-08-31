"""统一执行仓库质量检查并生成汇总报告"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = ROOT / "quality-reports"


@dataclass
class CheckResult:
    name: str
    command: list[str]
    exit_code: int
    duration_seconds: float
    log_file: str

    @property
    def passed(self) -> bool:
        return self.exit_code == 0


def _tool_command(name: str) -> str:
    suffix = ".exe" if os.name == "nt" else ""
    candidates = [
        Path(sysconfig.get_path("scripts")) / f"{name}{suffix}",
        ROOT / ".venv" / "Scripts" / f"{name}{suffix}",
        ROOT / ".venv" / "bin" / name,
    ]
    for installed_tool in candidates:
        if installed_tool.is_file():
            return str(installed_tool)
    discovered = shutil.which(name)
    if discovered:
        return discovered
    raise FileNotFoundError(f"未找到 {name} 请先安装 requirements/dev.txt")


def _commands(report_dir: Path, min_score: float, with_security: bool) -> list[tuple[str, list[str]]]:
    coverage_file = report_dir / "coverage.json"
    commands = [
        (
            "tests",
            [
                sys.executable, "-m", "pytest", "-m", "not slow",
                f"--cov-report=json:{coverage_file}",
            ],
        ),
        ("ruff", [_tool_command("ruff"), "check", "src", "tests", "scripts", "tools"]),
        (
            "complexity",
            [_tool_command("xenon"), "--max-absolute", "D", "--max-modules", "C",
             "--max-average", "B", "src/mobo"],
        ),
        ("documentation", [sys.executable, "tools/check_docs.py"]),
        (
            "score",
            [sys.executable, "tools/quality_score.py", "--report-dir", str(report_dir),
             "--min-score", str(min_score)],
        ),
    ]
    if with_security:
        commands.append((
            "dependency-security",
            [
                sys.executable, "-m", "pip_audit", "-r", "requirements/runtime.txt",
                "-r", "requirements/server.txt", "--no-deps", "--disable-pip",
                "--cache-dir", str(report_dir / "pip-audit-cache"),
            ],
        ))
    return commands


def _run_check(name: str, command: list[str], report_dir: Path) -> CheckResult:
    print(f"[RUN] {name}", flush=True)
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    duration = round(time.monotonic() - started, 2)
    log_path = report_dir / f"{name}.log"
    output = completed.stdout
    if completed.stderr:
        output += ("\n" if output else "") + completed.stderr
    log_path.write_text(output, encoding="utf-8")
    state = "PASS" if completed.returncode == 0 else "FAIL"
    print(f"[{state}] {name} {duration:.2f}s -> {log_path}", flush=True)
    return CheckResult(name, command, completed.returncode, duration, log_path.name)


def _markdown(results: list[CheckResult], score: dict | None) -> str:
    passed = sum(result.passed for result in results)
    rows = [
        "# 仓库质量检查汇总", "",
        f"**检查结果：{passed}/{len(results)} 项通过**", "",
    ]
    if score:
        rows.extend([f"**工程质量评分：{score['score']}/100（{score['grade']}）**", ""])
    rows.extend(["| 检查项 | 状态 | 耗时 | 日志 |", "|---|---|---:|---|"])
    for result in results:
        state = "通过" if result.passed else f"失败 ({result.exit_code})"
        rows.append(
            f"| {result.name} | {state} | {result.duration_seconds:.2f}s | {result.log_file} |"
        )
    rows.extend(["", "生成目录中的日志用于定位具体失败原因", ""])
    return "\n".join(rows)


def _load_score(report_dir: Path) -> dict | None:
    score_path = report_dir / "quality-score.json"
    if not score_path.is_file():
        return None
    return json.loads(score_path.read_text(encoding="utf-8"))


def _write_summary(report_dir: Path, results: list[CheckResult]) -> None:
    score = _load_score(report_dir)
    payload = {
        "passed": sum(result.passed for result in results),
        "total": len(results),
        "score": score,
        "checks": [asdict(result) | {"passed": result.passed} for result in results],
    }
    (report_dir / "quality-check-summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    markdown = _markdown(results, score)
    (report_dir / "quality-check-summary.md").write_text(markdown, encoding="utf-8")
    print(f"\n{markdown}")


def main() -> int:
    parser = argparse.ArgumentParser(description="执行仓库质量检查并生成汇总报告")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--min-score", type=float, default=60)
    parser.add_argument("--with-security", action="store_true", help="额外执行联网依赖漏洞审计")
    args = parser.parse_args()
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONUTF8", "1")

    results = [
        _run_check(name, command, report_dir)
        for name, command in _commands(report_dir, args.min_score, args.with_security)
    ]
    _write_summary(report_dir, results)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
