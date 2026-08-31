"""生成仓库本地工程质量评分和明细报告"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "src" / "mobo"
DEFAULT_REPORT_DIR = ROOT / "quality-reports"


def _python_files() -> list[Path]:
    return sorted(SOURCE_DIR.rglob("*.py"))


def _complexity_metrics() -> dict[str, Any]:
    from radon.complexity import cc_rank, cc_visit
    from radon.metrics import mi_visit

    complexities: list[int] = []
    maintainability: list[float] = []
    ranks: Counter[str] = Counter()
    for path in _python_files():
        source = path.read_text(encoding="utf-8")
        maintainability.append(float(mi_visit(source, multi=True)))
        for block in cc_visit(source):
            value = int(block.complexity)
            complexities.append(value)
            ranks[cc_rank(value)] += 1
    return {
        "file_count": len(maintainability),
        "block_count": len(complexities),
        "maintainability_average": round(mean(maintainability), 2),
        "maintainability_minimum": round(min(maintainability), 2),
        "complexity_average": round(mean(complexities), 2) if complexities else 0,
        "complexity_maximum": max(complexities, default=0),
        "complexity_ranks": dict(sorted(ranks.items())),
    }


def _coverage_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "percent": 0.0}
    payload = json.loads(path.read_text(encoding="utf-8"))
    percent = float(payload.get("totals", {}).get("percent_covered", 0))
    return {"available": True, "percent": round(percent, 2)}


def _ruff_metrics() -> dict[str, Any]:
    command = [
        sys.executable, "-m", "ruff", "check", "src", "tests", "scripts", "tools",
        "--output-format", "json",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.returncode not in {0, 1}:
        raise RuntimeError(completed.stderr.strip() or "ruff 执行失败")
    findings = json.loads(completed.stdout or "[]")
    return {"violation_count": len(findings)}


def _dead_code_metrics() -> dict[str, Any]:
    from vulture import Vulture

    analyzer = Vulture()
    analyzer.scavenge([str(SOURCE_DIR)])
    findings = analyzer.get_unused_code(min_confidence=100, sort_by_size=True)
    return {
        "certain_count": len(findings),
        "items": [f"{item.filename}:{item.first_lineno}:{item.name}" for item in findings],
    }


def _documentation_metrics() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "tools/check_docs.py"], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    return {"passed": completed.returncode == 0}


def _components(metrics: dict[str, Any]) -> dict[str, float]:
    complexity = metrics["complexity"]
    ranks = complexity["complexity_ranks"]
    block_count = max(1, complexity["block_count"])
    rank_weights = {"A": 1.0, "B": 0.7, "C": 0.3, "D": 0.1, "E": 0.0, "F": 0.0}
    rank_score = 12.0 * sum(ranks.get(rank, 0) * weight for rank, weight in rank_weights.items())
    rank_score /= block_count
    worst_score = 8.0 * max(0.0, min(1.0, (41 - complexity["complexity_maximum"]) / 31))
    return {
        "maintainability": round(min(25.0, complexity["maintainability_average"] * 0.25), 2),
        "complexity": round(rank_score + worst_score, 2),
        "coverage": round(min(30.0, metrics["coverage"]["percent"] * 0.3), 2),
        "lint": round(max(0.0, 15.0 - metrics["ruff"]["violation_count"] * 0.5), 2),
        "dead_code": round(max(0.0, 5.0 - metrics["dead_code"]["certain_count"] * 0.5), 2),
        "documentation": 5.0 if metrics["documentation"]["passed"] else 0.0,
    }


def _grade(score: float) -> str:
    for threshold, grade in ((90, "A"), (80, "B"), (70, "C"), (60, "D")):
        if score >= threshold:
            return grade
    return "E"


def _markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    rows = [
        "# 仓库工程质量评分",
        "",
        f"**总分：{report['score']}/100（{report['grade']}）**",
        "",
        "| 维度 | 得分 | 满分 |",
        "|---|---:|---:|",
    ]
    limits = {
        "maintainability": 25, "complexity": 20, "coverage": 30,
        "lint": 15, "dead_code": 5, "documentation": 5,
    }
    rows.extend(
        f"| {name} | {score:.2f} | {limits[name]} |"
        for name, score in report["components"].items()
    )
    rows.extend([
        "", "## 关键指标", "",
        f"- 覆盖率：{metrics['coverage']['percent']:.2f}%",
        f"- 平均可维护性指数：{metrics['complexity']['maintainability_average']:.2f}",
        f"- 最大圈复杂度：{metrics['complexity']['complexity_maximum']}",
        f"- Ruff 问题：{metrics['ruff']['violation_count']}",
        f"- 100%置信度死代码：{metrics['dead_code']['certain_count']}",
        "",
        "> 本评分用于仓库内部趋势比较，不替代 SonarQube 或安全审计结论",
    ])
    return "\n".join(rows) + "\n"


def _run_tests(report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "pytest", "-m", "not slow",
        f"--cov-report=json:{report_dir / 'coverage.json'}",
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def build_report(report_dir: Path) -> dict[str, Any]:
    metrics = {
        "complexity": _complexity_metrics(),
        "coverage": _coverage_metrics(report_dir / "coverage.json"),
        "ruff": _ruff_metrics(),
        "dead_code": _dead_code_metrics(),
        "documentation": _documentation_metrics(),
    }
    components = _components(metrics)
    score = round(sum(components.values()), 2)
    return {"score": score, "grade": _grade(score), "components": components, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description="生成仓库本地工程质量评分")
    parser.add_argument("--run-tests", action="store_true", help="评分前运行非慢速测试并生成覆盖率")
    parser.add_argument("--min-score", type=float, default=0, help="低于该分数时返回非零状态")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    report_dir = args.report_dir.resolve()
    if args.run_tests:
        _run_tests(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(report_dir)
    (report_dir / "quality-score.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    markdown = _markdown(report)
    (report_dir / "quality-score.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 1 if report["score"] < args.min_score else 0


if __name__ == "__main__":
    raise SystemExit(main())
