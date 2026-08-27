"""检查根文档与仓库公共表面是否一致，仅依赖 Python 标准库。"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "tools" / "docs_surface.json"
REQUIRED_ROOT_DOCS = {
    "AGENTS.md",
    "ARCHITECTURE.md",
    "DEFORM_KEY_KEYWORDS.md",
    "DOE_HTTP_API.md",
    "README.md",
    "interface_protocol.md",
    "接口参数文档.md",
}
EXTRA_DOCS = (ROOT / "src" / "mobo" / "api" / "BACKEND_STARTUP.md",)
LINK_PATTERN = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
ENV_PATTERN = re.compile(r"\bMOBO_[A-Z0-9_]+\b")


def _route_decorator(decorator: ast.expr) -> tuple[str, str] | None:
    if not isinstance(decorator, ast.Call) or not decorator.args:
        return None
    function = decorator.func
    if not isinstance(function, ast.Attribute) or function.attr not in {"get", "post"}:
        return None
    owner = function.value
    if not isinstance(owner, ast.Name) or owner.id not in {"app", "doe_api"}:
        return None
    path = decorator.args[0]
    if not isinstance(path, ast.Constant) or not isinstance(path.value, str):
        return None
    return function.attr.upper(), path.value


def extract_routes(root: Path) -> list[dict[str, str]]:
    routes: set[tuple[str, str]] = set()
    for relative in ("src/mobo/api/app.py", "src/mobo/api/handler.py"):
        tree = ast.parse((root / relative).read_text(encoding="utf-8"), filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    route = _route_decorator(decorator)
                    if route:
                        routes.add(route)
    return [{"method": method, "path": path} for method, path in sorted(routes)]


def extract_scripts(root: Path) -> dict[str, str]:
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    return dict(sorted(project.get("scripts", {}).items()))


def extract_packages(root: Path) -> list[str]:
    package_root = root / "src" / "mobo"
    return sorted(
        path.name for path in package_root.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    )


def extract_modules(root: Path) -> list[str]:
    package_root = root / "src" / "mobo"
    return sorted(
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def extract_environment_variables(root: Path) -> list[str]:
    variables: set[str] = set()
    candidates = [*root.glob("setup_env.*"), *(root / "src" / "mobo").rglob("*.py")]
    for path in candidates:
        variables.update(ENV_PATTERN.findall(path.read_text(encoding="utf-8")))
    return sorted(variables)


def extract_pytest_markers(root: Path) -> list[str]:
    with (root / "pyproject.toml").open("rb") as stream:
        markers = tomllib.load(stream)["tool"]["pytest"]["ini_options"].get("markers", [])
    return sorted(marker.split(":", 1)[0].strip() for marker in markers)


def build_surface(root: Path = ROOT) -> dict[str, Any]:
    return {
        "api_routes": extract_routes(root),
        "cli_scripts": extract_scripts(root),
        "environment_variables": extract_environment_variables(root),
        "modules": extract_modules(root),
        "packages": extract_packages(root),
        "pytest_markers": extract_pytest_markers(root),
    }


def _document_paths(root: Path) -> list[Path]:
    return [*sorted(root.glob("*.md")), *(root / path.relative_to(ROOT) for path in EXTRA_DOCS)]


def _check_required_docs(root: Path) -> list[str]:
    present = {path.name for path in root.glob("*.md")}
    return [f"缺少根文档：{name}" for name in sorted(REQUIRED_ROOT_DOCS - present)]


def _check_links(root: Path) -> list[str]:
    errors: list[str] = []
    for document in _document_paths(root):
        if not document.is_file():
            errors.append(f"缺少文档：{document.relative_to(root).as_posix()}")
            continue
        text = document.read_text(encoding="utf-8")
        for raw_target in LINK_PATTERN.findall(text):
            target = raw_target.strip().strip("<>").split(maxsplit=1)[0]
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = target.split("#", 1)[0]
            if target and not (document.parent / target).resolve().exists():
                relative = document.relative_to(root).as_posix()
                errors.append(f"失效链接：{relative} -> {raw_target}")
    return errors


def _check_document_coverage(root: Path, surface: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    api_text = (root / "DOE_HTTP_API.md").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    agents = (root / "AGENTS.md").read_text(encoding="utf-8")
    architecture = (root / "ARCHITECTURE.md").read_text(encoding="utf-8")

    for route in surface["api_routes"]:
        signature = f"{route['method']} {route['path']}"
        if signature not in api_text:
            errors.append(f"DOE_HTTP_API.md 未记录路由：{signature}")
    for script in surface["cli_scripts"]:
        if script not in readme:
            errors.append(f"README.md 未记录 CLI：{script}")
        if script not in agents:
            errors.append(f"AGENTS.md 常用命令未记录 CLI：{script}")
    for package in surface["packages"]:
        if f"`{package}`" not in architecture and f"mobo.{package}" not in architecture:
            errors.append(f"ARCHITECTURE.md 未记录包：mobo.{package}")
    for marker in surface["pytest_markers"]:
        if f"`{marker}`" not in agents:
            errors.append(f"AGENTS.md 未记录 pytest marker：{marker}")
    if "python tools/check_docs.py" not in readme or "python tools/check_docs.py" not in agents:
        errors.append("README.md 和 AGENTS.md 必须记录文档检查命令")

    current_protocols = [
        root / "README.md", root / "DOE_HTTP_API.md", root / "interface_protocol.md",
        root / "接口参数文档.md",
    ]
    for document in current_protocols:
        text = document.read_text(encoding="utf-8")
        for stale in ("/Users/", "../../data"):
            if stale in text:
                errors.append(f"{document.name} 包含过时路径：{stale}")
    return errors


def _check_snapshot(surface: dict[str, Any], snapshot: Path) -> list[str]:
    if not snapshot.is_file():
        return ["缺少 tools/docs_surface.json；运行 --update-snapshot 创建"]
    recorded = json.loads(snapshot.read_text(encoding="utf-8"))
    if recorded != surface:
        return [
            "代码公共表面与 tools/docs_surface.json 不一致；更新文档并人工确认后运行 "
            "python tools/check_docs.py --update-snapshot"
        ]
    return []


def check_repository(root: Path = ROOT, *, check_snapshot: bool = True) -> list[str]:
    surface = build_surface(root)
    errors = [
        *_check_required_docs(root),
        *_check_links(root),
        *_check_document_coverage(root, surface),
    ]
    if check_snapshot:
        errors.extend(_check_snapshot(surface, root / "tools" / "docs_surface.json"))
    return errors


def _write_snapshot(root: Path, surface: dict[str, Any]) -> None:
    destination = root / "tools" / "docs_surface.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(surface, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-snapshot", action="store_true",
        help="文档人工确认后更新公共表面快照",
    )
    args = parser.parse_args()
    surface = build_surface(ROOT)
    errors = check_repository(ROOT, check_snapshot=not args.update_snapshot)
    if errors:
        print("文档一致性检查失败：")
        for error in errors:
            print(f"- {error}")
        return 1
    if args.update_snapshot:
        _write_snapshot(ROOT, surface)
        print(f"已更新：{SNAPSHOT.relative_to(ROOT).as_posix()}")
    else:
        print("文档一致性检查通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
