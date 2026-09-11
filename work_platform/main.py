"""Run the MOBO desktop UI from the repository root."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for entry in (str(SRC), str(Path(__file__).resolve().parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

try:
    from mobo_ui.app import main  # noqa: E402
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("PySide6"):
        raise SystemExit(
            "未安装 MOBO Desktop 的 GUI 依赖 PySide6。\n"
            "请在仓库根目录运行：\n"
            "  .\\scripts\\setup_env.ps1 -Recreate -WithGui\n"
            "然后重新执行：\n"
            "  .\\.venv\\Scripts\\python.exe .\\work_platform\\main.py"
        ) from None
    raise


if __name__ == "__main__":
    raise SystemExit(main())
