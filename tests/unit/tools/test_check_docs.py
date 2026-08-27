"""文档知识库一致性检查器测试。"""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "check_docs.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_docs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_documentation_is_consistent():
    checker = _load_checker()
    assert checker.check_repository(ROOT) == []
