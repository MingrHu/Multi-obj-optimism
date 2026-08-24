"""集中式路径解析测试。"""

import importlib

import mobo.common.paths as paths


def test_project_dir_contains_data():
    """项目根目录应包含 data 目录（或 pyproject.toml）。"""
    assert (paths.PROJECT_DIR / "data").is_dir() or (paths.PROJECT_DIR / "pyproject.toml").exists()


def test_data_subdirs_derive_from_data_dir():
    assert paths.MODELS_DIR == paths.DATA_DIR / "models"
    assert paths.TEST_DIR == paths.DATA_DIR / "TEST"
    assert paths.KEY_FILE_DIR == paths.DATA_DIR / "keyfile"


def test_model_family_dir():
    assert paths.model_family_dir("PRG") == paths.MODELS_DIR / "PRG"
    assert paths.model_family_dir("DNN") == paths.MODELS_DIR / "DNN"


def test_env_override(monkeypatch, tmp_path):
    """MOBO_PROJECT_DIR / MOBO_DATA_DIR 环境变量应覆盖默认推导。"""
    proj = tmp_path / "proj"
    data = tmp_path / "mydata"
    proj.mkdir()
    data.mkdir()
    monkeypatch.setenv("MOBO_PROJECT_DIR", str(proj))
    monkeypatch.setenv("MOBO_DATA_DIR", str(data))

    reloaded = importlib.reload(paths)
    try:
        assert reloaded.PROJECT_DIR == proj.resolve()
        assert reloaded.DATA_DIR == data.resolve()
        assert reloaded.MODELS_DIR == data.resolve() / "models"
    finally:
        # 还原模块状态，避免影响其他用例
        monkeypatch.delenv("MOBO_PROJECT_DIR", raising=False)
        monkeypatch.delenv("MOBO_DATA_DIR", raising=False)
        importlib.reload(paths)
