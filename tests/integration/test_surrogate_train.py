"""集成测试：用示例数据集训练非 DNN 代理模型（输出重定向到 tmp）。"""

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def patch_models_dir(monkeypatch, tmp_path):
    """把 save_model 的输出目录重定向到 tmp，避免污染仓库 data/models。"""
    import mobo.surrogate.common as common
    monkeypatch.setattr(common, "MODELS_DIR", tmp_path)
    return tmp_path


def test_prg_fun_trains_and_saves(simulated_data_file, patch_models_dir):
    from mobo.surrogate.polynomial import prg_fun
    vars_out = ["1", "2", "3", "grain", "load"]
    prg_fun(simulated_data_file, vars_out, 3)
    # 两个目标各产出 model + scalers
    prg_dir = patch_models_dir / "PRG"
    assert (prg_dir / "grain_model.pkl").exists()
    assert (prg_dir / "load_model.pkl").exists()
    assert (prg_dir / "grain_scalers.pkl").exists()


def test_svr_fun_trains_and_saves(simulated_data_file, patch_models_dir):
    from mobo.surrogate.svr import svr_fun
    vars_out = ["1", "2", "3", "grain", "load"]
    svr_fun(simulated_data_file, vars_out, 3)
    assert (patch_models_dir / "SVR" / "grain_model.pkl").exists()


def test_rf_fun_trains_and_saves(simulated_data_file, patch_models_dir):
    from mobo.surrogate.random_forest import rf_run
    vars_out = ["1", "2", "3", "grain", "load"]
    rf_run(simulated_data_file, vars_out, 3)
    assert (patch_models_dir / "RF" / "grain_model.pkl").exists()
