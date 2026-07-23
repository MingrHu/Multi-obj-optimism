"""DEFORM 自动化流水线演示脚本。

从 ``AutoScript/auto_script_test.py`` 搬入的手写 demo（非 pytest），演示
采样 → 初始化 → 求解 → 提取的完整调用链。使用 :mod:`mobo.automation.service`
的服务接口，路径统一取自 :mod:`mobo.common.paths`。

平台边界：真实执行依赖 Windows 平台的 DEFORM 环境；此脚本主要用于演示接口。
非 Windows 环境下 :class:`~mobo.automation.pipeline.ForgingTask` 支持 ``dry_run``
仅推进状态，服务层默认走真实调用，故此脚本在无 DEFORM 时会在求解阶段报错。
"""

import time

from mobo.automation.service import (
    create_sampling_task,
    init_execution_task,
    query_execution_status,
    run_execution_step,
    run_extract_data,
)
from mobo.common.paths import PROJECT_DIR

# 用于示例的固定任务 ID
_TASK_ID = "2026-04-10-demo"


def _wait_until_done(task_id: str, poll_interval: float = 3.0) -> None:
    """轮询任务状态直到底层 TaskStatus 回到 DONE（``status == "0"``）。"""
    while True:
        status = query_execution_status(task_id)
        if status["status"] == "0":
            break
        time.sleep(poll_interval)


def sample_generate_test() -> None:
    """演示：用 LHS 生成工艺参数样本。"""
    param_ranges = {
        "temp1": (875.0, 965.0),  # 工件温度范围 (℃)
        "temp2": (300.0, 700.0),  # 上模具温度范围 (℃)
        "temp3": (300.0, 700.0),  # 下模具温度范围 (℃)
        "speed": (10.0, 50.0),    # 锻造速度范围 (mm/s)
    }
    save_dir = f"{PROJECT_DIR}/data/TEST"
    print(create_sampling_task("1001", save_dir, "lhs", param_ranges, 1000))


def generate_keyfile_test() -> None:
    """演示：初始化执行任务并等待 KEY 文件生成完成。"""
    param_table = [["temp", "temp", "temp", "speed"],
                   ["workpiece", "topdie", "butdie", "topdie"]]
    target_table = [["grain", "load"],
                    ["workpiece", "topdie"]]
    in_progress = [False, True]
    paths_config = {
        "smp_file": f"{PROJECT_DIR}/data/AUTO/smp.txt",
        "std_key_file": f"{PROJECT_DIR}/data/AUTO/MODEL.KEY",
        "temp_key_path": f"{PROJECT_DIR}/data/AUTO/temp_key",
        "res_db_path": f"{PROJECT_DIR}/data/AUTO/res_db",
        "res_key_path": f"{PROJECT_DIR}/data/AUTO/res_key",
        "res_txt_path": f"{PROJECT_DIR}/data/AUTO/res_txt",
    }

    print(init_execution_task(_TASK_ID, paths_config, param_table, target_table, in_progress, 100))
    _wait_until_done(_TASK_ID)


def run_process_test() -> None:
    """演示：推进求解阶段并等待完成。"""
    print(run_execution_step(_TASK_ID))
    _wait_until_done(_TASK_ID)


def extra_data_test() -> None:
    """演示：推进数据提取阶段并等待完成。"""
    print(run_extract_data(_TASK_ID))
    _wait_until_done(_TASK_ID)


if __name__ == "__main__":
    # sample_generate_test()
    generate_keyfile_test()
    run_process_test()
    extra_data_test()
