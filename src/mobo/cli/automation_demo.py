"""
    DEFORM 自动化流水线演示脚本
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
_TASK_ID = "2026-07-28-demo"


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
        "roll_tmp": (940.0, 1020.0),  # 碾环工件温度范围 (℃)
        "driving_roll_tmp": (200.0, 260.0),  # 驱动辊具温度范围 (℃)
        "pressure_roll_tmp": (200.0, 260.0),  # 压力辊具温度范围 (℃)
        "pressure_roll_speed_upper": (2.0, 2.3),    # 锻造上限速度范围 (mm/s)
        "pressure_roll_speed_lower": (0.2, 1.9),    # 锻造下限速度范围 (mm/s)
        "driving_roll_rad_speed": (0.5, 1.5),    # 驱动辊角速度范围 (rad/s)
    }
    save_dir = f"{PROJECT_DIR}/data/TEST"
    print(create_sampling_task(_TASK_ID, save_dir, "lhs", param_ranges, 200))


def generate_keyfile_test() -> None:
    """演示：初始化执行任务并等待 KEY 文件生成完成。"""
    # 测试碾环
    param_table = [["roll_tmp", "roll_tmp", "roll_tmp", "pressure_roll_speed_upper","pressure_roll_speed_lower","driving_roll_rad_speed"],
                   ["workpiece", "driving_roll", "pressure_roll", "pressure_roll", "pressure_roll", "driving_roll"]]
    # target_table = [["grain", "load"],
    #                 ["workpiece", "topdie"]]
    # 碾环内/外圈圆度也走统一的提取流程（对象为 workpiece 环件）
    target_table = [["roundness_inner", "roundness_outer"],
                    ["workpiece", "workpiece"]]
    in_progress = [False, False]
    paths_config = {
        "smp_file": f"{PROJECT_DIR}/data/TEST/smp.txt",
        "std_key_file": f"{PROJECT_DIR}/data/keyfile/RINGROLL.KEY",
        "temp_key_path": f"{PROJECT_DIR}/data/TEST/temp_key",
        "res_db_path": f"{PROJECT_DIR}/data/TEST/res_db",
        "res_key_path": f"{PROJECT_DIR}/data/TEST/res_key",
        "res_txt_path": f"{PROJECT_DIR}/data/TEST/res_txt",
    }

    print(init_execution_task(_TASK_ID, paths_config, param_table, target_table, in_progress, 3400))
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
    # generate_keyfile_test()
    # run_process_test()
    extra_data_test()
