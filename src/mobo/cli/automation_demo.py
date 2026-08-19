"""
    DEFORM 自动化流水线演示脚本

    每一步都只需 ``_TASK_ID`` 即可独立调用：``init_execution_task`` 会把路径、参数表、
    目标表等必要输入落盘到 ``data/tasks/<task_id>/state.json``；之后 ``run_process_test`` /
    ``extra_data_test`` 仅凭 task_id 就能从磁盘续跑，无需重新传入这些参数。
"""

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


def _print_status(task_id: str) -> None:
    """打印当前任务状态（从 state.json 读取）。"""
    print(query_execution_status(task_id))


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
    """演示：初始化执行任务（落盘输入参数并生成 KEY 文件）。"""
    # 测试碾环
    # 1 输入参数表
    param_table = [["roll_tmp", "roll_tmp", "roll_tmp", "pressure_roll_speed_upper","pressure_roll_speed_lower","driving_roll_rad_speed"],
                   ["workpiece", "driving_roll", "pressure_roll", "pressure_roll", "pressure_roll", "driving_roll"]]
    # 2 提取目标表：第 0 行目标名、第 1 行对象名、第 2 行 select_component（分量）
    target_table = [["load", "grain_morph"],
                    ["driving_roll", "workpiece"],
                    [2, 1]]
    in_progress = [True, False]
    paths_config = {
        "smp_file": f"{PROJECT_DIR}/data/TEST/smp.txt",
        "std_key_file": f"{PROJECT_DIR}/data/keyfile/RINGROLL.KEY",
        "temp_key_path": f"{PROJECT_DIR}/data/TEST/temp_key",
        "res_db_path": f"{PROJECT_DIR}/data/TEST/res_db",
        "res_key_path": f"{PROJECT_DIR}/data/TEST/res_key",
        "res_txt_path": f"{PROJECT_DIR}/data/TEST/res_txt",
        "process_info_file":f"{PROJECT_DIR}/data/TEST/process_info.json"
    }

    print(init_execution_task(_TASK_ID, paths_config, param_table, target_table, in_progress, 3400))


def run_process_test() -> None:
    """演示：仅凭 task_id 推进求解阶段（参数从 state.json 续跑）。"""
    print(run_execution_step(_TASK_ID))


def extra_data_test() -> None:
    """演示：仅凭 task_id 推进数据提取阶段（参数从 state.json 续跑）。"""
    # 提取前先按任务信息校正结果 DB 目录序号（历史乱序自动纠正，已对齐则无改动）
    # print(align_result_db_dirs(_TASK_ID))
    print(run_extract_data(_TASK_ID))


if __name__ == "__main__":
    # 每一步可单独运行，只要 _TASK_ID 一致即可接着上一步继续
    # sample_generate_test()
    # generate_keyfile_test()
    # run_process_test()
    extra_data_test()
