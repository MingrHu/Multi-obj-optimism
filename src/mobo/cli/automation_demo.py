"""7050 碾环单工步任务演示入口。

任务的参数范围、模板 KEY、参数/目标表和工作目录均由
``RING_7050_SINGLE_TASK_1`` 统一提供。初始化后，求解、提取和状态查询
都可以仅凭任务 ID 跨进程续跑。
"""

from mobo.automation.service import (
    init_execution_task,
    query_execution_status,
    run_execution_step,
    run_extract_data,
)
from mobo.automation.task_collection import RING_7050_SINGLE_TASK_1

_TASK = RING_7050_SINGLE_TASK_1
_TASK_ID = _TASK.task_id


def _sample_file() -> str:
    """返回默认 LHS 样本文件路径。"""
    return str(_TASK.workspace / "samples" / f"{_TASK_ID}-lhs.txt")


def sample_generate_test(n_samples: int = 200) -> None:
    """根据单工步任务定义生成 LHS 样本。"""
    print(_TASK.generate_samples(method="lhs", n_samples=n_samples))


def generate_keyfile_test() -> None:
    """初始化可续跑任务并生成全部参数化 KEY。"""
    task = _TASK.build(_sample_file(), incremental=True)
    paths_config = {
        "smp_file": task.sample_file,
        "std_key_file": task.template_key,
        "temp_key_path": task.temp_key_dir,
        "res_db_path": task.result_db_dir,
        "res_key_path": task.result_key_dir,
        "res_txt_path": task.result_txt_dir,
        "process_info_file": task.process_info_file,
        "incremental_state_file": task.incremental_state_file,
        "incremental_output_file": task.incremental_output_file,
    }
    print(init_execution_task(
        _TASK_ID,
        paths_config,
        task.param_table,
        task.target_table,
        task.in_progress,
        incremental=True,
    ))


def run_process_test() -> None:
    """运行或从磁盘进度续跑 DEFORM 求解。"""
    print(run_execution_step(_TASK_ID))


def extra_data_test() -> None:
    """从已完成的 DB 提取目标并生成数据集。"""
    print(run_extract_data(_TASK_ID))


def status_test() -> None:
    """查询当前任务阶段和状态。"""
    print(query_execution_status(_TASK_ID))


if __name__ == "__main__":
    # 首次使用按顺序执行前四步；中断后可直接重新执行求解或提取。
    sample_generate_test(n_samples=250)
    generate_keyfile_test()
    # run_process_test()
    # extra_data_test()
    # status_test()
