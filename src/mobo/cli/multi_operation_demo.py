"""TC4 碾环多工步批处理入口。"""

from mobo.automation.multi_operation_service import (
    init_multi_operation_task,
    query_multi_operation_status,
    run_multi_operation_extract,
    run_multi_operation_task,
)
from mobo.automation.task_collection import TC4_RING_MULTI_TASK_1

_TASK = TC4_RING_MULTI_TASK_1
_TASK_ID = _TASK.task_id


def _sample_file() -> str:
    return str(_TASK.sample_dir / f"{_TASK_ID}-lhs.txt")


def sample_generate_test() -> None:
    """生成工步 2、3 的联合工艺参数样本。"""
    print(_TASK.generate_samples(method="lhs", n_samples=256))


def init_task_test() -> None:
    """初始化任务、工作目录和磁盘恢复状态。"""
    print(init_multi_operation_task(
        _TASK_ID,
        _sample_file(),
        _TASK.operation_configs(),
        str(_TASK.run_dir),
        max_parallel_samples=24,
        keep_checkpoints=True,
        incremental=True,
    ))


def run_process_test() -> None:
    """运行或从磁盘状态续跑多工步任务。"""
    print(run_multi_operation_task(_TASK_ID))


def extra_data_test() -> None:
    """提取已完成样本并生成结果数据集。"""
    print(run_multi_operation_extract(_TASK_ID))


def status_test() -> None:
    """查看任务及各样本、各工步状态。"""
    print(query_multi_operation_status(_TASK_ID))


if __name__ == "__main__":
    # 每一步可单独运行，只要 _TASK_ID 一致即可接着上一步继续
    # sample_generate_test()
    init_task_test()
    # run_process_test()
    # extra_data_test()
    # status_test()
