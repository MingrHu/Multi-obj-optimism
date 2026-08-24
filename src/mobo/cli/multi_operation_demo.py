"""DEFORM 多工步自动化流水线演示脚本。

采样、初始化、求解和状态查询均可独立运行。任务配置会持久化到磁盘，后续步骤只需
使用同一个 ``_TASK_ID`` 即可续跑。
"""

import argparse

from mobo.automation.multi_operation_service import (
    create_multi_operation_sampling_task,
    init_multi_operation_task,
    query_multi_operation_status,
    run_multi_operation_task,
)
from mobo.common.paths import PROJECT_DIR

_TASK_ID = "tc4-three-operation-demo"
_ROOT_DIR = f"{PROJECT_DIR}/data/AUTO/tc4_multi_batch"


def _tc4_operations():
    source = f"{PROJECT_DIR}/data/AUTO/tc4_mo_rebuild/SourceKeys"
    return [
        {
            "name": "TC4 Stage 1",
            "template_key": f"{source}/1.KEY",
            "parameters": [
                {"name": "roll_tmp", "object": "workpiece", "range": [940, 1020]},
            ],
        },
        {
            "name": "TC4 Stage 2",
            "template_key": f"{source}/2.KEY",
            "parameters": [
                {"name": "roll_tmp", "object": "workpiece", "range": [930, 1010]},
            ],
        },
        {
            "name": "TC4 Stage 3",
            "template_key": f"{source}/3.KEY",
            "parameters": [
                {"name": "roll_tmp", "object": "workpiece", "range": [920, 1000]},
            ],
        },
    ]


def _print_status() -> None:
    print(query_multi_operation_status(_TASK_ID))


def sample_generate_test() -> None:
    """步骤一：联合生成三个工步各自的工艺参数样本。"""
    print(create_multi_operation_sampling_task(
        _TASK_ID,
        _tc4_operations(),
        _ROOT_DIR,
        method="lhs",
        n_samples=4,
    ))


def init_execution_test() -> None:
    """步骤二：初始化多工步计算目录和磁盘恢复状态。"""
    sample_file = f"{_ROOT_DIR}/{_TASK_ID}-multi-lhs.txt"
    print(init_multi_operation_task(
        _TASK_ID,
        sample_file,
        _tc4_operations(),
        f"{_ROOT_DIR}/runs",
        max_parallel_samples=1,
        keep_checkpoints=True,
    ))


def run_process_test() -> None:
    """步骤三：仅凭 task_id 运行或续跑全部多工步样本。"""
    print(run_multi_operation_task(_TASK_ID))


def query_status_test() -> None:
    """查看每个样本和每个工步的计算、失败及恢复状态。"""
    _print_status()


def main() -> None:
    parser = argparse.ArgumentParser(description="DEFORM 多工步批处理演示")
    parser.add_argument("action", choices=["sample", "init", "run", "status"])
    action = parser.parse_args().action
    {
        "sample": sample_generate_test,
        "init": init_execution_test,
        "run": run_process_test,
        "status": query_status_test,
    }[action]()


if __name__ == "__main__":
    main()
