"""TC4 碾环多工步批处理入口。"""

from pathlib import Path

from mobo.automation.multi_operation_service import (
    init_multi_operation_task,
    query_multi_operation_status,
    run_multi_operation_extract,
    run_multi_operation_task,
)
from mobo.automation.task_collection import TC4_RING_MULTI_TASK_1

_TASK = TC4_RING_MULTI_TASK_1
_TASK_ID = _TASK.task_id

# 采样方式只需在这里修改：
# - "lhs"：随机生成 _LHS_SAMPLE_COUNT 个样本；
# - "full"：按 _FULL_LEVEL_NUMS 生成全因子样本。
_SAMPLE_METHOD = "lhs"
_LHS_SAMPLE_COUNT = 256
_FULL_LEVEL_NUMS = (1, 1, 1, 1, 1, 1)

# 只处理完整采样 TXT 的指定行范围：从 0 开始，结束下标不包含。
# 例如 300 个样本分两台机器：机器 A 使用 (0, 200)，机器 B 使用 (200, 300)。
# 使用 (0, None) 表示处理全部样本，并保持历史状态文件名。
_SAMPLE_START = 0
_SAMPLE_END = None


def _sample_file() -> str:
    suffix = "lhs" if _SAMPLE_METHOD == "lhs" else "fullfactorial"
    return str(_TASK.sample_dir / f"{_TASK_ID}-{suffix}.txt")


def _generate_samples() -> str:
    if _SAMPLE_METHOD == "lhs":
        return _TASK.generate_samples(method="lhs", n_samples=_LHS_SAMPLE_COUNT)
    if _SAMPLE_METHOD == "full":
        return _TASK.generate_samples(method="full", level_nums=_FULL_LEVEL_NUMS)
    raise ValueError(f"不支持的采样方式: {_SAMPLE_METHOD}")


def _ensure_sample_file() -> str:
    sample_file = _sample_file()
    if not Path(sample_file).is_file():
        return _generate_samples()
    return sample_file


def sample_generate_test() -> None:
    """按照文件顶部的采样配置重新生成样本。"""
    print(_generate_samples())


def generate_keyfile_test() -> None:
    """初始化任务并预生成指定 TXT 行范围内全部工步的参数化 KEY。"""
    print(init_multi_operation_task(
        _TASK_ID,
        _ensure_sample_file(),
        _TASK.operation_configs(),
        str(_TASK.run_dir),
        max_parallel_samples=24,
        keep_checkpoints=True,
        incremental=True,
        sample_start=_SAMPLE_START,
        sample_end=_SAMPLE_END,
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
    sample_generate_test()
    # generate_keyfile_test()
    # run_process_test()
    # extra_data_test()
    # status_test()
