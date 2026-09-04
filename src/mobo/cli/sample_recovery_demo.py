"""单工步参数化 KEY 反向恢复样本文件的演示入口。"""

from pathlib import Path

from mobo.automation.task_collection import get_single_operation_task_definition
from mobo.utils.sample_recovery import recover_samples


def recover_sample_test(
    task_id: str,
    *,
    key_dir: str | None = None,
    output_file: str | None = None,
    overwrite: bool = False,
) -> str:
    """按任务 ID 恢复样本；未传路径时使用该任务的默认工作目录。"""
    task = get_single_operation_task_definition(task_id)
    source = Path(key_dir) if key_dir is not None else task.workspace / "input_keys"
    destination = (
        Path(output_file)
        if output_file is not None
        else task.workspace / "samples" / f"{task_id}-recovered.txt"
    )
    recovered = recover_samples(
        task_id,
        source,
        destination,
        overwrite=overwrite,
    )
    print(f"样本恢复完成: {recovered}")
    return str(recovered)


if __name__ == "__main__":
    recover_sample_test("gh4169-ring-single-task-1")
