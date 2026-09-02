"""结果数据集提取。

将求解得到的 DB 文件逐步导出为 KEY 文件，再按目标调用
:class:`~mobo.automation.config.DeformConfig` 映射的提取函数，汇总为数据集 txt。
目标含文本类（应力/载荷/晶粒，逐帧解析）与几何类（碾环内/外圈圆度，按 KEY 文件
几何计算），统一由 ``TAR_FUNC`` 的适配器处理，本模块只做编排。
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, List, Sequence

from mobo.common.logging import logger
from .config import DeformConfig
from .keyfile import derive_output_path, read_key_frames
from .solver import db_to_key, query_db_steps


def _export_key(db_file: str, key_file: str, step: str) -> str:
    """导出一个确定存在的 DB 结果步，并拒绝静默产生空文件。"""
    if os.path.isfile(key_file) and os.path.getsize(key_file) > 0:
        return key_file
    db_to_key(db_file, key_file, step)
    if not os.path.isfile(key_file) or os.path.getsize(key_file) == 0:
        label = "终态" if step == "" else f"第 {step} 步"
        db_size = os.path.getsize(db_file) if os.path.isfile(db_file) else -1
        key_size = os.path.getsize(key_file) if os.path.isfile(key_file) else -1
        logger.error(
            f"DEFORM KEY 导出失败: db={db_file}, db_size={db_size}, step={step or 'latest'}, "
            f"key={key_file}, key_size={key_size}; 前处理器原始输出见 deform_operation.log"
        )
        raise FileNotFoundError(f"DEFORM 未导出{label} KEY: {key_file}")
    return key_file


def export_terminal_key(db_file: str, save_dir: str) -> str:
    """使用 DEFORM 数据库默认最新结果集导出终态 KEY。"""
    os.makedirs(save_dir, exist_ok=True)
    key_file = derive_output_path(db_file, save_dir, "_terminal", "KEY")
    return _export_key(db_file, key_file, "")


def export_saved_step_keys(db_file: str, save_dir: str) -> List[str]:
    """查询并导出 DB 中实际存在的全部保存步，不猜测连续步号。"""
    os.makedirs(save_dir, exist_ok=True)
    key_files: List[str] = []
    steps = query_db_steps(db_file)
    existing_count = sum(
        1
        for step in steps
        if os.path.isfile(derive_output_path(db_file, save_dir, f"_step_{step}", "KEY"))
        and os.path.getsize(
            derive_output_path(db_file, save_dir, f"_step_{step}", "KEY")
        ) > 0
    )
    logger.info(
        f"DB 保存步 KEY 提取开始: db={db_file}, total_steps={len(steps)}, "
        f"existing_keys={existing_count}, pending={len(steps) - existing_count}, "
        f"first_step={steps[0]}, last_step={steps[-1]}, output_dir={save_dir}"
    )
    for position, step in enumerate(steps, 1):
        key_file = derive_output_path(db_file, save_dir, f"_step_{step}", "KEY")
        try:
            key_files.append(_export_key(db_file, key_file, str(step)))
        except Exception:
            logger.error(
                f"DB 保存步 KEY 提取中断: db={db_file}, step={step}, "
                f"position={position}/{len(steps)}, exported={len(key_files)}, "
                f"remaining={len(steps) - position}"
            )
            raise
    logger.info(
        f"DB 保存步 KEY 提取完成: db={db_file}, exported_or_reused={len(key_files)}"
    )
    return key_files


def _extract_values(
    db_file: str,
    step_dir: str,
    target_table: List[List[Any]],
    in_progress: Sequence[bool],
) -> List[str]:
    """终态目标读取最新帧；全过程目标读取 DB 实际保存的全部帧。"""
    target_names, object_names, select_components = target_table
    terminal_key = export_terminal_key(db_file, step_dir)
    terminal_frames = read_key_frames([terminal_key])
    saved_keys: List[str] = []
    saved_frames: List[List[str]] = []
    if any(in_progress):
        saved_keys = export_saved_step_keys(db_file, step_dir)
        saved_frames = read_key_frames(saved_keys)

    values: List[str] = []
    for index, target_name in enumerate(target_names):
        extractor = DeformConfig.get_target_function(target_name)
        if extractor is None:
            raise ValueError(f"未知目标提取器: {target_name}")
        object_id = DeformConfig.get_object_id(object_names[index])
        use_progress = bool(in_progress[index])
        values.append(extractor(
            saved_keys if use_progress else [terminal_key],
            saved_frames if use_progress else terminal_frames,
            object_id,
            use_progress,
            select_components[index],
        ))
    return values


def extract_dataset(
    db_files: Sequence[str],
    key_export_dir: str,
    param_table: List[List[str]],
    target_table: List[List[Any]],
    in_progress: Sequence[bool],
    result_dir: str,
) -> str:
    """从一批 DB 文件提取目标值，汇总为数据集 txt 文件。

    :param db_files: 结果 DB 文件路径序列
    :param key_export_dir: 逐步 KEY 导出的根目录
    :param param_table: 参数表（前两行为表头，其后每行对应一个样本的工艺参数）
    :param target_table: 目标表 ``[[目标名...], [对象名...], [select_component...]]``
    :param in_progress: 每个目标是否走全过程提取
    :param result_dir: 数据集输出目录
    :return: 输出数据集文件完整路径
    """
    # 新建数据集
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    out_path = os.path.join(result_dir, f"{timestamp}_result.txt")

    # 每行 = 工艺参数列 + 目标列（制表符分隔，无行号列、无表头），
    # 与 surrogate.load_and_preprocess_data 的 (sep='\t', header=None) 读取约定对齐
    with open(out_path, "w", encoding="utf-8") as f:
        for i, db_file in enumerate(db_files):
            logger.info(f"当前提取的文件为：{db_file}")
            try:
                step_dir = os.path.join(key_export_dir, str(i))
                # 样本工艺参数（param_table 前两行为表头，样本从第 2 行起）
                row: List[Any] = list(param_table[i + 2])
                row.extend(_extract_values(
                    db_file, step_dir, target_table, in_progress
                ))
                line = "\t".join(map(str, row)) + "\n"
                logger.info(line)
                # 追加
                f.write(line)
                f.flush()
            except Exception as e:
                logger.error(f"数据提取失败！！！:{str(e)}")
                continue


    return out_path


def extract_dataset_row(
    db_file: str,
    sample_index: int,
    key_export_dir: str,
    parameters: Sequence[Any],
    target_table: List[List[Any]],
    in_progress: Sequence[bool],
) -> List[Any]:
    """从一个已完成 DB 导出 KEY 并生成一行数据，供增量批处理调用。"""
    step_dir = os.path.join(key_export_dir, str(sample_index))
    row: List[Any] = list(parameters)
    row.extend(_extract_values(db_file, step_dir, target_table, in_progress))
    return row


__all__ = [
    "export_saved_step_keys",
    "export_terminal_key",
    "extract_dataset",
    "extract_dataset_row",
]
