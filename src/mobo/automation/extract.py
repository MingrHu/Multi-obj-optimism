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
from .solver import db_to_key


def _export_all_steps(db_file: str, save_dir: str, max_step: int) -> List[str]:
    """把一个 DB 文件的 0..max_step-1 步全部导出为 KEY 文件。

    :param db_file: DB 文件路径
    :param save_dir: KEY 导出目录
    :param max_step: 最大步数
    :return: 各步 KEY 文件路径列表
    """
    os.makedirs(save_dir, exist_ok=True)
    key_files: List[str] = []
    for step in range(max_step):
        key_file = derive_output_path(db_file, save_dir, str(step), "KEY")
        key_files.append(key_file)
        while not os.path.exists(key_file):
            db_to_key(db_file, key_file, str(step))
    return key_files


def extract_dataset(
    db_files: Sequence[str],
    key_export_dir: str,
    max_step: int,
    param_table: List[List[str]],
    target_table: List[List[str]],
    in_progress: Sequence[bool],
    result_dir: str,
) -> str:
    """从一批 DB 文件提取目标值，汇总为数据集 txt 文件。

    :param db_files: 结果 DB 文件路径序列
    :param key_export_dir: 逐步 KEY 导出的根目录
    :param max_step: 每个 DB 的最大步数
    :param param_table: 参数表（前两行为表头，其后每行对应一个样本的工艺参数）
    :param target_table: 目标表 ``[[目标名...], [对象名...], [select_component...]]``
    :param in_progress: 每个目标是否走全过程提取
    :param result_dir: 数据集输出目录
    :return: 输出数据集文件完整路径
    """
    target_names, object_names, select_component = (
        target_table[0], target_table[1], target_table[2]
    )
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
                # 导出key文件
                key_files = _export_all_steps(db_file, step_dir, max_step)
                frames = read_key_frames(key_files)

                # 样本工艺参数（param_table 前两行为表头，样本从第 2 行起）
                row: List[Any] = list(param_table[i + 2])
                for idx, target_name in enumerate(target_names):
                    # 1 提取函数
                    extractor = DeformConfig.get_target_function(target_name)
                    # 2 提取对象
                    object_id = DeformConfig.get_object_id(object_names[idx])
                    row.append(extractor(
                        key_files, frames, object_id,
                        in_progress[idx], select_component[idx],
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


__all__ = ["extract_dataset"]
