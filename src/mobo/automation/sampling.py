"""实验设计（DOE）采样。

提供拉丁超立方（LHS）与全因子两种采样方法，生成 DEFORM 求解所需的工艺参数样本
（制表符分隔、保留两位小数的 txt 文件）。本模块为纯数据处理，无任何子进程依赖，
可独立测试。

采样算法与原实现保持一致：
- LHS：在参数区间内生成 LHS 样本并追加所有边界组合，去重后保存；
- 全因子：每个参数按各自水平数等间隔取值，做笛卡尔积。
"""

from __future__ import annotations

import os
from itertools import product
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from pydoe import lhs

from mobo.common.logging import logger

ParamRanges = Dict[str, Tuple[float, float]]


def lhs_samples(n_samples: int, param_ranges: ParamRanges) -> pd.DataFrame:
    """生成拉丁超立方采样样本（不含边界组合）。

    :param n_samples: LHS 样本数
    :param param_ranges: 参数区间字典 ``{name: (low, high)}``
    :return: 采样结果 DataFrame（列为参数名，保留两位小数）
    """
    unit = lhs(len(param_ranges), samples=n_samples)  # [0,1) 区间样本
    scaled = np.zeros_like(unit)
    for i, (_, (low, high)) in enumerate(param_ranges.items()):
        scaled[:, i] = unit[:, i] * (high - low) + low
    df = pd.DataFrame(scaled, columns=list(param_ranges.keys()))
    return df.round(2)


def boundary_samples(param_ranges: ParamRanges) -> pd.DataFrame:
    """生成所有参数上下界的笛卡尔组合（边界样本）。

    :param param_ranges: 参数区间字典
    :return: 边界组合 DataFrame
    """
    low_high = [[low, high] for (low, high) in param_ranges.values()]
    combinations = list(product(*low_high))
    return pd.DataFrame(combinations, columns=list(param_ranges.keys()))


def generate_lhs(n_samples: int, param_ranges: ParamRanges) -> pd.DataFrame:
    """生成 LHS 样本并追加边界组合，去重后返回（两位小数字符串）。

    :param n_samples: LHS 样本数
    :param param_ranges: 参数区间字典
    :return: 合并去重后的样本 DataFrame（元素为两位小数字符串）
    """
    df = lhs_samples(n_samples, param_ranges)
    boundary = boundary_samples(param_ranges)
    combined = pd.concat([df, boundary], ignore_index=True).drop_duplicates(
        subset=list(param_ranges.keys())
    )
    return combined.applymap(lambda x: f"{x:.2f}") # type: ignore


def generate_full_factorial(param_ranges: ParamRanges, level_nums: Sequence[int]) -> pd.DataFrame:
    """生成全因子样本：每个参数按各自水平数等间隔取值后做笛卡尔积。

    :param param_ranges: 参数区间字典
    :param level_nums: 各参数的水平数，顺序与 ``param_ranges`` 一致
    :return: 全因子样本 DataFrame（保留两位小数）
    """
    keys = list(param_ranges.keys())
    if len(level_nums) != len(keys):
        raise ValueError(
            f"level_nums 长度({len(level_nums)})必须等于参数个数({len(keys)})"
        )

    param_levels = {}
    for i, key in enumerate(keys):
        low, high = param_ranges[key]
        param_levels[key] = np.linspace(low, high, level_nums[i])

    combinations = list(product(*param_levels.values()))
    df = pd.DataFrame(combinations, columns=keys)
    return df.round(2)


def save_samples(df: pd.DataFrame, method_tag: str, save_dir: str) -> str:
    """把样本保存为制表符分隔、无表头的 txt 文件。

    :param df: 样本 DataFrame
    :param method_tag: 采样方法标识（用于文件名 ``IN<tag>.txt``）
    :param save_dir: 保存目录（不存在则创建）
    :return: 输出文件完整路径
    """
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"IN{method_tag}.txt")
    df.to_csv(out_path, sep="\t", index=False, header=False)
    logger.info(f"采样数据已保存至 {out_path}，共 {len(df)} 个样本")
    return out_path


def generate_samples(
    method: str,
    param_ranges: ParamRanges,
    save_dir: str,
    n_samples: int = 0,
    level_nums: Sequence[int] = (),
) -> str:
    """按方法生成并保存样本。

    :param method: 采样方法，``"lhs"`` 或 ``"full"``
    :param param_ranges: 参数区间字典
    :param save_dir: 保存目录
    :param n_samples: LHS 样本数（method="lhs" 时使用）
    :param level_nums: 各参数水平数（method="full" 时必填）
    :return: 输出文件完整路径
    :raises ValueError: 方法不支持或 full 采样缺少 level_nums
    """
    if method == "lhs":
        df = generate_lhs(n_samples, param_ranges)
        return save_samples(df, "lhs", save_dir)
    if method == "full":
        if not level_nums:
            raise ValueError("full 采样必须提供 level_nums")
        df = generate_full_factorial(param_ranges, level_nums)
        return save_samples(df, "fullfactorial", save_dir)
    raise ValueError(f"不支持的采样方法: {method}")


__all__ = [
    "ParamRanges",
    "lhs_samples",
    "boundary_samples",
    "generate_lhs",
    "generate_full_factorial",
    "save_samples",
    "generate_samples",
]
