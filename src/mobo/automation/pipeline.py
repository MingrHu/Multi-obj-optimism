"""
DEFORM 自动化流水线。
编排「采样 → 生成 KEY → 求解 → 提取数据集」的完整流程。核心是 :class:`ForgingTask`
任务类（取代旧的 ``Doe_execute``），每个阶段异步执行并维护 :class:`TaskStatus` 状态。
"""

from __future__ import annotations

import os
import threading
from enum import IntEnum
from typing import Callable, List, Optional, Sequence, Tuple

from mobo.common.logging import logger
from .extract import extract_dataset
from .keyfile import derive_output_path, generate_key_files
from .sampling import ParamRanges, generate_samples
from .solver import DeformSolver, key_to_db_batch


class TaskStatus(IntEnum):
    """任务状态：与旧实现的整数取值保持一致（done=0 / running=1 / failed=-1）。"""

    DONE = 0
    RUNNING = 1
    FAILED = -1


def generate_sample_file(
    task_id:str,
    method: str,
    param_ranges: ParamRanges,
    save_dir: str,
    n_samples: int = 0,
    level_nums: Sequence[int] = (),
) -> str:
    """生成工艺参数样本文件（LHS 或全因子）
    
    :param task_id: 任务id
    :param method: ``"lhs"`` 或 ``"full"``
    :param param_ranges: 参数区间字典
    :param save_dir: 保存目录
    :param n_samples: LHS 样本数
    :param level_nums: 全因子各参数水平数
    :return: 样本文件路径
    """
    return generate_samples(task_id,method, param_ranges, save_dir, n_samples, level_nums)


class ForgingTask:
    """锻造工艺 DOE 求解任务

    :param sample_file: 样本文件路径（工艺参数取值，每行一个样本）
    :param template_key: 模板 KEY 文件路径
    :param temp_key_dir: 生成的输入 KEY 保存目录
    :param result_db_dir: 结果 DB 保存目录
    :param result_key_dir: 结果 KEY（逐步导出）保存目录
    :param result_txt_dir: 数据集输出目录
    :param param_table: 工艺参数固定表头 ``[[参数名...], [对象名...]]``（2×n）
    :param target_table: 目标固定表头 ``[[目标名...], [对象名...], [select_component...]]``（3×m）
    :param in_progress: 每个目标是否走全过程提取（1×m）
    :param max_step: KEY 求解过程最大步数
    :param dry_run: 为 True 时只推进状态、不真正调用 DEFORM（用于非 Windows/测试）
    :param key_files: 内存记录的生成批量key文件
    :param db_files: 内存记录的结果DB文件
    :param process_num: 当前已经运行完成的db数量

    """

    def __init__(
        self,
        sample_file: str,
        template_key: str,
        temp_key_dir: str,
        result_db_dir: str,
        result_key_dir: str,
        result_txt_dir: str,
        param_table: List[List[str]],
        target_table: List[List[str]],
        in_progress: List[bool],
        max_step: int,
        process_info_file:str,
        *,
        dry_run: bool = False,
        max_parallel: int = 24,
    ) -> None:
        self.sample_file = sample_file
        self.template_key = template_key
        self.temp_key_dir = temp_key_dir
        self.result_db_dir = result_db_dir
        self.result_key_dir = result_key_dir
        self.result_txt_dir = result_txt_dir
        self.param_table = param_table
        self.target_table = target_table
        self.in_progress = in_progress
        self.max_step = max_step
        self.dry_run = dry_run
        self.max_parallel = max_parallel

        # 中间产物
        self.key_files: List[str] = []
        self.db_files: List[str] = []

        # 任务运行时信息存放路径
        self.process_info_file = process_info_file

        self.status: TaskStatus = TaskStatus.DONE

    def _run_async(self, name: str, work: Callable[[], None]) -> Optional[threading.Thread]:
        """在后台线程执行一个阶段，并维护状态转移
        :param name: 阶段名称（用于日志）
        :param work: 阶段实际工作（无参可调用）
        :return: 启动的线程；若前置状态不满足则返回 None
        """
        if self.status != TaskStatus.DONE:
            logger.error(f"无法执行 {name}：上一阶段未完成（当前状态 {self.status.name}）")
            return None

        def runner() -> None:
            try:
                logger.info(f"{name} 开始")
                self.status = TaskStatus.RUNNING
                work()
                self.status = TaskStatus.DONE
                logger.info(f"✅ {name} 完成")
            except Exception as exc:
                self.status = TaskStatus.FAILED
                logger.error(f"❌ {name} 失败: {exc}")

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        return thread

    def load_samples_into_table(self) -> None:
        """把样本文件的每行数值追加到参数表（表头之后）。"""
        with open(self.sample_file, "r", encoding="utf-8") as f:
            for line in f:
                self.param_table.append(line.split())

    def generate_keys(self) -> Optional[threading.Thread]:
        """阶段一：把工艺参数写入模板，批量生成输入 KEY 文件（异步）。"""
        self.load_samples_into_table()

        def work() -> None:
            if self.dry_run:
                self.key_files = ["<dry-run>"]
                return
            self.key_files = generate_key_files(self.template_key, self.param_table, self.temp_key_dir)

        thread = self._run_async("生成 KEY 文件", work)
        if thread is not None:
            thread.join()  # 等待 KEY 文件生成完成，避免后续阶段找不到文件
        return

    def prepare_db_files(self) -> List[Tuple[str, str]]:
        """按结果 DB 目录约定重建 ``db_files``（确定性文件名，支持续跑）。

        每个输入 KEY 对应 ``result_db_dir/<i>/<stem>.DB``；已生成的直接复用。
        :return: 未生成 DB 的 (key_path, db_path) 列表，供 KEY→DB 转换使用
        """
        self.db_files = []
        pending: List[Tuple[str, str]] = []
        # 根据生成的key文件的名称重建结果db文件的路径
        for i, key_file in enumerate(self.key_files):
            db_dir = os.path.join(self.result_db_dir, str(i))
            os.makedirs(db_dir, exist_ok=True)
            db_file = derive_output_path(key_file, db_dir, "", "DB")
            self.db_files.append(db_file)
            if not os.path.exists(db_file):
                pending.append((key_file, db_file))
        return pending

    def run_solver(self) -> Optional[threading.Thread]:
        """阶段二：KEY→DB 转换并提交求解（异步）"""
        pending = self.prepare_db_files()
        key_paths = [k for k, _ in pending]
        db_paths = [d for _, d in pending]

        def work() -> None:
            if self.dry_run:
                return
            key_to_db_batch(key_paths, db_paths)
            DeformSolver(max_parallel=self.max_parallel).run_all(self.db_files)

        thread = self._run_async("求解运行", work)
        if thread is not None:
            thread.join()  # 等待求解
        return 

    def extract(self) -> Optional[threading.Thread]:
        """阶段三：从结果 DB 提取目标值，汇总数据集（异步）。"""

        def work() -> None:
            if self.dry_run:
                return
            extract_dataset(
                self.db_files,
                self.result_key_dir,
                self.max_step,
                self.param_table,
                self.target_table,
                self.in_progress,
                self.result_txt_dir,
            )

        thread = self._run_async("提取数据", work)
        if thread is not None:
            thread.join()  # 等待提取
        return 


__all__ = ["TaskStatus", "generate_sample_file", "ForgingTask"]
