"""DEFORM 子进程驱动。

通过子进程调用 DEFORM 的 ``DEF_PRE_64.exe``（KEY↔DB 转换）与 ``DEF_ARM_CTL.COM``
（提交求解）。**仅能在装有 DEFORM 的 Windows 环境真实运行**，其他平台仅供结构化
测试（打桩 :class:`subprocess.Popen`）。

与原实现的差异（不改变发送给 DEFORM 的命令串）：
- 命令输入文件改用 :mod:`tempfile` 生成唯一文件名，线程/并发安全，且不再污染当前
  工作目录；
- 操作日志写入集中式 ``LOGS_DIR/deform_operation.log``；
- 全局求解计数器封装进 :class:`DeformSolver`，取代模块级可变全局量。
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from queue import Queue
from typing import Any, List, Sequence

from mobo.common.logging import logger
from mobo.common.paths import LOGS_DIR

# DEFORM 可执行程序（需在 PATH 中，或设置对应环境变量）
DEF_PRE_64 = "DEF_PRE_64.exe"
DEF_ARM_CTL = "DEF_ARM_CTL.COM"

# DEFORM 操作日志路径（集中到 LOGS_DIR）
_OPERATION_LOG = os.path.join(str(LOGS_DIR), "deform_operation.log")


def _run_pre_with_commands(commands: str) -> None:
    """把命令串写入临时文件，用输入重定向驱动 DEF_PRE_64，并记录输出到日志。

    :param commands: 发送给 DEF_PRE_64 的完整命令串（含换行）
    """
    os.makedirs(str(LOGS_DIR), exist_ok=True)
    fd, cmd_file = tempfile.mkstemp(prefix="deform_pre_", suffix=".txt", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(commands)

        command = f'"{DEF_PRE_64}" < "{cmd_file}"'
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=True,
        )
        with open(_OPERATION_LOG, "a", encoding="utf-8") as log_file:
            while True:
                output = process.stdout.readline()  # type: ignore
                if not output and process.poll() is not None:
                    break
                if output:
                    log_file.write(output)
    finally:
        if os.path.exists(cmd_file):
            os.remove(cmd_file)


def key_to_db(key_path: str, db_path: str) -> None:
    """将 KEY 文件转换为 DB 文件。

    :param key_path: 输入 KEY 文件路径
    :param db_path: 输出 DB 文件路径
    """
    commands = f"E\n2\n1\n{key_path}\nE\nE\n7\n2\n{db_path}\nY\nE\nY\n"
    _run_pre_with_commands(commands)


def key_to_db_batch(key_paths: Sequence[str], db_paths: Sequence[str]) -> None:
    """批量将 KEY 文件转换为 DB 文件。

    :param key_paths: 输入 KEY 文件路径序列
    :param db_paths: 与之一一对应的输出 DB 文件路径序列
    """
    for key_path, db_path in zip(key_paths, db_paths):
        key_to_db(key_path, db_path)


def db_to_key(db_path: str, key_path: str, step: str = "") -> None:
    """将 DB 文件的指定步导出为 KEY 文件。

    :param db_path: 输入 DB 文件路径
    :param key_path: 输出 KEY 文件路径
    :param step: 导出的步数（需准确，否则 DEFORM 报错）
    """
    commands = f"E\n2\n2\n{db_path}\n{step}\nE\nE\n8\n{key_path}\nE\nY\n"
    _run_pre_with_commands(commands)


class DeformSolver:
    """DEFORM 求解调度器，封装并发求解计数。

    通过 ``DEF_ARM_CTL.COM`` 异步提交 DB 求解任务，并限制同时运行的进程数，
    取代原实现中裸露在模块级的全局计数器与锁。
    """

    def __init__(self, max_parallel: int = 12) -> None:
        """
        :param max_parallel: 最大并行求解进程数
        """
        self.max_parallel = max_parallel
        self._running = 0
        self._lock = threading.Lock()

    @property
    def running(self) -> int:
        """当前正在运行的求解进程数。"""
        with self._lock:
            return self._running

    @staticmethod
    def _feed_stdin(process: Any, content: str) -> None:
        """向子进程标准输入写入一行（附带 DEFORM 需要的短暂延时）。"""
        time.sleep(0.1)
        process.stdin.write(content + "\n")
        process.stdin.flush()

    def _solve_one(self, db_path: str, work_dir: str) -> None:
        """在独立线程中提交单个 DB 的求解。"""
        try:
            process = subprocess.Popen(
                DEF_ARM_CTL,
                stdin=subprocess.PIPE,
                shell=False,
                cwd=work_dir,
                text=True,
            )
            self._feed_stdin(process, db_path)
            self._feed_stdin(process, "B")
            process.wait()
            if process.stdin:
                process.stdin.close()
            logger.info(f"当前任务计算完成！请查看 {db_path} 结果")
        except Exception as exc:
            logger.error(f"求解进程出错: {exc}")
        finally:
            with self._lock:
                self._running -= 1

    def submit(self, db_path: str, work_dir: str) -> None:
        """异步提交一个求解任务（占用一个并发名额）。"""
        with self._lock:
            self._running += 1
            logger.info(f"当前正在计算的任务有：{self._running} 个")
        thread = threading.Thread(target=self._solve_one, args=(db_path, work_dir), daemon=True)
        thread.start()

    def run_all(self, db_paths: List[str]) -> None:
        """按最大并行数调度求解全部 DB 文件，直至全部完成。

        :param db_paths: 待求解的 DB 文件路径列表
        """
        task_queue: Queue = Queue()
        for i, db_path in enumerate(db_paths):
            task_queue.put((i + 1, db_path))

        while True:
            if self.running >= self.max_parallel:
                time.sleep(3)
                continue

            if task_queue.qsize() == 0 and self.running == 0:
                break
            if task_queue.qsize() == 0:
                time.sleep(60)
                continue

            file_num, db_path = task_queue.get()
            work_dir = os.path.dirname(db_path)
            stem = os.path.splitext(os.path.basename(db_path))[0]
            solve_target = os.path.join(work_dir, stem)

            self.submit(solve_target, work_dir)
            logger.info(f"开始计算第 {file_num} 个，结果 DB 将保存至 {work_dir}")
            # 防止过快提交导致计数尚未更新
            time.sleep(5)


__all__ = [
    "DEF_PRE_64",
    "DEF_ARM_CTL",
    "key_to_db",
    "key_to_db_batch",
    "db_to_key",
    "DeformSolver",
]
