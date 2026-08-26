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

import json
import os
import re
import subprocess
import tempfile
import threading
import time
from queue import Queue
from typing import Any, Callable, List, Sequence

from mobo.common.logging import logger
from mobo.common.paths import LOGS_DIR

# DEFORM 可执行程序（需在 PATH 中，或设置对应环境变量）
DEF_PRE_64 = os.environ.get("MOBO_DEF_PRE_64", "DEF_PRE_64.exe")
DEF_ARM_CTL = os.environ.get("MOBO_DEF_ARM_CTL", "DEF_ARM_CTL.COM")

# DEFORM 操作日志路径（集中到 LOGS_DIR）
_OPERATION_LOG = os.path.join(str(LOGS_DIR), "deform_operation.log")

# DEF_PRE_64 的 DB→KEY 导出不支持同一进程内并发执行。
_DB_TO_KEY_LOCK = threading.Lock()


def _run_pre_with_commands(commands: str) -> str:
    """把命令串写入临时文件，用输入重定向驱动 DEF_PRE_64，并记录输出到日志。

    :param commands: 发送给 DEF_PRE_64 的完整命令串（含换行）
    """
    os.makedirs(str(LOGS_DIR), exist_ok=True)
    fd, cmd_file = tempfile.mkstemp(prefix="deform_pre_", suffix=".txt", text=True)
    output_lines: List[str] = []
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
                    output_lines.append(output)
    finally:
        if os.path.exists(cmd_file):
            os.remove(cmd_file)
    return "".join(output_lines)


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
    for key_path, db_path in zip(key_paths, db_paths, strict=True):
        key_to_db(key_path, db_path)


def db_to_key(db_path: str, key_path: str, step: str = "") -> None:
    """将 DB 文件的指定步导出为 KEY 文件。

    :param db_path: 输入 DB 文件路径
    :param key_path: 输出 KEY 文件路径
    :param step: 导出的步数（需准确，否则 DEFORM 报错）
    """
    commands = f"E\n2\n2\n{db_path}\n{step}\nE\nE\n8\n{key_path}\nE\nY\n"
    with _DB_TO_KEY_LOCK:
        _run_pre_with_commands(commands)


def query_db_steps(db_path: str) -> List[int]:
    """查询 DEFORM DB 实际保存的结果步号，不根据 NSTEP/STPINC 猜测。"""
    commands = f"E\n2\n2\n{db_path}\n\nE\nE\nY\n"
    output = _run_pre_with_commands(commands)
    match = re.search(r"Step Numbers:\s*(.*?)\s*Step Number\s*=", output, re.DOTALL)
    if match is None:
        raise RuntimeError(f"无法从 DEFORM 输出解析数据库保存步号: {db_path}")
    steps = sorted({int(value) for value in re.findall(r"-?\d+", match.group(1))})
    if not steps:
        raise RuntimeError(f"DEFORM 数据库没有可用结果步: {db_path}")
    return steps


def run_key_actions(key_path: str) -> None:
    """载入并执行包含 DBREAD/KFREAD/GENDB 的前处理动作 KEY。"""
    commands = f"E\n2\n1\n{key_path}\nE\nE\nY\n"
    _run_pre_with_commands(commands)


def solve_db_sync(db_path: str) -> None:
    """同步求解单个 DB，并以进程返回码和 LOG 的 NORMAL STOP 判定成功。"""
    work_dir = os.path.dirname(os.path.abspath(db_path))
    stem = os.path.splitext(os.path.basename(db_path))[0]
    for name in ("FOR003", "FOR003.LOCK"):
        residue = os.path.join(work_dir, name)
        if os.path.exists(residue):
            os.remove(residue)
            logger.info(f"已清理 DEFORM 异常退出残留文件: {residue}")
    process = subprocess.Popen(
        DEF_ARM_CTL,
        stdin=subprocess.PIPE,
        shell=False,
        cwd=work_dir,
        text=True,
    )
    if process.stdin is None:
        raise RuntimeError("无法打开 DEFORM 求解器标准输入")
    process.stdin.write(stem + "\nB\n")
    process.stdin.flush()
    process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"DEFORM 求解器退出码为 {return_code}")
    log_path = os.path.join(work_dir, stem + ".LOG")
    if not os.path.exists(log_path):
        raise RuntimeError(f"DEFORM 未生成求解日志: {log_path}")
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        log_text = f.read()
    normal_markers = (
        "NORMAL STOP",
        "Simulation Module Indicates End of Simulation",
    )
    if not any(marker in log_text for marker in normal_markers) or "ABNORMAL STOP" in log_text:
        raise RuntimeError(f"DEFORM 求解未正常结束，请检查: {log_path}")


class DeformSolver:
    """DEFORM 求解调度器，封装并发求解计数

    通过 ``DEF_ARM_CTL.COM`` 异步提交 DB 求解任务，并限制同时运行的进程数，
    取代原实现中裸露在模块级的全局计数器与锁
    """

    def __init__(self, max_parallel: int = 12,process_info_file:str = "",
                 on_completed: Callable[[str], None] | None = None) -> None:
        """
        :param max_parallel: 最大并行求解进程数
        :param process_info_file: 求解过程信息记录文件位置
        """
        self.max_parallel = max_parallel
        self.process_info_file = process_info_file
        self.on_completed = on_completed
        self._running = 0
        self._lock = threading.Lock()

    @property
    def running(self) -> int:
        """当前正在运行的求解进程数"""
        with self._lock:
            return self._running

    def _load_progress(self) -> dict:
        """读取求解进度文件；不存在或损坏则返回空进度。

        进度 schema::

            {
                "total": int,           # DB 总数
                "completed": int,       # 已完成数
                "created_at": str,      # 进度文件首次初始化时间
                "updated_at": str,      # 最近一次写入时间
                "db_files": [
                    {
                        "db_path": str,
                        "done": bool,
                        "started_at": str,   # 该 DB 开始求解时间
                        "finished_at": str,  # 该 DB 求解完成时间
                    },
                    ...
                ]
            }
        """
        empty = {"total": 0, "completed": 0, "created_at": "", "updated_at": "", "db_files": []}
        if not self.process_info_file or not os.path.exists(self.process_info_file):
            return empty
        try:
            with open(self.process_info_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return empty

    def _init_progress(self, db_paths: List[str]) -> dict:
        """按待求解 DB 列表初始化/合并进度。

        保留已有条目的 ``done`` 与时间戳（``started_at``/``finished_at``），
        并保留首次初始化的 ``created_at``（续跑时不刷新）。
        """
        prev = self._load_progress()
        prev_items = {it["db_path"]: it for it in prev.get("db_files", [])}
        now = time.strftime("%Y-%m-%d %H:%M:%S")

        db_files = []
        for p in db_paths:
            old = prev_items.get(p, {})
            db_files.append({
                "db_path": p,
                "done": bool(old.get("done")),
                "started_at": old.get("started_at", ""),
                "finished_at": old.get("finished_at", ""),
            })
        progress = {
            "total": len(db_files),
            "completed": sum(1 for it in db_files if it["done"]),
            "created_at": prev.get("created_at") or now,
            "updated_at": now,
            "db_files": db_files,
        }
        self._save_progress(progress)
        return progress

    def _save_progress(self, progress: dict) -> None:
        """原子写入求解进度（tempfile + os.replace）。"""
        if not self.process_info_file:
            return
        directory = os.path.dirname(self.process_info_file) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".process_", suffix=".json", dir=directory, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.process_info_file)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def _mark_started(self, db_path: str) -> None:
        """记录某个 DB 的开始求解时间（线程安全落盘）。"""
        if not self.process_info_file:
            return
        with self._lock:
            progress = self._load_progress()
            for item in progress.get("db_files", []):
                if item["db_path"] == db_path:
                    item["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            progress["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._save_progress(progress)

    def _mark_done(self, db_path: str) -> None:
        """把某个 DB 标记为完成、记录结束时间并刷新完成计数（线程安全落盘）。"""
        if not self.process_info_file:
            return
        with self._lock:
            progress = self._load_progress()
            for item in progress.get("db_files", []):
                if item["db_path"] == db_path:
                    item["done"] = True
                    item["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            progress["completed"] = sum(1 for it in progress["db_files"] if it["done"])
            progress["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._save_progress(progress)

    def pending_db_files(self, db_paths: List[str]) -> List[str]:
        """据进度文件返回仍需求解的 DB 文件（已完成的跳过）。

        无进度文件时视为全部待求解；用于续跑时恢复求解队列。
        """
        done = {
            item["db_path"] for item in self._load_progress().get("db_files", [])
            if item.get("done")
        }
        return [p for p in db_paths if p not in done]

    @staticmethod
    def _feed_stdin(process: Any, content: str) -> None:
        """向子进程标准输入写入一行（附带 DEFORM 需要的短暂延时）"""
        time.sleep(0.1)
        process.stdin.write(content + "\n")
        process.stdin.flush()

    def _solve_one(self, db_path: str, work_dir: str, db_key: str = "") -> None:
        """在独立线程中提交单个 DB 的求解

        :param db_path: 喂给 DEFORM 的求解目标（去扩展名）
        :param work_dir: 工作目录
        :param db_key: 进度文件中记录的完整 DB 路径（完成后据此标记 done）
        """
        self._mark_started(db_key or db_path)
        try:
            solve_db_sync(db_key or db_path)
            logger.info(f"当前任务计算完成！请查看 {db_path} 结果")
            self._mark_done(db_key or db_path)
            if self.on_completed is not None:
                try:
                    self.on_completed(db_key or db_path)
                except Exception as exc:
                    logger.error(f"完成后处理失败: {exc}")
        except Exception as exc:
            logger.error(f"求解进程出错: {exc}")
        finally:
            with self._lock:
                self._running -= 1

    def submit(self, db_path: str, work_dir: str, db_key: str = "") -> None:
        """异步提交一个求解任务

        :param db_key: 进度文件中记录的完整 DB 路径（用于完成标记）
        """
        with self._lock:
            self._running += 1
            logger.info(f"当前正在计算的任务有：{self._running} 个")
        thread = threading.Thread(
            target=self._solve_one, args=(db_path, work_dir, db_key), daemon=True
        )
        thread.start()

    def run_all(self, db_paths: List[str]) -> None:
        """按最大并行数调度求解全部 DB 文件，直至全部完成。

        据进度文件跳过已完成的 DB（支持任务被中断后仅凭进度文件续跑），
        并在每个 DB 求解完成时把进度落盘。

        :param db_paths: 待求解的 DB 文件路径列表
        """
        self._init_progress(db_paths)
        if self.on_completed is not None:
            completed = {
                item["db_path"] for item in self._load_progress().get("db_files", [])
                if item.get("done")
            }
            for db_path in db_paths:
                if db_path in completed:
                    try:
                        self.on_completed(db_path)
                    except Exception as exc:
                        logger.error(f"已完成 DB 后处理失败: {exc}")
        pending = self.pending_db_files(db_paths)

        task_queue: Queue = Queue()
        for i, db_path in enumerate(pending):
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

            self.submit(solve_target, work_dir, db_key=db_path)
            logger.info(f"开始计算第 {file_num} 个，结果 DB 将保存至 {work_dir}")
            # 防止过快提交导致计数尚未更新
            time.sleep(5)


__all__ = [
    "DEF_PRE_64",
    "DEF_ARM_CTL",
    "key_to_db",
    "key_to_db_batch",
    "db_to_key",
    "query_db_steps",
    "run_key_actions",
    "solve_db_sync",
    "DeformSolver",
]
