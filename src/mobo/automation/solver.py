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
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from queue import Queue
from typing import Any, BinaryIO, Callable, Iterator, List, Sequence

from mobo.common.logging import logger
from mobo.common.paths import LOGS_DIR

from .execution_control import (BatchStopped, batch_scope, check_stopped, current_control, process_alive)

# DEFORM 可执行程序（需在 PATH 中，或设置对应环境变量）
DEF_PRE_64 = os.environ.get("MOBO_DEF_PRE_64", "DEF_PRE_64.exe")
DEF_ARM_CTL = os.environ.get("MOBO_DEF_ARM_CTL", "DEF_ARM_CTL.COM")

# DEFORM 操作日志路径（集中到 LOGS_DIR）
_OPERATION_LOG = os.path.join(str(LOGS_DIR), "deform_operation.log")

# DEF_PRE_64 的全部前处理操作在同一进程内串行执行。
_PRE_LOCK = threading.Lock()
_PRE_PROCESS_LOCK = os.path.join(tempfile.gettempdir(), "mobo_deform_pre.lock")


@contextmanager
def _interprocess_lock(path: str, *, blocking: bool = True) -> Iterator[None]:
    """使用操作系统文件锁协调多个 Python 进程。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    stream: BinaryIO = open(path, "a+b")
    if os.path.getsize(path) == 0:
        stream.write(b"\0")
        stream.flush()
    stream.seek(0)
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt

            mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
            try:
                msvcrt.locking(stream.fileno(), mode, 1)
            except OSError as exc:
                if not blocking:
                    raise BlockingIOError(f"文件锁已被占用: {path}") from exc
                raise
        else:
            import fcntl

            flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            try:
                fcntl.flock(stream.fileno(), flags)
            except OSError as exc:
                if not blocking:
                    raise BlockingIOError(f"文件锁已被占用: {path}") from exc
                raise
        acquired = True
        yield
    finally:
        try:
            if acquired:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def _pre_command_summary(commands: str) -> str:
    """从菜单命令中提取便于关联日志的操作、输入文件和输出文件。"""
    lines = commands.splitlines()
    if len(lines) >= 4 and lines[:3] == ["E", "2", "2"]:
        db_path = lines[3]
        step = lines[4] if len(lines) > 4 and lines[4] else "latest"
        if "8" in lines[5:]:
            export_index = lines.index("8", 5)
            key_path = lines[export_index + 1] if export_index + 1 < len(lines) else "?"
            return f"db_to_key db={db_path} step={step} key={key_path}"
        return f"query_db_steps db={db_path}"
    if len(lines) >= 4 and lines[:3] == ["E", "2", "1"]:
        key_path = lines[3]
        if "7" in lines[4:]:
            generate_index = lines.index("7", 4)
            db_path = lines[generate_index + 2] if generate_index + 2 < len(lines) else "?"
            return f"key_to_db key={key_path} db={db_path}"
        return f"run_key_actions key={key_path}"
    return "unknown_pre_operation"


def _run_pre_with_commands(commands: str) -> str:
    """把命令串写入临时文件，用输入重定向驱动 DEF_PRE_64，并记录输出到日志。

    :param commands: 发送给 DEF_PRE_64 的完整命令串（含换行）
    """
    check_stopped()
    with _PRE_LOCK:
        with _interprocess_lock(_PRE_PROCESS_LOCK):
            for attempt in range(3):
                try:
                    check_stopped()
                    return _run_pre_with_commands_unlocked(commands)
                except RuntimeError as exc:
                    if "Access Denied" not in str(exc) or attempt == 2:
                        raise
                    delay = 2 * (attempt + 1)
                    logger.warning(
                        f"DEFORM 前处理器遇到文件占用，{delay} 秒后进行第 "
                        f"{attempt + 2}/3 次尝试"
                    )
                    time.sleep(delay)
    raise RuntimeError("DEFORM 前处理器重试流程异常结束")


def _run_pre_with_commands_unlocked(commands: str) -> str:
    """执行一个已经取得全局前处理锁的 DEF_PRE_64 命令。"""
    os.makedirs(str(LOGS_DIR), exist_ok=True)
    fd, cmd_file = tempfile.mkstemp(prefix="deform_pre_", suffix=".txt", text=True)
    output_lines: List[str] = []
    summary = _pre_command_summary(commands)
    call_id = f"{time.time_ns()}-{threading.get_ident()}"
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
        control = current_control()
        if control is not None:
            control.register(process)
        with open(_OPERATION_LOG, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n[MOBO PRE START] id={call_id} {summary}\n")
            while True:
                output = process.stdout.readline()  # type: ignore
                if not output and process.poll() is not None:
                    break
                if output:
                    log_file.write(output)
                    output_lines.append(output)
            return_code = process.poll()
            log_file.write(
                f"[MOBO PRE END] id={call_id} return_code={return_code} "
                f"output_chars={sum(len(line) for line in output_lines)}\n"
            )
        check_stopped()
        if return_code not in (0, None):
            output_tail = "".join(output_lines)[-2000:].strip()
            message = (
                f"DEFORM 前处理器异常退出: id={call_id}, return_code={return_code}, "
                f"{summary}, output_tail={output_tail!r}"
            )
            logger.error(message)
            raise RuntimeError(message)
    except Exception as exc:
        logger.error(f"DEFORM 前处理器调用异常: id={call_id}, {summary}, error={exc}")
        raise
    finally:
        if "control" in locals() and control is not None:
            control.unregister(process)
        if os.path.exists(cmd_file):
            os.remove(cmd_file)
    return "".join(output_lines)


def key_to_db(key_path: str, db_path: str) -> None:
    """将 KEY 文件转换为 DB 文件。

    :param key_path: 输入 KEY 文件路径
    :param db_path: 输出 DB 文件路径
    """
    commands = f"E\n2\n1\n{key_path}\nE\nE\n7\n2\n{db_path}\nY\nE\nY\n"
    try:
        _run_pre_with_commands(commands)
    except BatchStopped:
        Path(db_path).with_name(".mobo_interrupted").touch()
        raise


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


def recover_interrupted_db(db_path: str) -> bool:
    """在副本中恢复最新已保存步；失败时归档原件，由调用方从 KEY 重建。"""
    directory = Path(db_path).resolve().parent
    raw = directory / "FOR003"
    marker = directory / ".mobo_interrupted"
    status = directory / "DEFORM_RUNNING_JOB_STATUS.TXT"
    if status.exists():
        match = re.search(r"processID\s*=\s*(\d+)", status.read_text(errors="ignore"))
        if match and process_alive(int(match.group(1))):
            raise RuntimeError(f"DEFORM 仍在使用数据库，不能恢复: {db_path}")
    if not raw.exists() and not marker.exists() and not (directory / "FOR003.LOCK").exists():
        return Path(db_path).is_file()
    check_stopped()
    source = raw if raw.exists() else Path(db_path)
    recovered = False
    with tempfile.TemporaryDirectory(prefix="deform_recovery_", dir=directory) as scratch:
        candidate = Path(scratch) / "restart.DB"
        try:
            shutil.copy2(source, candidate)
            steps = query_db_steps(str(candidate))
            action = Path(scratch) / "restart.KEY"
            action.write_text(f"DBREAD 0\n{candidate}\nGENDB 1 1 0\n{candidate}\n", encoding="utf-8")
            run_key_actions(str(action))
            restored = query_db_steps(str(candidate))
            if not any(step < 0 and abs(step) >= max(map(abs, steps)) for step in restored):
                raise RuntimeError("最新保存步未生成可续算的负步号")
            recovered = True
        except BatchStopped:
            raise
        except Exception as exc:
            logger.warning(f"中断数据库恢复失败，将保留原件并从 KEY 重建: {exc}")
        check_stopped()
        archive = directory / "recovery" / str(time.time_ns())
        archive.mkdir(parents=True)
        for old in (raw, directory / "FOR003.LOCK", Path(db_path), marker, status):
            if old.exists():
                shutil.move(str(old), str(archive / old.name))
        if recovered:
            shutil.copy2(candidate, db_path)
            logger.info(f"已恢复最新保存步: {db_path}; 原件归档至 {archive}")
    return recovered


def solve_db_sync(db_path: str, *, required_stroke: tuple[str, int, float] | None = None) -> None:
    """同一数据库仅允许一个求解器，取消仅作用于登记的进程树。"""
    with _interprocess_lock(db_path + ".run.lock", blocking=False):
        _solve_db_sync_unlocked(db_path, required_stroke=required_stroke)


def _solve_db_sync_unlocked(db_path: str, *, required_stroke=None) -> None:
    """同步求解单个 DB，并以进程返回码和 LOG 的 NORMAL STOP 判定成功。"""
    work_dir = os.path.dirname(os.path.abspath(db_path))
    _configure_solver_bits(work_dir)
    stem = os.path.splitext(os.path.basename(db_path))[0]
    check_stopped()
    if any(os.path.exists(os.path.join(work_dir, name)) for name in ("FOR003", "FOR003.LOCK", ".mobo_interrupted", "DEFORM_RUNNING_JOB_STATUS.TXT")):
        if not recover_interrupted_db(db_path):
            raise RuntimeError("中断数据库无法恢复，请从原始 KEY 重新生成 DB")
    process = subprocess.Popen(
        DEF_ARM_CTL,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        shell=False,
        cwd=work_dir,
        text=True,
    )
    control = current_control()
    try:
        if control is not None:
            control.register(process)
        if process.stdin is None:
            raise RuntimeError("无法打开 DEFORM 求解器标准输入")
        process.stdin.write(stem + "\nB\n")
        process.stdin.flush()
        process.stdin.close()
        return_code = process.wait()
    finally:
        if control is not None:
            control.unregister(process)
            if control.event.is_set():
                Path(os.path.join(work_dir, ".mobo_interrupted")).touch()
    check_stopped()
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
    message_path = os.path.join(work_dir, stem + ".MSG")
    if not os.path.isfile(message_path) or os.path.getsize(message_path) == 0:
        raise RuntimeError(f"DEFORM 未产生求解消息，不能判定模拟完成: {message_path}")
    if required_stroke is not None:
        validate_db_stroke(db_path, required_stroke)


def validate_db_stroke(db_path: str, required_stroke: tuple[str, int, float]) -> None:
    """重新导出DB最新结果集，校验实际行程；不复用可能过期的终态KEY。"""
    object_id, axis, target = required_stroke
    with tempfile.TemporaryDirectory(prefix="mobo_stroke_") as directory:
        key_path = os.path.join(directory, "terminal.KEY")
        db_to_key(db_path, key_path)
        values = []
        with open(key_path, encoding="utf-8") as stream:
            for line in stream:
                tokens = line.split()
                if len(tokens) == 5 and tokens[:2] == ["STROKE", object_id]:
                    values.append(float(tokens[axis + 2]))
        if (len(values) != 1 or not math.isfinite(values[0])
                or not math.isclose(abs(values[0]), target, abs_tol=0.01, rel_tol=0)):
            raise RuntimeError(
                f"工况未完成目标行程：对象{object_id}实际STROKE={values}，"
                f"要求{target} mm（容差0.01 mm），不能作为完整终态结果"
            )


def _configure_solver_bits(work_dir: str) -> None:
    """安装目录存在 64 位引擎时，在任务目录使用 DEFORM 的原生标记选用它。"""
    controller = shutil.which(DEF_ARM_CTL)
    if os.name != "nt" or controller is None:
        return
    engine = os.path.join(os.path.dirname(controller), "DEF_SIM_64.EXE")
    if os.path.isfile(engine):
        marker = os.path.join(work_dir, "64bit.DAT")
        with open(marker, "a", encoding="ascii"):
            pass


class DeformSolver:
    """DEFORM 求解调度器，封装并发求解计数

    通过 ``DEF_ARM_CTL.COM`` 异步提交 DB 求解任务，并限制同时运行的进程数，
    取代原实现中裸露在模块级的全局计数器与锁
    """

    def __init__(self, max_parallel: int = 12,process_info_file:str = "",
                 on_completed: Callable[[str], None] | None = None,
                 required_stroke: tuple[str, int, float] | None = None) -> None:
        """
        :param max_parallel: 最大并行求解进程数
        :param process_info_file: 求解过程信息记录文件位置
        """
        self.max_parallel = max_parallel
        self.process_info_file = process_info_file
        self.on_completed = on_completed
        self.required_stroke = required_stroke
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
                    item["stopped"] = False
                    item["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            progress["status"] = "running"
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
            progress["status"] = ("completed" if progress["completed"] == len(progress["db_files"]) else "running")
            progress["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._save_progress(progress)

    def _mark_stopped(self) -> None:
        """工作线程全部退出后，记录未完成工况的停止状态。"""
        if not self.process_info_file:
            return
        with self._lock:
            progress = self._load_progress()
            for item in progress.get("db_files", []):
                if not item.get("done"):
                    item["stopped"] = True
            progress["status"] = "stopped"
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
        try:
            check_stopped()
            self._mark_started(db_key or db_path)
            if self.required_stroke is None:
                solve_db_sync(db_key or db_path)
            else:
                solve_db_sync(db_key or db_path, required_stroke=self.required_stroke)
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
        control = current_control()
        def worker():
            with batch_scope(control):
                self._solve_one(db_path, work_dir, db_key)
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def run_all(self, db_paths: List[str]) -> None:
        """按最大并行数调度求解全部 DB 文件，直至全部完成。

        据进度文件跳过已完成的 DB（支持任务被中断后仅凭进度文件续跑），
        并在每个 DB 求解完成时把进度落盘。

        :param db_paths: 待求解的 DB 文件路径列表
        """
        progress = self._init_progress(db_paths)
        if self.required_stroke is not None:
            for item in progress["db_files"]:
                if not item.get("done"):
                    continue
                try:
                    validate_db_stroke(item["db_path"], self.required_stroke)
                except Exception as exc:
                    item["done"] = False
                    item["validation_error"] = str(exc)
                    progress["completed"] = sum(bool(row.get("done")) for row in progress["db_files"])
                    self._save_progress(progress)
                    raise
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

        control = current_control()
        while True:
            if control is not None and control.event.is_set():
                if self.running:
                    time.sleep(0.1)
                    continue
                self._mark_stopped()
                control.check()
            if self.running >= self.max_parallel:
                control.event.wait(0.2) if control is not None else time.sleep(3)
                continue

            if task_queue.qsize() == 0 and self.running == 0:
                break
            if task_queue.qsize() == 0:
                control.event.wait(0.2) if control is not None else time.sleep(60)
                continue

            file_num, db_path = task_queue.get()
            work_dir = os.path.dirname(db_path)
            stem = os.path.splitext(os.path.basename(db_path))[0]
            solve_target = os.path.join(work_dir, stem)

            self.submit(solve_target, work_dir, db_key=db_path)
            logger.info(f"开始计算第 {file_num} 个，结果 DB 将保存至 {work_dir}")
            # 防止过快提交导致计数尚未更新
            control.event.wait(5) if control is not None else time.sleep(5)


__all__ = [
    "DEF_PRE_64",
    "DEF_ARM_CTL",
    "key_to_db",
    "key_to_db_batch",
    "db_to_key",
    "query_db_steps",
    "run_key_actions",
    "solve_db_sync",
    "validate_db_stroke",
    "DeformSolver",
]
