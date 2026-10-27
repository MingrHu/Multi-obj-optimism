"""本工作台批处理的合作式取消与已登记子进程树管理。"""

from __future__ import annotations

import contextvars
import os
import subprocess
import threading
import weakref
from contextlib import contextmanager

from mobo.common.logging import logger


class BatchStopped(RuntimeError):
    """用户主动停止；与求解失败区分。"""


_current = contextvars.ContextVar("mobo_batch_control", default=None)
_controls: weakref.WeakSet = weakref.WeakSet()
_controls_lock = threading.Lock()


def current_control():
    return _current.get()


@contextmanager
def batch_scope(control):
    token = _current.set(control)
    try:
        yield
    finally:
        _current.reset(token)


def check_stopped() -> None:
    control = current_control()
    if control is not None:
        control.check()


def process_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x100000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _terminate_tree(process) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        result = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode and process.poll() is None:
            raise RuntimeError(f"停止DEFORM进程树失败: PID={process.pid}")
    else:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
    process.wait(timeout=10)


class BatchControl:
    """取消标记先于杀进程，确保队列不再补交任务。"""

    def __init__(self):
        self.event = threading.Event()
        self._lock = threading.Lock()
        self._processes = {}
        self.errors: list[str] = []
        with _controls_lock:
            _controls.add(self)

    def check(self):
        if self.event.is_set():
            raise BatchStopped("批处理已停止，未完成工况保留待续跑")

    def register(self, process):
        with self._lock:
            self._processes[process.pid] = process
        if self.event.is_set():
            self._terminate(process)
            if process.poll() is None:
                process.wait()
            self.check()

    def unregister(self, process):
        with self._lock:
            self._processes.pop(process.pid, None)

    def _terminate(self, process):
        try:
            _terminate_tree(process)
        except Exception as exc:
            self.errors.append(str(exc))
            logger.error(str(exc))

    def request_stop(self):
        with self._lock:
            if self.event.is_set():
                return
            self.event.set()
            processes = list(self._processes.values())
        threading.Thread(target=lambda: [self._terminate(process) for process in processes],
                         daemon=True).start()

    def wait(self, seconds):
        if self.event.wait(seconds):
            self.check()


def stop_all_batches():
    """只停止本Python工作台登记的批处理，不按程序名称误杀外部任务。"""
    with _controls_lock:
        controls = list(_controls)
    for control in controls:
        control.request_stop()
    return len(controls)
