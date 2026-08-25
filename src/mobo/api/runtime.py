"""进程内后台任务与协作式取消控制。"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from .errors import ConflictError


@dataclass
class RunningTask:
    # 每个后台任务同时保存执行线程和供业务循环检查的取消事件
    thread: threading.Thread
    cancel: threading.Event


class TaskRegistry:
    def __init__(self) -> None:
        # DOE 标识和操作类型共同组成键 因此同一 DOE 可区分训练与优化
        self._tasks: dict[tuple[str, str], RunningTask] = {}
        self._lock = threading.Lock()

    def start(self, doe_id: str, operation: str, target: Callable[[threading.Event], None]) -> None:
        key = (doe_id, operation)
        with self._lock:
            # 拒绝重复提交 防止两个线程同时覆盖同一份任务状态和模型产物
            current = self._tasks.get(key)
            if current and current.thread.is_alive():
                raise ConflictError(f"{operation} 已在运行")
            cancel = threading.Event()
            thread = threading.Thread(
                target=self._run, args=(key, cancel, target), daemon=True,
                name=f"doe-{operation}-{doe_id}",
            )
            self._tasks[key] = RunningTask(thread, cancel)
            thread.start()

    def _run(self, key, cancel, target) -> None:
        try:
            target(cancel)
        finally:
            # 无论正常完成 中止或异常都移除运行时记录 避免后续请求误判仍在运行
            with self._lock:
                self._tasks.pop(key, None)

    def stop(self, doe_id: str, operation: str) -> bool:
        with self._lock:
            task = self._tasks.get((doe_id, operation))
            if not task or not task.thread.is_alive():
                return False
            # 只设置协作式取消信号 业务层会在安全检查点结束任务并更新落盘状态
            task.cancel.set()
            return True

    def running(self, doe_id: str, operation: str) -> bool:
        with self._lock:
            task = self._tasks.get((doe_id, operation))
            return bool(task and task.thread.is_alive())


registry = TaskRegistry()

__all__ = ["TaskRegistry", "registry"]
