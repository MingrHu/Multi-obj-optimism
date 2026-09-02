"""批处理过程中增量生成数据集的通用检查点。"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from typing import Any, Sequence


class IncrementalDataset:
    """以样本序号为主键，原子保存数据行并生成稳定的数据集文件。"""

    def __init__(self, state_file: str, output_file: str) -> None:
        self.state_file = os.path.abspath(state_file)
        self.output_file = os.path.abspath(output_file)
        self._lock = threading.Lock()
        self._ensure_state()

    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _atomic_text(path: str, content: str) -> None:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".incremental_", dir=directory, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _load(self) -> dict[str, Any]:
        with open(self.state_file, "r", encoding="utf-8") as stream:
            return json.load(stream)

    def _save(self, state: dict[str, Any]) -> None:
        self._atomic_text(
            self.state_file,
            json.dumps(state, ensure_ascii=False, indent=2),
        )

    def _ensure_state(self) -> None:
        if not os.path.exists(self.state_file):
            now = self._now()
            self._save({
                "version": 1,
                "output_file": self.output_file,
                "created_at": now,
                "updated_at": now,
                "samples": {},
            })
        if not os.path.exists(self.output_file):
            self._rebuild_output()

    def _rebuild_output(self, state: dict[str, Any] | None = None) -> None:
        """根据状态中已完成的样本，按样本序号原子重建结果文件。"""
        current = state if state is not None else self._load()
        completed = (
            current["samples"][key]
            for key in sorted(current["samples"], key=int)
            if current["samples"][key].get("status") == "completed"
        )
        content = "".join("\t".join(item["row"]) + "\n" for item in completed)
        self._atomic_text(self.output_file, content)

    def is_completed(self, sample_index: int) -> bool:
        with self._lock:
            item = self._load()["samples"].get(str(sample_index), {})
            return item.get("status") == "completed"

    def mark_started(self, sample_index: int) -> None:
        with self._lock:
            state = self._load()
            previous = state["samples"].get(str(sample_index), {})
            state["samples"][str(sample_index)] = {
                **previous,
                "status": "extracting",
                "started_at": self._now(),
                "attempts": int(previous.get("attempts", 0)) + 1,
                "error": "",
            }
            state["updated_at"] = self._now()
            self._save(state)

    def mark_failed(
        self,
        sample_index: int,
        error: str,
        *,
        error_type: str = "",
        traceback_text: str = "",
    ) -> None:
        """记录提取失败原因和调用栈，保留已有行以便后续重试诊断。"""
        with self._lock:
            state = self._load()
            previous = state["samples"].get(str(sample_index), {})
            state["samples"][str(sample_index)] = {
                **previous,
                "status": "failed",
                "error": error,
                "error_type": error_type,
                "traceback": traceback_text,
                "failed_at": self._now(),
            }
            state["updated_at"] = self._now()
            self._save(state)

    def commit(self, sample_index: int, row: Sequence[Any]) -> None:
        """提交一行并按样本序号原子重建输出文件，重复提交会覆盖而非追加。"""
        with self._lock:
            state = self._load()
            state["samples"][str(sample_index)] = {
                "status": "completed",
                "row": [str(value) for value in row],
                "finished_at": self._now(),
                "error": "",
            }
            state["updated_at"] = self._now()
            self._save(state)
            self._rebuild_output(state)


__all__ = ["IncrementalDataset"]
