"""DNN 独立进程调度与协作式中止"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from threading import Event

from mobo.common.paths import model_family_dir
from .common import _MODEL_OUTPUT_DIR, model_output_dir

_CANCEL: ContextVar[Event | None] = ContextVar("dnn_cancel", default=None)


@contextmanager
def training_control(cancel: Event):
    # 后台训练线程将取消事件传给当前调用 不影响其他 DOE 的训练
    token = _CANCEL.set(cancel)
    try:
        yield
    finally:
        _CANCEL.reset(token)


def worker_count(target_count: int) -> int:
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    default = min(8, max(1, (available or 1) // 2))
    limit = int(os.environ.get("MOBO_DNN_MAX_WORKERS", default))
    if limit < 1:
        raise ValueError("MOBO_DNN_MAX_WORKERS 必须为正整数")
    return min(target_count, default, limit)


def _worker_environment() -> dict[str, str]:
    # 在子进程导入 TensorFlow 和数值库前限制线程 避免嵌套并行争抢 CPU
    environment = os.environ.copy()
    for name in (
        "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    ):
        environment[name] = "1"
    return environment


def _terminate(processes) -> None:
    for process, _ in processes:
        if process.poll() is None:
            process.terminate()
    for process, _ in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _run_workers(spec_file: Path, targets: int, limit: int, cancel: Event | None) -> None:
    active = []
    next_target = 0
    environment = _worker_environment()
    try:
        while next_target < targets or active:
            if cancel is not None and cancel.is_set():
                raise RuntimeError("DNN 训练已中止")
            while next_target < targets and len(active) < limit:
                log_path = spec_file.parent / f"target_{next_target}.log"
                with log_path.open("wb") as log:
                    process = subprocess.Popen(
                        [sys.executable, "-m", "mobo.surrogate.dnn_process",
                         str(spec_file), str(next_target)],
                        env=environment, stdout=log, stderr=subprocess.STDOUT,
                    )
                active.append((process, log_path))
                next_target += 1
            for process, log_path in active[:]:
                code = process.poll()
                if code is None:
                    continue
                if code != 0:
                    detail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                    raise RuntimeError(f"DNN 输出训练失败 返回码 {code}\n{detail}")
                process.wait()
                active.remove((process, log_path))
            if active:
                if cancel is not None:
                    cancel.wait(0.1)
                else:
                    time.sleep(0.1)
    finally:
        _terminate(active)


def train_parallel(file: str, vars_out: list[str], n_var: int, params: dict) -> None:
    targets = len(vars_out) - n_var
    if targets < 1 or n_var < 1:
        raise ValueError("DNN 至少需要一个输入和一个输出")
    output_dir = Path(_MODEL_OUTPUT_DIR.get() or model_family_dir("DNN")).resolve()
    spec = {
        "file": str(Path(file).resolve()), "vars_out": vars_out,
        "n_var": n_var, "params": params, "output_dir": str(output_dir),
    }
    # 任务描述与子进程日志仅存在于临时目录 模型仍保存到调用方指定目录
    with tempfile.TemporaryDirectory(prefix="mobo_dnn_") as directory:
        spec_file = Path(directory) / "training.json"
        spec_file.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
        _run_workers(spec_file, targets, worker_count(targets), _CANCEL.get())


def _worker_main(spec_file: str, target_index: int) -> None:
    from .dnn import _train_outputs

    spec = json.loads(Path(spec_file).read_text(encoding="utf-8"))
    with model_output_dir(spec["output_dir"]):
        _train_outputs(
            spec["file"], spec["vars_out"], spec["n_var"],
            spec["params"], [target_index],
        )


if __name__ == "__main__":
    _worker_main(sys.argv[1], int(sys.argv[2]))
