"""DOE API 运行依赖预加载与就绪状态"""

from __future__ import annotations

import threading
from importlib.metadata import version
from typing import Any

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {"ready": False, "components": {}, "error": None}


def _load_components() -> dict[str, str]:
    # 所有HTTP运行依赖必须在Gunicorn开始处理并发请求前完成首次加载
    import joblib
    import numpy
    import pandas
    import pyDOE

    from mobo.automation.sampling import generate_samples
    from mobo.optimization.ga.run import _load_model
    from mobo.optimization.rl.parameterized import run_parameterized_rl
    from mobo.optimization.service import run_optimization
    from mobo.surrogate.evaluate import SurrogateModelEvaluator
    from mobo.surrogate.service import train_surrogate

    entries = {
        "sampling": (pyDOE.lhs, generate_samples),
        "surrogate_training": (train_surrogate,),
        "surrogate_evaluation": (SurrogateModelEvaluator,),
        "inference": (joblib.load, _load_model),
        "optimization_ga": (run_optimization,),
        "optimization_rl": (run_parameterized_rl,),
    }
    unavailable = [name for name, values in entries.items() if not all(map(callable, values))]
    if unavailable:
        raise RuntimeError(f"DOE运行依赖未提供可调用入口: {unavailable}")
    return {
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "pyDOE": version("pyDOE"),
        **{name: "ready" for name in entries},
    }


def load_runtime_dependencies() -> dict[str, Any]:
    """在服务启动阶段加载DOE依赖 加载失败时阻止worker启动"""
    with _LOCK:
        if _STATE["ready"]:
            return get_readiness()
        try:
            components = _load_components()
        except Exception as exc:
            _STATE.update(
                ready=False,
                components={},
                error=f"{type(exc).__name__}: {exc}",
            )
            raise RuntimeError("DOE运行依赖预加载失败") from exc
        _STATE.update(ready=True, components=components, error=None)
        return get_readiness()


def get_readiness() -> dict[str, Any]:
    """返回不包含服务器路径的依赖就绪状态"""
    return {
        "ready": bool(_STATE["ready"]),
        "components": dict(_STATE["components"]),
        "error": _STATE["error"],
    }


__all__ = ["get_readiness", "load_runtime_dependencies"]
