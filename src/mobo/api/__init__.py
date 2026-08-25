"""面向外部调用方的稳定 JSON/字典接口。"""

__all__ = ["ApiValidationError", "train_surrogate", "run_optimization", "query_task"]


def __getattr__(name):
    """延迟导入，允许服务启动前通过 MOBO_DATA_DIR 配置数据目录。"""
    if name == "ApiValidationError":
        from .validation import ApiValidationError

        return ApiValidationError
    if name in {"train_surrogate", "run_optimization", "query_task"}:
        from .facade import query_task, run_optimization, train_surrogate

        return {
            "train_surrogate": train_surrogate,
            "run_optimization": run_optimization,
            "query_task": query_task,
        }[name]
    raise AttributeError(name)
