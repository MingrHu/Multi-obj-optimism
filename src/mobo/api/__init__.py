"""DOE HTTP API package."""

from .app import create_app

# 包级入口仅公开应用工厂 避免调用方依赖内部 handler 和 service 实现
__all__ = ["create_app"]
