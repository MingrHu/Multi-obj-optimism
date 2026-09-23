"""Flask 应用工厂与 API 启动入口。"""

from __future__ import annotations

import os

from flask import Flask

from . import readiness


def create_app(config: dict | None = None) -> Flask:
    # 在Gunicorn开始处理多线程请求前完成DOE全部运行依赖的首次加载
    readiness.load_runtime_dependencies()
    # 使用应用工厂便于生产部署 测试环境也可以传入独立配置
    app = Flask(__name__)
    app.json.sort_keys = False
    if config:
        app.config.update(config)

    # 延迟导入路由模块 避免仅导入包时提前加载业务层及算法依赖
    from . import service
    from .handler import doe_api, register_error_handlers

    # 后台线程不跨进程重启恢复；生产启动时收敛上次进程遗留的瞬态优化状态。
    if not app.testing:
        service._recover_interrupted_optimizations()
    app.register_blueprint(doe_api)
    register_error_handlers(app)

    @app.get("/health")
    def health():
        dependency_status = readiness.get_readiness()
        status = 200 if dependency_status["ready"] else 503
        return {
            "code": 0 if dependency_status["ready"] else 1,
            "message": "ok" if dependency_status["ready"] else "service not ready",
            "data": {"service": "mobo-doe", "dependencies": dependency_status},
        }, status

    return app


def main() -> None:
    app = create_app()
    # 地址和端口通过环境变量覆盖 便于同一代码适配本机和服务器环境
    app.run(
        host=os.environ.get("MOBO_API_HOST", "0.0.0.0"),
        port=int(os.environ.get("MOBO_API_PORT", "5050")),
        debug=False,
    )


if __name__ == "__main__":
    main()


__all__ = ["create_app", "main"]
