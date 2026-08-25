"""Flask 应用工厂与 API 启动入口。"""

from __future__ import annotations

import os

from flask import Flask


def create_app(config: dict | None = None) -> Flask:
    # 使用应用工厂便于生产部署 测试环境也可以传入独立配置
    app = Flask(__name__)
    if config:
        app.config.update(config)

    # 延迟导入路由模块 避免仅导入包时提前加载业务层及算法依赖
    from .handler import doe_api, register_error_handlers

    app.register_blueprint(doe_api)
    register_error_handlers(app)

    @app.get("/health")
    def health():
        return {"code": 0, "message": "ok", "data": {"service": "mobo-doe"}}

    return app


def main() -> None:
    app = create_app()
    # 地址和端口通过环境变量覆盖 便于同一代码适配本机和服务器环境
    app.run(
        host=os.environ.get("MOBO_API_HOST", "0.0.0.0"),
        port=int(os.environ.get("MOBO_API_PORT", "5000")),
        debug=False,
    )


if __name__ == "__main__":
    main()


__all__ = ["create_app", "main"]
