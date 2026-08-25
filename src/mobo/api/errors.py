"""API 层可转换为 HTTP 响应的业务异常。"""


class ApiError(Exception):
    # 同时保存业务码和 HTTP 状态码 由 handler 统一转换为 JSON 响应
    def __init__(self, message: str, status: int = 400, code: int = 1) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code


class NotFoundError(ApiError):
    # 查询或删除不存在的 DOE 资源时固定返回 404
    def __init__(self, message: str) -> None:
        super().__init__(message, status=404, code=404)


class ConflictError(ApiError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status=409, code=409)


__all__ = ["ApiError", "NotFoundError", "ConflictError"]
