class ModelRuntimeError(Exception):
    """模型运行时可归类错误的基类。"""


class AuthenticationError(ModelRuntimeError):
    """认证缺失、过期或刷新失败。"""


class RateLimitError(ModelRuntimeError):
    """服务端限流。"""


class QuotaError(ModelRuntimeError):
    """账号额度耗尽。"""


class TransportError(ModelRuntimeError):
    """请求或响应协议错误。"""
