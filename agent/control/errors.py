"""控制面领域错误。"""


class ControlError(RuntimeError):
    """表示可安全映射到协议边界的控制面错误。"""


class ThreadNotFoundError(ControlError):
    pass


class ThreadBusyError(ControlError):
    pass


class TurnNotFoundError(ControlError):
    pass


class RuntimeClosedError(ControlError):
    pass


class PluginManagementError(ControlError):
    pass


class ControlAdmissionError(ControlError):
    """表示 queued/running turn 超出控制面准入容量。"""

    error_type = "resource-exhausted"
    failure_type = "operation_rejected"
    code = "resource-exhausted"
    retryable = True
