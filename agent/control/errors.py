"""控制面领域错误。"""


class ControlError(RuntimeError):
    """表示可安全映射到协议边界的控制面错误。"""


class ThreadNotFoundError(ControlError):
    pass


class ThreadBusyError(ControlError):
    pass


class TurnNotFoundError(ControlError):
    pass


class TurnStateTransitionError(ControlError):
    pass


class SlowConsumerError(ControlError):
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


class ControlExecutionError(ControlError):
    def __init__(self, error_type: str, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable
