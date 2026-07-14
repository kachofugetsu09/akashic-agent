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


class ControlExecutionError(ControlError):
    def __init__(self, error_type: str, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable
