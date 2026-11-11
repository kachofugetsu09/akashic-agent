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


class UpdateNotFoundError(ControlError):
    """表示查询的插件更新 ID 在当前 workspace 记录中不存在。"""

    def __init__(self, update_id: str) -> None:
        super().__init__(f"插件更新记录不存在: {update_id}")
        self.update_id = update_id


class ControlAdmissionError(ControlError):
    """表示 queued/running turn 超出控制面准入容量。"""

    error_type = "resource-exhausted"
    failure_type = "operation_rejected"
    code = "resource-exhausted"
    retryable = True
