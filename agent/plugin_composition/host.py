"""唯一宿主只读事实；Channel 不从宿主取得控制或清理能力。"""
from __future__ import annotations

from dataclasses import dataclass

from agent.plugin_composition.model import ServiceKey


@dataclass(frozen=True, slots=True)
class HostInfo:
    """Expose the process boot identity and validation-only marker."""

    boot_id: str
    validation: bool

    def __post_init__(self) -> None:
        if not isinstance(self.boot_id, str) or not self.boot_id or self.boot_id.strip() != self.boot_id:
            raise ValueError("HostInfo.boot_id 必须是非空字符串")
        if not isinstance(self.validation, bool):
            raise TypeError("HostInfo.validation 必须是 bool")


HOST_INFO = ServiceKey[HostInfo]("core.host_info")

