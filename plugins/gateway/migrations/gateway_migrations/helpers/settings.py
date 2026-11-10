"""Gateway 在自身边界校验控制监听设置。"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GatewayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool = True
    listen: str = ""
    max_connections: int = Field(default=32, gt=0)
    ingress_queue_size: int = Field(default=128, gt=0)
    outbound_queue_size: int = Field(default=512, gt=0)
    max_message_bytes: int = Field(default=2 * 1024 * 1024, gt=0)

    @classmethod
    def from_legacy(cls, raw: dict[str, object]) -> GatewayConfig:
        """一次性保留旧 Core 的数值转换规则；新固定输入使用严格 schema。"""
        values = dict(raw)
        values["listen"] = str(values.get("listen", "")).strip()
        for name in ("max_connections", "ingress_queue_size", "outbound_queue_size", "max_message_bytes"):
            if name in values:
                value = values[name]
                if not isinstance(value, (int, float, str)):
                    raise ValueError(f"app_server.{name} 必须是整数")
                values[name] = int(value)
        return cls.model_validate(values)
