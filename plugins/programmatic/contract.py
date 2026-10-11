"""程序来源公开参数和调用端口；GitHub Watch 等外置插件使用同一模型。"""
from __future__ import annotations

from typing import Protocol
from pydantic import BaseModel, ConfigDict, Field
from agent.plugin_composition import ServiceKey
from plugins.gateway.contract import RequestTransport


class SessionIdParams(BaseModel):
    """程序调用在自己的 RPC 输入边界校验参数。"""
    model_config = ConfigDict(extra="forbid", strict=True)
    session_id: str = Field(min_length=1, max_length=512)


class AdmitParams(SessionIdParams):
    persist_memory: bool = False


class SendParams(SessionIdParams):
    message_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=1_048_576)
    model_id: str | None = Field(default=None, min_length=1, max_length=512)
    reasoning_effort: str | None = Field(default=None, min_length=1, max_length=64)


class PauseParams(SessionIdParams):
    message_id: str = Field(min_length=1, max_length=256)


class ResumeParams(PauseParams):
    input_id: str = Field(min_length=1, max_length=256)


class ResultParams(SessionIdParams):
    input_id: str = Field(min_length=1, max_length=256)


class ProgrammaticService(Protocol):
    async def call(
        self, method: str, params: BaseModel, transport: RequestTransport | None = None,
    ) -> dict[str, object]: ...


PROGRAMMATIC = ServiceKey[ProgrammaticService]("programmatic.v1")
