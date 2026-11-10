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
    """程序 Session 的创建参数；scope 决定记忆策略落在哪个范围。

    scope 是宽键中已声明的维度（如 ``{"project": "github-review"}``）。同一维度值的
    记忆策略由 Akasha 的 scope 策略决定，所以程序来源可以把自己放进一个既不学习、
    也不召回的范围内，而不必改动 learning 轴——learning 只控制「写」，控制不了「读」。
    """

    persist_memory: bool = False
    scope: dict[str, str] | None = Field(default=None, max_length=16)


def parse_scope_pairs(pairs: list[str]) -> dict[str, str]:
    """把重复的 `DIM=VALUE` 收成映射；维度名与取值仍交给 SessionAttributes 校验。"""
    scope: dict[str, str] = {}
    for pair in pairs:
        dimension, separator, value = pair.partition("=")
        if not separator or not dimension or not value:
            raise ValueError(f"--scope 需要 DIM=VALUE，收到 {pair!r}")
        if dimension in scope:
            raise ValueError(f"--scope 维度重复: {dimension}")
        scope[dimension] = value
    return scope


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


# Fleet Proactive Feedback 与 GitHub Watch 提交程序输入；调用与传输适配仍由本插件拥有。
PROGRAMMATIC = ServiceKey[ProgrammaticService]("programmatic.v1")
