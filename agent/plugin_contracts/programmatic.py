"""程序化调用能力的公开结构合同。

`programmatic.v1` 是「程序化来源」的公开名字：Core 的控制服务端需要按名字
调用它，插件提供实现。RPC 参数模型与 key 因此由合同层拥有。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import Field

from agent.plugin_composition.model import ServiceKey
from agent.plugin_contracts.control_method import RequestTransport
from agent.plugin_contracts.control_models import SessionIdParams, StrictModel


class AdmitParams(SessionIdParams):
    """显式接纳一次程序化输入。"""

    persist_memory: bool = False


class SendParams(SessionIdParams):
    """发送一条程序化消息。"""

    message_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=1_048_576)


class PauseParams(SessionIdParams):
    """暂停一次程序化输入。"""

    message_id: str = Field(min_length=1, max_length=256)


class ResumeParams(PauseParams):
    """恢复一次程序化输入。"""

    input_id: str = Field(min_length=1, max_length=256)


class ResultParams(SessionIdParams):
    """读取一次程序化输入的结果。"""

    input_id: str = Field(min_length=1, max_length=256)


PARAMS: dict[str, type[StrictModel]] = {
    "programmatic/session/admit": AdmitParams,
    "programmatic/message/send": SendParams,
    "programmatic/message/pause": PauseParams,
    "programmatic/message/resume": ResumeParams,
    "programmatic/message/result": ResultParams,
}


@runtime_checkable
class ProgrammaticPort(Protocol):
    """程序化来源对控制面可见的方法子集。"""

    async def call(
        self,
        method: str,
        params: StrictModel,
        transport: RequestTransport | None = None,
    ) -> dict[str, object]:
        """按 RPC 方法名执行一次程序化调用。"""
        ...


PROGRAMMATIC = ServiceKey[ProgrammaticPort]("programmatic.v1")
