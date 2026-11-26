from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol
from asyncio import Future
from plugins.ledger.contract import CallRef
from plugins.ui.contract import (
    NAMES as NAMES,
    RequestTransport as RequestTransport,
    RpcMethod as RpcMethod,
    RpcNames as RpcNames,
    TransportCall as TransportCall,
)

from agent.plugin_composition import ServiceKey


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


class FrameResolver(Protocol):
    """读取一个 Input 对应的最终 Output 身份。"""

    def __call__(self) -> str | None: ...


class FrameReservation(Protocol):
    """等待已接纳的最终 Output 写出；route 结束时抛出 LookupError。"""

    async def wait_output(self, message_id: str) -> None: ...


class FrameClaim(Protocol):
    """持有一个 ToolCall 的最终写出回执，直到消费或放弃。"""

    @property
    def ending_message_id(self) -> str | None: ...
    async def wait_output(self) -> None: ...
    def consume(self) -> None: ...
    def abort(self) -> None: ...


class FrameRouteStage(Protocol):
    """操作成功才提交替代 route；放弃不改变当前 owner。"""

    @property
    def reservation(self) -> FrameReservation: ...
    def commit(self) -> FrameReservation: ...
    def abort(self) -> None: ...


class ControlFrames(Protocol):
    """绑定连接与 Input 的实际写出回执；缺失或结束的 route 抛出 LookupError。"""

    def route_input_with_owner(self, session_id: str, input_id: str, connection_id: str,
                               resolver: FrameResolver) -> tuple[FrameReservation, bool]: ...
    def stage_input(self, session_id: str, input_id: str, connection_id: str,
                    resolver: FrameResolver) -> FrameRouteStage: ...
    async def wait_input(self, session_id: str, input_id: str, ending: str) -> None: ...
    def release_input(self, session_id: str, input_id: str) -> None: ...
    def settle_input(self, session_id: str, input_id: str, error: BaseException | None = None) -> None: ...
    def active_session_ids(self) -> frozenset[str]: ...
    def active_input_ids(self, session_id: str) -> tuple[str, ...]: ...
    def arm_claim(self, session_id: str, input_id: str, call_ref: CallRef) -> FrameClaim: ...
    def claim_for(self, session_id: str, call_ref: CallRef) -> FrameClaim | None: ...
    def track_page(self, connection_id: str, page: Mapping[str, object], written: Future[None]) -> bool: ...
    def fail_connection(self, connection_id: str, error: BaseException) -> None: ...


CONTROL_FRAMES = ServiceKey[ControlFrames]("gateway.frames.v1")
