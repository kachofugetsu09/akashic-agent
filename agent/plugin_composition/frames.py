"""控制帧（control frames）的组合内核归属。

`core.control_frames.v1` 是 Core 拥有的能力 key：程序化调用与消息投递用它登记
「一次 Input 的最终 Output 路由」。此前 key 定义在 `agent.control.frame_book`，
插件要声明依赖就必须 import Core 的实现模块。按设计把它归位到组合内核，并用
Protocol 描述插件可见的方法子集；`ServiceKey` 只按 name 相等，归位零运行时语义。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from agent.plugin_composition.model import ServiceKey
from agent.plugin_contracts.message import CallRef

# 解析「某个 Input 的最终 Output」的回调；由消费者提供。
FrameResolver = Callable[[], str | None]


@runtime_checkable
class FrameClaimPort(Protocol):
    """一次 ToolCall 的最终 Output 认领；由帧簿签发。"""

    call_ref: CallRef
    ending_message_id: str | None

    async def wait_output(self) -> None:
        """等待本次认领对应的最终 Output。"""
        ...

    def consume(self) -> None:
        """标记已消费，释放认领。"""
        ...

    def abort(self) -> None:
        """放弃认领，释放路由。"""
        ...


@runtime_checkable
class FrameRouteStagePort(Protocol):
    """替换路由的候选；提交后才对其它消费者可见。"""

    def commit(self) -> object:
        """提交候选路由。"""
        ...

    def abort(self) -> None:
        """放弃候选路由。"""
        ...


@runtime_checkable
class FrameBookPort(Protocol):
    """帧簿对插件可见的方法子集。"""

    def route_input_with_owner(
        self,
        session_id: str,
        input_id: str,
        connection_id: str,
        resolver: FrameResolver,
    ) -> tuple[object, bool]:
        """登记一条 Input 路由，并报告本次调用是否创建了 owner。"""
        ...

    def stage_input(
        self,
        session_id: str,
        input_id: str,
        connection_id: str,
        resolver: FrameResolver,
    ) -> FrameRouteStagePort:
        """准备一条恢复路由，但不改变当前 active 路由。"""
        ...

    def active_input_ids(self, session_id: str) -> tuple[str, ...]:
        """返回该 Session 当前的 Input 身份。"""
        ...

    def release_input(self, session_id: str, input_id: str) -> None:
        """释放未被认领的普通路由。"""
        ...

    def settle_input(
        self, session_id: str, input_id: str, error: BaseException | None = None
    ) -> None:
        """结束来源已确认的普通路由。"""
        ...

    async def wait_input(self, session_id: str, input_id: str, ending: str) -> None:
        """等待某次 Input 的指定终态。"""
        ...

    def arm_claim(self, session_id: str, input_id: str, call_ref: CallRef) -> FrameClaimPort:
        """登记一个 ToolCall 认领。"""
        ...

    def claim_for(self, session_id: str, call_ref: CallRef) -> FrameClaimPort | None:
        """返回某个 ToolCall 当前存活的认领。"""
        ...


CONTROL_FRAMES = ServiceKey[FrameBookPort]("core.control_frames.v1")
