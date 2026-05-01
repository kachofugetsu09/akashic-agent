from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent.looping.handlers import process_spawn_completion_event
from bus.events import InboundItem, InboundMessage, OutboundMessage, SpawnCompletionItem

if TYPE_CHECKING:
    from agent.core.passive_turn import AgentCore
    from agent.core.runtime_support import AgentLoopRunner, PromptRenderRunner
    from agent.context import ContextBuilder
    from agent.looping.ports import SessionServices
    from agent.tools.registry import ToolRegistry


@dataclass
class CoreRunnerDeps:
    agent_core: "AgentCore"
    session: "SessionServices | None" = None
    context: "ContextBuilder | None" = None
    tools: "ToolRegistry | None" = None
    memory_window: int | None = None
    run_agent_loop_fn: "AgentLoopRunner | None" = None
    prompt_render_fn: "PromptRenderRunner | None" = None


class CoreRunner:
    """
    ┌──────────────────────────────────────┐
    │ CoreRunner                           │
    ├──────────────────────────────────────┤
    │ 1. 判断是否内部事件                  │
    │ 2. spawn completion 走 helper        │
    │ 3. 普通被动消息走 AgentCore          │
    └──────────────────────────────────────┘
    """

    def __init__(self, deps: CoreRunnerDeps) -> None:
        self._agent_core = deps.agent_core
        self._session = deps.session
        self._context = deps.context
        self._tools = deps.tools
        self._memory_window = deps.memory_window
        self._run_agent_loop_fn = deps.run_agent_loop_fn
        self._prompt_render_fn = deps.prompt_render_fn

    async def process(
        self,
        msg: InboundItem,
        key: str,
        *,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        # 1. 先处理 typed 内部工作项，统一走默认 helper 链。
        match msg:
            case SpawnCompletionItem():
                if (
                    self._session is not None
                    and self._tools is not None
                    and self._memory_window is not None
                    and self._run_agent_loop_fn is not None
                    and self._prompt_render_fn is not None
                ):
                    return await process_spawn_completion_event(
                        item=msg,
                        key=key,
                        session_svc=self._session,
                        pipeline=self._agent_core.pipeline,
                        tools=self._tools,
                        memory_window=self._memory_window,
                        run_agent_loop_fn=self._run_agent_loop_fn,
                        prompt_render_fn=self._prompt_render_fn,
                        dispatch_outbound=dispatch_outbound,
                    )
                raise RuntimeError("spawn completion 缺少处理依赖")
            case InboundMessage():
                # 2. 默认普通被动消息统一走 AgentCore。
                return await self._agent_core.process(
                    msg,
                    key,
                    dispatch_outbound=dispatch_outbound,
                )
        raise TypeError(f"unsupported inbound item: {type(msg).__name__}")
