from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from core.error_context import current_session_key
from core.common.diagnostic_log import diagnostic_line
from agent.control.context import current_turn_id
from agent.control.ids import new_turn_id
from agent.context import ContextBuilder
from agent.core.passive_turn import (
    AgentCore,
    AgentCoreDeps,
    DefaultContextStore,
    DefaultReasoner,
)
from agent.looping.interrupt import InterruptResult, TurnInterruptState
from agent.core.runner import CoreRunner, CoreRunnerDeps
from agent.core.runtime_support import ToolDiscoveryState
from agent.looping.ports import (
    AgentLoopConfig,
    AgentLoopDeps,
    LLMConfig,
    LLMServices,
    MemoryConfig,
    MemoryServices,
    SessionServices,
)
from agent.retrieval.default_pipeline import DefaultMemoryRetrievalPipeline
from agent.retrieval.protocol import MemoryRetrievalPipeline
from agent.turns.outbound import BusOutboundPort

# 为保持兼容重新导出：现有调用方从 core.py 导入这些名称。
__all__ = [
    "AgentLoop",
]
from bus.event_bus import EventBus
from bus.events import (
    InboundItem,
    InboundMessage,
    OutboundMessage,
    SpawnCompletionItem,
)
from bus.events_lifecycle import (
    StreamDeltaReady,
    TurnStarted,
)
from bus.processing import ProcessingState
from bus.queue import MessageBus
from proactive_v2.presence import PresenceStore
from agent.provider import LLMProvider
from agent.tools.shell import ShellTool
from agent.tools.unified_exec import ExecutionCleanupReport
from agent.tools.registry import ToolRegistry
from session.manager import SessionManager

if TYPE_CHECKING:
    from core.memory.engine import MemoryEngine
    from core.memory.markdown import MemoryProfileApi
    from core.memory.runtime import MemoryRuntime
    from agent.tool_hooks.base import ToolHook
    from agent.plugins.snapshot import RuntimeSnapshotStore

logger = logging.getLogger("agent.loop")
_MANUAL_CONSOLIDATION_TIMEOUT_SECONDS = 30.0

StreamDelta: TypeAlias = dict[str, str] | str
StreamSink: TypeAlias = Callable[[StreamDelta], Awaitable[None]]
StreamSinkFactory: TypeAlias = Callable[[object], StreamSink | None]
StreamSupportPolicy: TypeAlias = Callable[[str], bool]


def _is_positive_int(value: str) -> bool:
    try:
        return int(value) > 0
    except ValueError:
        return False


def _is_nonempty(value: str) -> bool:
    return bool(value)


_STREAM_SUPPORT_POLICIES: dict[str, StreamSupportPolicy] = {
    "programmatic": _is_nonempty,
    "telegram": _is_positive_int,
    "web": _is_nonempty,
    "mobile": _is_nonempty,
    # 飞书私聊渠道：chat_id 形如 oc_xxx，全程支持流式预览（卡片 PATCH 消费 StreamDeltaReady）。
    "feishu": _is_nonempty,
}


def _supports_stream_events(channel: str, chat_id: str) -> bool:
    policy = _STREAM_SUPPORT_POLICIES.get(channel)
    return bool(policy is not None and policy(chat_id))


def _suppresses_stream_events(msg: object) -> bool:
    metadata: object = getattr(msg, "metadata", None)
    if not isinstance(metadata, dict):
        return False
    typed = cast(dict[str, object], metadata)
    return bool(typed.get("suppress_stream_events"))


def _item_content(item: InboundItem) -> str:
    if isinstance(item, InboundMessage):
        return item.content
    return f"[后台任务完成] {item.event.label or item.event.status or item.event.job_id}"


class AgentLoop:
    """
    主循环：从 MessageBus 消费 InboundMessage，
    驱动 LLM + 工具调用，将结果发回 MessageBus。
    对话历史按 session_key 独立维护，格式为 OpenAI messages。
    """

    def __init__(
        self,
        deps: AgentLoopDeps,
        config: AgentLoopConfig,
    ) -> None:
        # 1. 先挂基础运行时对象和配置。
        self._llm_config = config.llm
        self.bus = deps.bus
        self.tools = deps.tools
        self.memory_window = config.memory.window
        self._running = False
        self._processing_state = deps.processing_state
        self._event_bus = deps.event_bus or EventBus()
        self._passive_runtime_lock = asyncio.Lock()
        self._runtime_snapshot_store: RuntimeSnapshotStore | None = None

        # ── 中断控制面（纯内存态） ──
        self._active_tasks: dict[str, asyncio.Task[OutboundMessage]] = {}
        self._active_turn_states: dict[str, TurnInterruptState] = {}
        self._interrupt_states: dict[str, TurnInterruptState] = {}

        # 2. 再解析 memory runtime 入口。
        memory_engine = self._resolve_memory_runtime(deps)
        markdown_memory = self._resolve_markdown_runtime(deps)
        self._tool_search_enabled = bool(config.llm.tool_search_enabled)
        self._memory_engine = memory_engine
        self._markdown_memory = markdown_memory
        memory_profile = (
            markdown_memory.store
            if markdown_memory is not None
            else cast("MemoryProfileApi", self._memory_engine)
        )
        self._context = deps.context or ContextBuilder(
            deps.workspace,
            memory=memory_profile,
            multimodal=config.llm.multimodal,
            vl_available=config.llm.vl_available,
        )
        self._llm_services = deps.llm_services or LLMServices(
            provider=deps.provider,
            light_provider=deps.light_provider or deps.provider,
        )
        self._session_services = deps.session_services or SessionServices(
            session_manager=deps.session_manager,
            presence=deps.presence,
        )

        # 3. 最后把 passive chain 装起来。
        self._assemble_passive_runtime(
            deps=deps,
            config=config,
        )
        self._configure_stream_events()

    def set_stream_sink_factory(self, factory: StreamSinkFactory | None) -> None:
        setter = getattr(self._reasoner, "set_stream_sink_factory", None)
        if callable(setter):
            _ = setter(self._wrap_stream_sink_factory(factory))

    def bind_runtime_snapshot_store(self, store: RuntimeSnapshotStore) -> None:
        self._runtime_snapshot_store = store

    def _configure_stream_events(self) -> None:
        setter = getattr(self._reasoner, "set_stream_sink_factory", None)
        if callable(setter):
            _ = setter(self._build_stream_event_sink)

    def _wrap_stream_sink_factory(
        self,
        factory: StreamSinkFactory | None,
    ) -> StreamSinkFactory | None:
        if factory is None:
            return None

        def _build(msg: object) -> StreamSink | None:
            if _suppresses_stream_events(msg):
                return None
            downstream = factory(msg)
            channel = str(getattr(msg, "channel", ""))
            chat_id = str(getattr(msg, "chat_id", ""))
            session_key = str(getattr(msg, "session_key", f"{channel}:{chat_id}"))
            if downstream is None:
                return None

            async def _push(delta: StreamDelta) -> None:
                if isinstance(delta, str):
                    payload = {"content_delta": delta}
                else:
                    payload = delta
                content_delta = payload.get("content_delta")
                if isinstance(content_delta, str) and content_delta:
                    self._append_partial_reply(session_key, content_delta)
                thinking_delta = payload.get("thinking_delta")
                if isinstance(thinking_delta, str) and thinking_delta:
                    self._append_partial_thinking(session_key, thinking_delta)
                await downstream(payload)

            return _push

        return _build

    def _build_stream_event_sink(self, msg: object) -> StreamSink | None:
        channel = str(getattr(msg, "channel", ""))
        chat_id = str(getattr(msg, "chat_id", ""))
        if _suppresses_stream_events(msg):
            return None
        if not _supports_stream_events(channel, chat_id):
            return None
        session_key = str(getattr(msg, "session_key", f"{channel}:{chat_id}"))

        async def _push(delta: StreamDelta) -> None:
            if isinstance(delta, str):
                payload = {"content_delta": delta}
            else:
                payload = delta
            content_delta = payload.get("content_delta")
            if isinstance(content_delta, str) and content_delta:
                self._append_partial_reply(session_key, content_delta)
            thinking_delta = payload.get("thinking_delta")
            if isinstance(thinking_delta, str) and thinking_delta:
                self._append_partial_thinking(session_key, thinking_delta)
            await self._event_bus.observe(
                StreamDeltaReady(
                    session_key=session_key,
                    channel=channel,
                    chat_id=chat_id,
                    turn_id=current_turn_id.get(),
                    content_delta=content_delta if isinstance(content_delta, str) else "",
                    thinking_delta=thinking_delta if isinstance(thinking_delta, str) else "",
                )
            )

        return _push

    def _append_partial_reply(self, session_key: str, delta: str) -> None:
        state = self._active_turn_states.get(session_key)
        if state is None or not delta:
            return
        state.partial_reply += delta

    def _append_partial_thinking(self, session_key: str, delta: str) -> None:
        state = self._active_turn_states.get(session_key)
        if state is None or not delta:
            return
        state.partial_thinking = (state.partial_thinking or "") + delta

    def _resolve_memory_runtime(
        self,
        deps: AgentLoopDeps,
    ) -> "MemoryEngine":
        if deps.memory_runtime is not None:
            return deps.memory_runtime.engine
        if deps.memory_services is not None and deps.memory_services.engine is not None:
            return deps.memory_services.engine
        raise ValueError("AgentLoop requires memory_runtime.engine")

    def _resolve_markdown_runtime(
        self,
        deps: AgentLoopDeps,
    ):
        if deps.memory_runtime is not None:
            return deps.memory_runtime.markdown
        return None

    def _assemble_passive_runtime(
        self,
        *,
        deps: AgentLoopDeps,
        config: AgentLoopConfig,
    ) -> None:
        # 1. 先组基础 service ports。
        llm_svc = self._llm_services
        memory_svc = MemoryServices(engine=self._memory_engine)
        session_svc = self._session_services
        # 2. 组执行层。
        self._tool_discovery = deps.tool_discovery or ToolDiscoveryState()
        self._reasoner = deps.reasoner or DefaultReasoner(
            llm=llm_svc,
            llm_config=config.llm,
            tools=deps.tools,
            discovery=self._tool_discovery,
            tool_search_enabled=self._tool_search_enabled,
            memory_window=config.memory.keep_count,
            context=self._context,
            event_bus=self._event_bus,
            non_preloadable_names=deps.tools.get_non_preloadable_names,
        )

        # 3. 最后串 passive prepare / execute / commit 主链。
        retrieval_pipeline = deps.retrieval_pipeline or DefaultMemoryRetrievalPipeline(
            memory=memory_svc,
        )
        self._retrieval_pipeline = retrieval_pipeline
        passive_context_store = DefaultContextStore(
            retrieval=retrieval_pipeline,
            context=self._context,
            history_window=config.memory.keep_count,
        )
        agent_core = AgentCore(
            AgentCoreDeps(
                session=session_svc,
                context_store=passive_context_store,
                context=self._context,
                tools=deps.tools,
                reasoner=self._reasoner,
                event_bus=self._event_bus,
                outbound_port=BusOutboundPort(self.bus),
                history_window=config.memory.keep_count,
                memory_consolidator=self,
            )
        )
        self._agent_core = agent_core
        self._core_runner = deps.core_runner or CoreRunner(
            CoreRunnerDeps(
                agent_core=agent_core,
                session=session_svc,
                context=self._context,
                tools=deps.tools,
                memory_window=config.memory.keep_count,
                run_agent_loop_fn=self._run_agent_loop,
                prompt_render_fn=self._reasoner.render_prompt,
            )
        )

    @property
    def light_model(self) -> str:
        # 1. 兼容外部读取 loop.light_model，真实值统一来自 llm 配置。
        return self._llm_config.light_model or self._llm_config.model

    @property
    def context(self) -> ContextBuilder:
        # 1. 兼容外部读取 loop.context，真实值统一来自私有 context 依赖。
        return self._context

    @property
    def light_provider(self):
        # 1. 兼容外部读取 loop.light_provider，真实值统一来自 llm services。
        return self._llm_services.light_provider

    @property
    def session_manager(self):
        # 1. 兼容外部读取 loop.session_manager，真实值统一来自 session services。
        return self._session_services.session_manager

    @light_model.setter
    def light_model(self, value: str) -> None:
        # 1. 兼容初始化期和少量外部覆写，统一回写到 llm 配置。
        self._llm_config.light_model = value

    @property
    def max_iterations(self) -> int:
        # 1. 兼容外部读取 loop.max_iterations，真实值统一来自 llm 配置。
        return int(self._llm_config.max_iterations)

    @max_iterations.setter
    def max_iterations(self, value: int) -> None:
        # 1. 兼容测试或外部直接改 loop.max_iterations，真实执行也同步生效。
        self._llm_config.max_iterations = int(value)

    async def run(self) -> None:
        """消费入站消息，并在每轮结束时收束总线与内存状态。"""

        self._running = True
        logger.info(f"AgentLoop 启动  max_iter={self.max_iterations}")
        try:
            while self._running:
                # 1. 等待下一条入站消息，空闲超时仅用于重新检查停止状态。
                try:
                    item = await asyncio.wait_for(
                        self.bus.consume_inbound(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue
                await self._run_inbound_turn(item)
        finally:
            self._running = False

    async def _run_inbound_turn(self, item: InboundItem) -> None:
        """执行一个入站 turn，并在状态清理后确认消息。"""

        # 1. 建立本轮中断状态和执行任务。
        key = item.session_key
        self._active_turn_states[key] = self._build_initial_turn_state(item, key)
        task = asyncio.create_task(
            self._process_with_runtime_admission(item),
            name=f"agent-turn:{key}",
        )
        self._active_tasks[key] = task

        # 2. 只吞掉本轮取消；运行器取消必须继续向生命周期 owner 传播。
        try:
            await task
        except asyncio.CancelledError:
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise
            logger.info(f"Turn cancelled for {key}")
        except Exception as e:
            logger.error(f"处理消息出错: {e}", exc_info=True)
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=item.channel,
                    chat_id=item.chat_id,
                    content=f"出错：{e}",
                )
            )
        finally:
            # 3. 先收束内存状态，再完成总线确认。
            del self._active_tasks[key]
            del self._active_turn_states[key]
            await self._complete_inbound(item)

    async def _complete_inbound(self, item: InboundItem) -> None:
        """在本轮清理中完成入站确认，并保留确认错误。"""

        # 1. 让确认独立于 AgentLoop task，避免运行器取消留下未完成的 lane 计数。
        completion = asyncio.create_task(
            self.bus.complete_inbound(item),
            name=f"agent-inbound-ack:{item.session_key}",
        )
        try:
            await asyncio.shield(completion)
        except asyncio.CancelledError as cancellation:
            # 2. 确认必须先收束；确认本身失败时保留真实错误。
            await completion
            raise cancellation

    @property
    def processing_state(self) -> ProcessingState | None:
        return self._processing_state

    @property
    def active_turn_states(self) -> dict[str, TurnInterruptState]:
        return self._active_turn_states

    def stop(self) -> None:
        self._running = False
        for task in self._active_tasks.values():
            if not task.done():
                _ = task.cancel()
        logger.info("AgentLoop 停止")

    def add_tool_hooks(self, hooks: list["ToolHook"]) -> None:
        self._reasoner.add_tool_hooks(hooks)

    def add_before_turn_plugin_modules(
        self,
        modules: list[object],
    ) -> None:
        self._agent_core.add_before_turn_plugin_modules(modules)

    def add_before_reasoning_plugin_modules(
        self,
        modules: list[object],
    ) -> None:
        self._agent_core.add_before_reasoning_plugin_modules(modules)

    def add_after_reasoning_plugin_modules(
        self,
        modules: list[object],
    ) -> None:
        self._agent_core.add_after_reasoning_plugin_modules(modules)

    def add_after_turn_plugin_modules(
        self,
        modules: list[object],
    ) -> None:
        self._agent_core.add_after_turn_plugin_modules(modules)

    def add_prompt_render_plugin_modules(
        self,
        modules: list[object],
    ) -> None:
        self._reasoner.add_prompt_render_plugin_modules(modules)

    def add_before_step_plugin_modules(
        self,
        modules: list[object],
    ) -> None:
        self._reasoner.add_before_step_plugin_modules(modules)

    def add_after_step_plugin_modules(
        self,
        modules: list[object],
    ) -> None:
        self._reasoner.add_after_step_plugin_modules(modules)

    # ── 中断控制面 ────────────────────────────────────────────────

    def request_interrupt(
        self,
        session_key: str,
        sender: str = "",
        command: str = "/stop",
    ) -> InterruptResult:
        """Channel 层调用的中断入口，不走 MessageBus。"""
        task = self._active_tasks.get(session_key)
        if task is None or task.done():
            return InterruptResult(
                status="idle",
                session_key=session_key,
                message="当前没有正在执行的任务。",
            )

        # 保存中断态（纯内存，不落库）
        active_state = self._active_turn_states.get(session_key)
        if active_state is None:
            active_state = TurnInterruptState(
                session_key=session_key,
                original_user_message="",
            )
        self._interrupt_states[session_key] = replace(
            active_state,
            interrupted_by=command,
            interrupted_at=time.monotonic(),
        )
        task.cancel()
        logger.info(
            f"Turn interrupted  session_key={session_key}  "
            f"sender={sender}  command={command}"
        )
        return InterruptResult(
            status="interrupted",
            session_key=session_key,
            message="本轮已中断。你可以继续补充要求，我会接着这件事处理。",
        )

    def _get_interrupt_state(self, session_key: str) -> TurnInterruptState | None:
        """读取中断态（含 TTL 过期检查），不提前消费。"""
        state = self._interrupt_states.get(session_key)
        if state is None:
            return None
        if state.expired:
            logger.info(f"Interrupt state expired for {session_key}, discarding")
            self._interrupt_states.pop(session_key, None)
            return None
        return state

    def _build_initial_turn_state(
        self,
        item: InboundItem,
        key: str,
    ) -> TurnInterruptState:
        # 1. 普通消息保留真实用户输入，spawn 回传用固定 marker 表示内部工作项。
        match item:
            case InboundMessage():
                return TurnInterruptState(
                    session_key=key,
                    original_user_message=item.content,
                    original_metadata=dict(item.metadata or {}),
                )
            case SpawnCompletionItem():
                return TurnInterruptState(
                    session_key=key,
                    original_user_message=_item_content(item),
                    original_metadata={},
                )
        raise TypeError(f"unsupported inbound item: {type(item).__name__}")

    async def _resume_interrupted_message(
        self,
        msg: InboundItem,
        key: str,
    ) -> tuple[InboundItem, bool]:
        # 1. 只有普通入站消息参与续跑，内部工作项不消费中断态。
        if not isinstance(msg, InboundMessage):
            return msg, False
        interrupted = self._get_interrupt_state(key)
        if interrupted is None:
            return msg, False

        # 2. 有中断态时，补一段结构化历史；当前用户消息保持原文。
        await self._persist_interrupted_turn_marker(key, interrupted)
        resumed = InboundMessage(
            channel=msg.channel,
            sender=msg.sender,
            chat_id=msg.chat_id,
            content=msg.content,
            timestamp=msg.timestamp,
            media=msg.media,
            metadata={**(msg.metadata or {}), "resumed_from_interrupt": True},
        )
        logger.info(f"Resuming interrupted turn for {key}")
        self._active_turn_states[key] = TurnInterruptState(
            session_key=key,
            original_user_message=msg.content,
            original_metadata=dict(resumed.metadata or {}),
        )
        return resumed, True

    async def _persist_interrupted_turn_marker(
        self,
        key: str,
        state: TurnInterruptState,
    ) -> None:
        if not state.original_user_message.strip():
            return
        session = self.session_manager.get_or_create(key)
        start = len(session.messages)
        session.add_message(
            "user",
            state.original_user_message,
        )
        tool_chain = (
            cast(list[dict[str, Any]], list(state.tool_chain_partial))
            if state.tool_chain_partial
            else None
        )
        session.add_message(
            "assistant",
            "[interrupted]",
            tools_used=list(state.tools_used) if state.tools_used else None,
            tool_chain=tool_chain,
            skip_post_memory=True,
        )
        await self.session_manager.append_messages(session, session.messages[start:])

    async def _observe_turn_started(
        self,
        msg: InboundItem,
        key: str,
    ) -> None:
        # 1. 对外发布被动 turn 开始事件，具体副作用由 observer 决定。
        await self._event_bus.observe(
            TurnStarted(
                session_key=key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=_item_content(msg),
                timestamp=msg.timestamp,
                turn_id=current_turn_id.get(),
            )
        )

    # ── 被动 turn 处理 ────────────────────────────────────────────

    async def _process(
        self,
        msg: InboundItem,
        session_key: str | None = None,
        busy_session_key: str | None = None,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        key = session_key or msg.session_key
        busy_key = busy_session_key or key
        # 给本 turn task 打上 session 归属，供 observe 全局错误采集关联。
        session_token = current_session_key.set(key)
        inherited_turn_id = (
            str(msg.metadata.get("control_turn_id") or "")
            if isinstance(msg, InboundMessage)
            else ""
        )
        turn_token = current_turn_id.set(inherited_turn_id or new_turn_id())
        try:
            # 1. 先处理可能存在的续跑态，并发布 turn started。
            msg, resumed_from_interrupt = await self._resume_interrupted_message(msg, key)
            await self._observe_turn_started(msg, key)
            content = _item_content(msg)
            preview = content[:60] + "..." if len(content) > 60 else content
            logger.info(f"Processing message from {msg.channel}: {preview}")

            # 2. 再进入 busy 状态并执行核心处理。
            if self._processing_state:
                self._processing_state.enter(busy_key)
            try:
                outbound = await self._core_runner.process(
                    msg,
                    key,
                    dispatch_outbound=dispatch_outbound,
                )
                if resumed_from_interrupt:
                    self._interrupt_states.pop(key, None)
                return outbound
            finally:
                # 3. 无论核心处理结果如何，都释放 busy 状态。
                if self._processing_state:
                    self._processing_state.exit(busy_key)
        finally:
            # 4. 当前 query 结束即回收其 shell，再恢复调用方上下文。
            try:
                await self._cleanup_shell_owner(key)
            finally:
                current_session_key.reset(session_token)
                current_turn_id.reset(turn_token)

    async def _cleanup_shell_owner(self, owner_session_key: str) -> None:
        """回收 turn 的 Shell，并把失败隔离为 execution 诊断。"""

        # 1. 未预期的 cleanup 异常只进入日志，不拥有 turn 终态。
        try:
            report = await self._terminate_shell_owner(owner_session_key)
        except Exception as exc:
            logger.exception(
                diagnostic_line(
                    "AgentLoop.shell_cleanup",
                    event="cleanup_degraded",
                    flow="passive",
                    phase="cleanup",
                    session=owner_session_key,
                    turn=current_turn_id.get(),
                    action="retain_turn_finality",
                    reason="cleanup_exception",
                    error_type=type(exc).__name__,
                    note=str(exc),
                )
            )
            return

        # 2. manager 的已知失败保留 execution ownership 和完整明细。
        if report is not None and report.failures:
            failures = ";".join(
                f"{failure.execution_id}:{failure.error_type}:{failure.message}"
                for failure in report.failures
            )
            logger.error(
                diagnostic_line(
                    "AgentLoop.shell_cleanup",
                    event="cleanup_degraded",
                    flow="passive",
                    phase="cleanup",
                    session=owner_session_key,
                    turn=current_turn_id.get(),
                    action="retain_turn_finality",
                    reason="execution_cleanup_unconfirmed",
                    counts=(
                        f"attempted:{len(report.attempted_execution_ids)},"
                        f"failed:{len(report.failures)}"
                    ),
                    note=failures,
                )
            )

    async def _terminate_shell_owner(
        self,
        owner_session_key: str,
    ) -> ExecutionCleanupReport | None:
        shell = self.tools.get_tool("shell")
        if shell is None:
            return None
        if not isinstance(shell, ShellTool):
            raise TypeError("注册名 shell 必须由 ShellTool 拥有生命周期")
        return await shell.terminate_owner(owner_session_key)

    async def _process_with_runtime_admission(
        self,
        msg: InboundItem,
        session_key: str | None = None,
        busy_session_key: str | None = None,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        key = session_key or msg.session_key
        if self._passive_runtime_lock.locked():
            logger.info("[runtime_admission] 等待 passive runtime session=%s", key)
        async with self._passive_runtime_lock:
            store = self._runtime_snapshot_store
            if store is None or store.current is None:
                return await self._process(
                    msg,
                    session_key=session_key,
                    busy_session_key=busy_session_key,
                    dispatch_outbound=dispatch_outbound,
                )
            from agent.plugins.snapshot import (
                bind_runtime_snapshot,
                reset_runtime_snapshot,
            )

            lease = await store.acquire()
            async with lease as snapshot:
                token = bind_runtime_snapshot(lease)
                try:
                    return await self._process(
                        msg,
                        session_key=session_key,
                        busy_session_key=busy_session_key,
                        dispatch_outbound=dispatch_outbound,
                    )
                finally:
                    reset_runtime_snapshot(token)

    async def process_direct(
        self,
        content: str,
        session_key: str = "programmatic:direct",
        busy_session_key: str | None = None,
        channel: str = "programmatic",
        chat_id: str = "direct",
        omit_user_turn: bool = False,
        skip_post_memory: bool = False,
        skip_memory_retrieval: bool = False,
        stream_events: bool = False,
        disabled_tools: list[str] | None = None,
        sender: str = "user",
        media: list[str] | None = None,
        turn_id: str = "",
        stateless: bool = False,
    ) -> str:
        response = await self.process_direct_message(
            content,
            session_key=session_key,
            busy_session_key=busy_session_key,
            channel=channel,
            chat_id=chat_id,
            omit_user_turn=omit_user_turn,
            skip_post_memory=skip_post_memory,
            skip_memory_retrieval=skip_memory_retrieval,
            stream_events=stream_events,
            disabled_tools=disabled_tools,
            sender=sender,
            media=media,
            turn_id=turn_id,
            stateless=stateless,
        )
        return response.content

    async def process_direct_message(
        self,
        content: str,
        session_key: str = "programmatic:direct",
        busy_session_key: str | None = None,
        channel: str = "programmatic",
        chat_id: str = "direct",
        omit_user_turn: bool = False,
        skip_post_memory: bool = False,
        skip_memory_retrieval: bool = False,
        stream_events: bool = False,
        disabled_tools: list[str] | None = None,
        sender: str = "user",
        media: list[str] | None = None,
        metadata: dict[str, object] | None = None,
        turn_id: str = "",
        stateless: bool = False,
    ) -> OutboundMessage:
        """执行直接消息，并按需隔离会话历史与持久化。"""

        inbound_metadata = dict(metadata or {})
        if stateless:
            inbound_metadata.update(
                {
                    "omit_user_turn": True,
                    "omit_assistant_turn": True,
                    "skip_session_history": True,
                    "skip_memory_context_guard": True,
                    "skip_post_memory": True,
                    "skip_memory_retrieval": True,
                }
            )
        if omit_user_turn:
            inbound_metadata["omit_user_turn"] = True
        if skip_post_memory:
            inbound_metadata["skip_post_memory"] = True
        if skip_memory_retrieval:
            inbound_metadata["skip_memory_retrieval"] = True
        if not stream_events:
            inbound_metadata["suppress_stream_events"] = True
        if disabled_tools:
            inbound_metadata["disabled_tools"] = list(disabled_tools)
        if turn_id:
            inbound_metadata["control_turn_id"] = turn_id
        msg = InboundMessage(
            channel=channel,
            sender=sender,
            chat_id=chat_id,
            content=content,
            media=list(media or []),
            metadata=inbound_metadata,
        )
        response = await self._process_with_runtime_admission(
            msg,
            session_key=session_key,
            busy_session_key=busy_session_key,
            dispatch_outbound=False,
        )
        return response

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        request_time: datetime | None = None,
        preloaded_tools: set[str] | None = None,
    ) -> tuple[str, list[str], list[dict], set[str] | None, str | None]:
        from agent.core.passive_turn import build_turn_injection_prompt
        from agent.prompting import (
            PromptSectionRender,
            build_context_frame_content,
            build_context_frame_message,
        )

        # 1. 补充 deferred tools hint（与 run_turn 路径保持一致）。
        visible = preloaded_tools if self._tool_search_enabled else None
        hint = build_turn_injection_prompt(
            tools=self.tools,
            tool_search_enabled=self._tool_search_enabled,
            visible_names=visible,
        )
        if hint:
            hint_message = build_context_frame_message(
                build_context_frame_content(
                    [
                        PromptSectionRender(
                            name="turn_injection",
                            content=hint,
                            is_static=False,
                        )
                    ]
                )
            )
            if initial_messages and initial_messages[-1].get("role") == "user":
                initial_messages = initial_messages[:-1] + [
                    hint_message,
                    initial_messages[-1],
                ]
            else:
                initial_messages = initial_messages + [hint_message]

        # 2. 内部事件链统一直接走新 Reasoner。
        result = await self._reasoner.run(
            initial_messages,
            request_time=request_time,
            preloaded_tools=preloaded_tools,
            preflight_injected=True,
        )
        tools_used = list(result.metadata.get("tools_used") or [])
        tool_chain = list(result.metadata.get("tool_chain") or [])
        visible_names = result.metadata.get("visible_names")
        return result.reply, tools_used, tool_chain, visible_names, result.thinking

    async def trigger_memory_consolidation(
        self,
        session_key: str,
        *,
        archive_all: bool = False,
        force: bool = False,
        drain_backlog: bool = True,
    ) -> bool:
        from core.memory.markdown import ConsolidateRequest

        session = self.session_manager.get_or_create(session_key)
        if self._markdown_memory is None:
            raise RuntimeError("markdown memory runtime unavailable")
        maintenance = self._markdown_memory.maintenance
        operation = maintenance.consolidate(
            ConsolidateRequest(
                session=session,
                archive_all=archive_all,
                force=force,
                drain_backlog=drain_backlog,
            )
        )
        if drain_backlog:
            result = await operation
        else:
            try:
                result = await asyncio.wait_for(
                    operation,
                    timeout=_MANUAL_CONSOLIDATION_TIMEOUT_SECONDS,
                )
            except TimeoutError as exc:
                raise TimeoutError("memory consolidation busy") from exc
        if result.trace.get("mode") == "markdown":
            await self.session_manager.save_async(session)
            return True
        return False


# ── 模块级辅助 ────────────────────────────────────────────────────
