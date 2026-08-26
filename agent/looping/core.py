from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

from core.error_context import current_client_message_id, current_session_key
from core.common.diagnostic_log import diagnostic_line
from agent.control.context import running_turn_id
from agent.control.ids import new_turn_id
from agent.context import ContextBuilder
from agent.core.passive_turn import (
    DefaultContextStore,
    DefaultReasoner,
    PassiveTurnDeps,
    PassiveTurnPipeline,
)
from agent.looping.interrupt import InterruptResult, TurnInterruptState
from agent.core.runtime_support import ToolDiscoveryState
from agent.looping.ports import (
    AgentLoopConfig,
    AgentLoopDeps,
    LLMConfig,
    LLMServices,
    SessionServices,
)
from agent.looping.session_lane import SessionLaneRegistry
from agent.model_runtime.registry import RoleBoundProvider, model_execution_scope
from agent.model_runtime.session_selection import (
    SessionModelSelection,
    read_session_model_selection,
    write_session_model_selection,
)
from agent.turns.outbound import OutboundDispatch
from agent.turn_effects import PostCommitEffect
from agent.plugin_composition.channels import InboundEnvelope, InboundOwner

# 为保持兼容重新导出：现有调用方从 core.py 导入这些名称。
__all__ = [
    "AgentLoop",
]
from bus.event_bus import EventBus
from bus.events import (
    InboundItem,
    InboundMessage,
    OutboundMessage,
)
from bus.events_lifecycle import (
    StreamDeltaReady,
    TurnStarted,
)
from bus.processing import ProcessingState
from bus.queue import MessageBus
from session.activity import PresenceStore
from agent.provider import LLMProvider
from agent.tools.shell import ShellTool
from agent.tools.unified_exec import ExecutionCleanupReport
from agent.tools.registry import ToolRegistry
from session.compaction_runtime import SessionCompactionRuntime
from session.manager import SessionManager

if TYPE_CHECKING:
    from agent.plugins.snapshot import RuntimeSnapshotStore

logger = logging.getLogger("agent.loop")
StreamDelta: TypeAlias = dict[str, str] | str
StreamSink: TypeAlias = Callable[[StreamDelta], Awaitable[None]]
StreamSinkFactory: TypeAlias = Callable[[object], StreamSink | None]
StreamSupportPolicy: TypeAlias = Callable[[str], bool]
RuntimeSelector: TypeAlias = Literal["stable", "latest"]


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
    "akashic": _is_nonempty,
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
    return item.content


def _inbound_client_message_id(msg: InboundItem) -> str:
    """每轮只解析/验证一次的入站 client_message_id（非字符串 fail-loud）。"""

    # 1. 非入站消息恒为 missing；入站消息缺失字段也算 missing（非 mobile 合法）。
    if not isinstance(msg, InboundMessage):
        return "missing"
    raw = (msg.metadata or {}).get("client_message_id")
    if raw is None:
        return "missing"
    # 2. 字段存在但非字符串是内部合同错误，fail-loud 抛出。
    if not isinstance(raw, str):
        raise TypeError("client_message_id 必须是字符串")
    return raw or "missing"


def _inbound_execution_turn_id(msg: InboundItem) -> str:
    """解析普通 InboundMessage 的权威 execution turn id（入站信任边界）。"""

    # 1. 非入站消息恒为空白；缺失字段由 owner 生成一次。
    if not isinstance(msg, InboundMessage):
        return ""
    metadata = msg.metadata or {}
    # 2. 字段存在但非字符串是内部合同错误，fail-loud 抛出。
    for label, value in (
        ("_control_execution_turn_id", metadata.get("_control_execution_turn_id")),
        ("control_turn_id", metadata.get("control_turn_id")),
    ):
        if value is not None and not isinstance(value, str):
            raise TypeError(f"{label} 必须是字符串")
    # 3. execution ID 是本次实际 turn owner；control ID 只在 direct-call 显式
    #    映射（interaction 分组）时参与，不静默二选一，优先级固定 execution。
    return (
        metadata.get("_control_execution_turn_id")
        or metadata.get("control_turn_id")
        or ""
    )


def _disable_candidate_side_effect_tools(
    msg: InboundMessage,
    candidate_plugin_ids: frozenset[str],
    tools: ToolRegistry | None,
    snapshot: object,
) -> None:
    """把本次 candidate lease 的副作用工具加入 turn-local 禁用集合。"""
    if tools is None or not candidate_plugin_ids:
        return
    mcp_registry = cast(Any, snapshot).mcp_server_registry
    raw_disabled = msg.metadata.get("disabled_tools")
    if isinstance(raw_disabled, str):
        disabled = {raw_disabled} if raw_disabled else set()
    elif isinstance(raw_disabled, (list, tuple, set)):
        disabled = {str(name) for name in raw_disabled if str(name)}
    else:
        disabled = set()
    for plugin_id in candidate_plugin_ids:
        disabled |= tools.get_non_read_only_source_tool_names(
            "plugin",
            plugin_id,
        )
        server_names = (
            ()
            if mcp_registry is None
            else (
                descriptor.name
                for descriptor in mcp_registry.descriptors
                if descriptor.owner == plugin_id
            )
        )
        for server_name in server_names:
            disabled |= tools.get_non_read_only_source_tool_names(
                "mcp",
                server_name,
            )
    msg.metadata["disabled_tools"] = sorted(disabled)


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
        self._running = False
        self._processing_state = deps.processing_state
        self._event_bus = deps.event_bus or EventBus()
        self._session_lanes = SessionLaneRegistry()
        self._runtime_snapshot_store: RuntimeSnapshotStore | None = None
        self._plugin_rollout_fact_provider: Callable[[], str] | None = None
        self._outbound_port = deps.outbound_port

        # ── 中断控制面（纯内存态） ──
        self._active_tasks: dict[str, asyncio.Task[OutboundMessage]] = {}
        self._active_turn_states: dict[str, TurnInterruptState] = {}
        self._interrupt_states: dict[str, TurnInterruptState] = {}

        # 2. Core 暂时保留 Markdown profile；嵌入式记忆属于 Prompt 插件。
        markdown_memory = self._resolve_markdown_runtime(deps)
        self._tool_search_enabled = bool(config.llm.tool_search_enabled)
        if deps.context is not None:
            self._context = deps.context
        else:
            if markdown_memory is None:
                raise ValueError("AgentLoop context fallback requires Core Markdown runtime")
            self._context = ContextBuilder(
                deps.workspace,
                memory=markdown_memory.store,
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
        self._compaction_runtime: SessionCompactionRuntime | None = None

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

    def bind_plugin_rollout_fact_provider(
        self,
        provider: Callable[[], str],
    ) -> None:
        self._plugin_rollout_fact_provider = provider

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
                    turn_id=running_turn_id.get(),
                    content_delta=(
                        content_delta if isinstance(content_delta, str) else ""
                    ),
                    thinking_delta=(
                        thinking_delta if isinstance(thinking_delta, str) else ""
                    ),
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

    def _resolve_markdown_runtime(self, deps: AgentLoopDeps):
        if deps.memory_runtime is None:
            return None
        return deps.memory_runtime.markdown

    def _assemble_passive_runtime(
        self,
        *,
        deps: AgentLoopDeps,
        config: AgentLoopConfig,
    ) -> None:
        # 1. 先组基础 service ports。
        llm_svc = self._llm_services
        session_svc = self._session_services
        compaction_runtime = session_svc.compaction_runtime
        markdown_runtime = self._resolve_markdown_runtime(deps)
        if compaction_runtime is None and markdown_runtime is not None:
            compaction_runtime = SessionCompactionRuntime(
                session_manager=session_svc.session_manager,
                markdown=markdown_runtime.maintenance,
            )
        if isinstance(compaction_runtime, SessionCompactionRuntime):
            self._compaction_runtime = compaction_runtime
        # 2. 组执行层。
        self._tool_discovery = deps.tool_discovery or ToolDiscoveryState()
        self._reasoner = deps.reasoner or DefaultReasoner(
            llm=llm_svc,
            llm_config=config.llm,
            tools=deps.tools,
            discovery=self._tool_discovery,
            tool_search_enabled=self._tool_search_enabled,
            context=self._context,
            event_bus=self._event_bus,
            non_preloadable_names=deps.tools.get_non_preloadable_names,
            compaction_runtime=compaction_runtime,
            context_compaction=config.context_compaction,
        )

        # 3. 最后串 passive prepare / execute / commit 主链。
        passive_context_store = DefaultContextStore(
            context=self._context,
        )
        self._passive_pipeline = PassiveTurnPipeline(
            PassiveTurnDeps(
                session=session_svc,
                context_store=passive_context_store,
                context=self._context,
                tools=deps.tools,
                reasoner=self._reasoner,
                event_bus=self._event_bus,
                outbound_port=deps.outbound_port,
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
                if isinstance(item, InboundEnvelope):
                    await self.bus.release_channel_inbound(
                        item,
                        InboundOwner.LANE,
                    )
                    raise RuntimeError(
                        "v3 Channel inbound 必须由 PassiveMessageWorker 消费"
                    )
                await self._run_inbound_turn(item)
        finally:
            self._running = False

    async def _run_inbound_turn(self, item: InboundItem) -> None:
        """执行一个入站 turn，并在状态清理后确认消息。"""

        key = item.session_key
        ownership_established = False
        try:
            # 1. 入站信任边界先于一切 owner map 写入：解析/生成本轮权威
            #    execution turn id（InboundMessage 复用 metadata 或生成一次；
            #    Spawn 内部工作项同样生成），类型错误原样抛出、不污染 maps。
            execution_turn_id = _inbound_execution_turn_id(item) or new_turn_id()
            # 2. 边界通过后建立本轮中断状态和 child task；同一个 ID 传给
            #    child，保证 running_turn_id / TurnStarted / 正常或错误 final
            #    同源，禁止 child 另生成而 parent 不知道。
            self._active_turn_states[key] = self._build_initial_turn_state(item, key)
            task = asyncio.create_task(
                self._process_with_runtime_admission(
                    item,
                    execution_turn_id=execution_turn_id,
                ),
                name=f"agent-turn:{key}",
            )
            self._active_tasks[key] = task
            # 3. child task 已建立并登记为本轮 execution owner 后才拥有确认
            #    权；边界校验失败或 create_task 失败绝不 ACK（保留 durable
            #    handoff 供恢复），也不存在可观察结果被静默确认。
            ownership_established = True
            try:
                # 4. 只吞掉本轮取消；运行器取消必须继续向生命周期 owner 传播。
                await task
            except asyncio.CancelledError:
                current_task = asyncio.current_task()
                if current_task is not None and current_task.cancelling():
                    raise
                logger.info(f"Turn cancelled for {key}")
            except Exception as e:
                # 5. 错误 final 必须携带本轮权威 execution turn id，禁止留给
                #    channel 按当前 active turn fallback（迟到错误会归到别的
                #    active turn）。
                logger.error(f"处理消息出错: {e}", exc_info=True)
                outbound_port = getattr(self, "_outbound_port", None)
                if outbound_port is None:
                    raise RuntimeError(
                        "AgentLoop passive committed Channel outbound port 未绑定"
                    ) from e
                await outbound_port.dispatch(
                    OutboundDispatch(
                        channel=item.channel,
                        chat_id=item.chat_id,
                        content=f"出错：{e}",
                        control_turn_id=execution_turn_id,
                    )
                )
        finally:
            # 6. 统一收束：本轮已建立的内存 maps 总是清理；只有 execution
            #    owner 已建立（child task 成功创建并登记）才完成总线确认
            #    （释放 lane / mobile durable handoff），边界失败与 task 建立
            #    失败保留 durable handoff 供恢复，绝不静默 ACK poison message。
            _ = self._active_tasks.pop(key, None)
            _ = self._active_turn_states.pop(key, None)
            if ownership_established:
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

    async def shutdown_compaction(self) -> None:
        """取消并等待 AgentLoop 拥有的 Markdown compaction 任务。"""

        if self._compaction_runtime is not None:
            await self._compaction_runtime.shutdown()

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
        return TurnInterruptState(
            session_key=key,
            original_user_message=item.content,
            original_metadata=dict(item.metadata or {}),
        )

    async def _resume_interrupted_message(
        self,
        msg: InboundItem,
        key: str,
    ) -> tuple[InboundItem, bool]:
        # 1. 普通入站消息按 session 恢复尚未完成的 Turn。
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
            effects={"post_commit": PostCommitEffect.SUPPRESS.value},
        )
        await self.session_manager.append_messages(session, session.messages[start:])

    async def _observe_turn_started(
        self,
        msg: InboundItem,
        key: str,
        client_message_id: str,
    ) -> None:
        control_turn_id = running_turn_id.get()
        if isinstance(msg, InboundMessage):
            control_turn_id = str(
                msg.metadata.get("control_turn_id") or control_turn_id
            )

        # 1. 对外发布被动 turn 开始事件，具体副作用由 observer 决定。
        #    身份使用入站边界已解析的同一个 client_message_id，禁止再次解析。
        await self._event_bus.observe(
            TurnStarted(
                session_key=key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=_item_content(msg),
                timestamp=msg.timestamp,
                turn_id=running_turn_id.get(),
                control_turn_id=control_turn_id,
                client_message_id=client_message_id,
            )
        )

    # ── 被动 turn 处理 ────────────────────────────────────────────

    async def _react(
        self,
        msg: InboundItem,
        key: str,
        *,
        dispatch_outbound: bool = True,
        command_admitted: bool = False,
    ) -> OutboundMessage:
        """把一个输入交给被动链路，返回它生成的消息。"""

        return await self._passive_pipeline.run(
            msg,
            key,
            dispatch_outbound=dispatch_outbound,
            command_admitted=command_admitted,
        )

    async def _process(
        self,
        msg: InboundItem,
        session_key: str | None = None,
        busy_session_key: str | None = None,
        dispatch_outbound: bool = True,
        execution_turn_id: str | None = None,
    ) -> OutboundMessage:
        key = session_key or msg.session_key
        busy_key = busy_session_key or key
        # 1. 本轮权威 execution turn id 与 client_message_id 都先于任何
        #    contextvar/副作用解析：bus 路径由 owner 显式传入，direct-call
        #    路径由 metadata 形成（execution 恒为 owner）；类型错误 fail-loud
        #    且不泄漏 contextvar。
        inherited_turn_id = (
            execution_turn_id
            if execution_turn_id is not None
            else _inbound_execution_turn_id(msg)
        )
        client_message_id = _inbound_client_message_id(msg)
        # 2. 给本 turn task 打上 session 归属，供 observe 全局错误采集关联。
        session_token = current_session_key.set(key)
        turn_token = running_turn_id.set(inherited_turn_id or new_turn_id())
        client_message_token = current_client_message_id.set(client_message_id)
        try:
            # 3. 先投影插件发布事实，再冻结本 turn 的模型 generation。
            rollout_fact_provider = getattr(self, "_plugin_rollout_fact_provider", None)
            if (
                isinstance(msg, InboundMessage)
                and msg.channel != "programmatic"
                and rollout_fact_provider is not None
            ):
                fact = rollout_fact_provider()
                if fact:
                    msg.metadata["_plugin_rollout_fact"] = fact

            # 4. Committed commands settle before model, Session, resume, and TurnStarted.
            if isinstance(msg, InboundMessage):
                command_result = await self._passive_pipeline.run_command(
                    msg,
                    key,
                    dispatch_outbound=dispatch_outbound,
                )
                if command_result is not None:
                    return command_result
            model_selection = await self._resolve_model_selection(msg, key)
            async with model_execution_scope(
                self._llm_services.provider,
                model_selection.model_ref or None,
                model_selection.reasoning_effort,
            ) as model_binding:
                if model_binding is not None and isinstance(msg, InboundMessage):
                    msg.metadata["model_binding"] = model_binding.describe("agent")

                # 5. 处理可能存在的续跑态，并发布 turn started。
                msg, resumed_from_interrupt = await self._resume_interrupted_message(
                    msg, key
                )
                await self._observe_turn_started(msg, key, client_message_id)
                content = _item_content(msg)
                preview = content[:60] + "..." if len(content) > 60 else content
                logger.info(f"Processing message from {msg.channel}: {preview}")

                # 6. 再进入 busy 状态并执行核心处理。
                if self._processing_state:
                    self._processing_state.enter(busy_key)
                try:
                    outbound = await self._react(
                        msg,
                        key,
                        dispatch_outbound=dispatch_outbound,
                        command_admitted=isinstance(msg, InboundMessage),
                    )
                    if resumed_from_interrupt:
                        self._interrupt_states.pop(key, None)
                    return outbound
                finally:
                    if self._processing_state:
                        self._processing_state.exit(busy_key)
        finally:
            # 7. 当前 query 结束即回收其 shell，再恢复调用方上下文。
            try:
                await self._cleanup_shell_owner(key)
            finally:
                current_session_key.reset(session_token)
                running_turn_id.reset(turn_token)
                current_client_message_id.reset(client_message_token)

    async def _resolve_model_selection(
        self,
        msg: InboundItem,
        session_key: str,
    ) -> SessionModelSelection:
        """Validate, persist, and resolve one conversation's model selection."""

        if not isinstance(msg, InboundMessage):
            return SessionModelSelection()
        provider = self._llm_services.provider
        if not isinstance(provider, RoleBoundProvider):
            return SessionModelSelection()
        registry = provider.registry
        await registry.refresh()
        session = self.session_manager.get_or_create(session_key)

        # 1. A client-supplied field is an explicit session-setting operation.
        if "model_runtime_id" in msg.metadata:
            raw_runtime_id = msg.metadata["model_runtime_id"]
            if not isinstance(raw_runtime_id, str):
                raise TypeError("model_runtime_id 必须是字符串")
            runtime_id = raw_runtime_id.strip()
            raw_effort = msg.metadata.get("model_reasoning_effort", "")
            if not isinstance(raw_effort, str):
                raise TypeError("model_reasoning_effort 必须是字符串")
            effort = raw_effort.strip()
            if runtime_id:
                if not registry.has_runtime(runtime_id):
                    raise ValueError(f"模型 runtime 不存在: {runtime_id}")
            write_session_model_selection(
                session.metadata,
                SessionModelSelection(runtime_id, effort),
            )
            self.session_manager.save(session)

        # 2. Existing metadata is authoritative when this message follows it.
        selection = read_session_model_selection(session.metadata)
        if selection.model_ref and not registry.has_runtime(selection.model_ref):
            raise ValueError(f"session 引用不存在的模型 runtime: {selection.model_ref}")
        return selection

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
                    turn=running_turn_id.get(),
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
                    turn=running_turn_id.get(),
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
        runtime_selector: RuntimeSelector = "stable",
        execution_turn_id: str | None = None,
    ) -> OutboundMessage:
        key = session_key or msg.session_key
        lane_key = busy_session_key or key
        async with self._session_lanes.hold(lane_key):
            store = self._runtime_snapshot_store
            # 只有入站边界已确定的权威 ID 才显式传给 _process；直接调用/内部
            # 工作项为 None 时保持原有 metadata 派生语义（兼容外部直连 _process）。
            process_kwargs: dict[str, str] = {}
            if execution_turn_id is not None:
                process_kwargs["execution_turn_id"] = execution_turn_id
            from agent.plugins.snapshot import get_current_runtime_lease

            bound_lease = get_current_runtime_lease()
            if bound_lease is not None:
                snapshot = bound_lease.snapshot
                if bound_lease.validation_candidate_plugin_ids:
                    if not isinstance(msg, InboundMessage):
                        raise RuntimeError(
                            "latest candidate 只接受普通 inbound message"
                        )
                    _disable_candidate_side_effect_tools(
                        msg,
                        bound_lease.validation_candidate_plugin_ids,
                        snapshot.tool_registry,
                        snapshot,
                    )
                return await self._process(
                    msg,
                    session_key=session_key,
                    busy_session_key=busy_session_key,
                    dispatch_outbound=dispatch_outbound,
                    **process_kwargs,
                )
            if store is None or store.current is None:
                if runtime_selector != "stable":
                    raise RuntimeError("latest RuntimeSnapshot 不可用")
                return await self._process(
                    msg,
                    session_key=session_key,
                    busy_session_key=busy_session_key,
                    dispatch_outbound=dispatch_outbound,
                    **process_kwargs,
                )
            from agent.plugins.snapshot import (
                bind_runtime_snapshot,
                reset_runtime_snapshot,
            )

            lease = await store.acquire(selector=runtime_selector)
            async with lease as snapshot:
                token = bind_runtime_snapshot(lease)
                try:
                    if lease.validation_candidate_plugin_ids:
                        if not isinstance(msg, InboundMessage):
                            raise RuntimeError(
                                "latest candidate 只接受普通 inbound message"
                            )
                        _disable_candidate_side_effect_tools(
                            msg,
                            lease.validation_candidate_plugin_ids,
                            snapshot.tool_registry,
                            snapshot,
                        )
                    return await self._process(
                        msg,
                        session_key=session_key,
                        busy_session_key=busy_session_key,
                        dispatch_outbound=dispatch_outbound,
                        **process_kwargs,
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
        stream_events: bool = False,
        disabled_tools: list[str] | None = None,
        sender: str = "user",
        media: list[str] | None = None,
        turn_id: str = "",
        runtime_selector: RuntimeSelector = "stable",
    ) -> str:
        response = await self.process_direct_message(
            content,
            session_key=session_key,
            busy_session_key=busy_session_key,
            channel=channel,
            chat_id=chat_id,
            stream_events=stream_events,
            disabled_tools=disabled_tools,
            sender=sender,
            media=media,
            turn_id=turn_id,
            runtime_selector=runtime_selector,
        )
        return response.content

    async def process_direct_message(
        self,
        content: str,
        session_key: str = "programmatic:direct",
        busy_session_key: str | None = None,
        channel: str = "programmatic",
        chat_id: str = "direct",
        stream_events: bool = False,
        disabled_tools: list[str] | None = None,
        sender: str = "user",
        media: list[str] | None = None,
        metadata: dict[str, object] | None = None,
        turn_input_source: object | None = None,
        timestamp: datetime | None = None,
        turn_id: str = "",
        interaction_id: str = "",
        attempt_replay: list[dict[str, Any]] | None = None,
        prior_tool_chain: list[dict[str, Any]] | None = None,
        prior_input_count: int = 0,
        runtime_selector: RuntimeSelector = "stable",
    ) -> OutboundMessage:
        """执行直接消息，并按需隔离会话历史与持久化。"""

        inbound_metadata = dict(metadata or {})
        if turn_input_source is not None:
            inbound_metadata["_control_turn_input_source"] = turn_input_source
        if attempt_replay:
            inbound_metadata["_control_attempt_replay"] = list(attempt_replay)
        if prior_tool_chain:
            inbound_metadata["_control_prior_tool_chain"] = list(prior_tool_chain)
        if prior_input_count:
            inbound_metadata["_control_prior_input_count"] = prior_input_count
        if not stream_events:
            inbound_metadata["suppress_stream_events"] = True
        if disabled_tools:
            inbound_metadata["disabled_tools"] = list(disabled_tools)
        if turn_id:
            # 可信入口形成一致对：execution turn id 恒为本次 attempt 的 owner；
            # control_turn_id 是 interaction 分组 id（attempt 重试延续同一
            # logical interaction 时显式不同），只在 direct-call 显式映射中参与。
            inbound_metadata["control_turn_id"] = interaction_id or turn_id
            inbound_metadata["_control_execution_turn_id"] = turn_id
        msg = InboundMessage(
            channel=channel,
            sender=sender,
            chat_id=chat_id,
            content=content,
            media=list(media or []),
            metadata=inbound_metadata,
            timestamp=timestamp or datetime.now().astimezone(),
        )
        response = await self._process_with_runtime_admission(
            msg,
            session_key=session_key,
            busy_session_key=busy_session_key,
            dispatch_outbound=False,
            runtime_selector=runtime_selector,
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
        tools_used = result.tools_used
        tool_chain = result.tool_chain
        visible_names = result.visible_names
        return result.reply, tools_used, tool_chain, visible_names, result.thinking


# ── 模块级辅助 ────────────────────────────────────────────────────
