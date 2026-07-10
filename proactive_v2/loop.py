"""
ProactiveLoop - 主动触达核心循环。

独立于 AgentLoop,定期:
  1. 拉取所有内容源的最新候选事件
  2. 获取用户最近聊天上下文
  3. 调用 LLM 反思:有没有值得主动说的
  4. 产出 TurnResult 并由统一 OutboundPort 发送消息
"""

from __future__ import annotations

import asyncio
import json
import logging
import random as _random_module
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from core.memory.engine import MemoryRetrievalApi

from core.error_context import current_session_key
from agent.looping.ports import SessionServices
from agent.core.proactive_kernel import ProactiveKernel
from agent.provider import LLMProvider
from agent.tool_hooks import ToolHook
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from agent.turns.outbound import PushToolOutboundPort
from agent.turns.orchestrator import TurnOrchestrator, TurnOrchestratorDeps
from bus.event_bus import EventBus
from core.common.strategy_trace import build_strategy_trace_envelope
from core.common.diagnostic_log import diagnostic_context, diagnostic_line
from proactive_v2.config import ProactiveConfig
from proactive_v2.lifecycle import ProactiveLifecycleSpec
from proactive_v2.mcp_sources import SharedMcpGateway
from proactive_v2.modules_schedule import ProactiveScheduler
from proactive_v2.modules_source import McpRuntimeModule
from proactive_v2.presence import PresenceStore
from proactive_v2.runtime_scope import ProactiveRuntimeScope
from proactive_v2.sensor import Sensor
from proactive_v2.state import ProactiveStateStore
from session.manager import SessionManager

logger = logging.getLogger(__name__)


class ProactiveLoop:
    _PROACTIVE_CONTEXT_FILE = "PROACTIVE_CONTEXT.md"
    _PROACTIVE_CONTEXT_TEMPLATE = """# Proactive Context

在这里写用户当前对主动推送的明确要求和规则。

- 主 agent 负责维护这份文件。
- proactive agent 每轮都会读取它,并把它视为需要遵守的规则,不是普通参考建议。
- 这里适合写白名单、黑名单、过滤条件、优先级、必须先验证的步骤。
- 这里不提供新闻事实,不提供候选内容,只定义规则。
- 写结论即可,不要写冗长过程。
"""

    def __init__(
        self,
        session_manager: SessionManager,
        provider: LLMProvider,
        push_tool: MessagePushTool,
        config: ProactiveConfig,
        model: str,
        max_tokens: int = 1024,
        state_store: ProactiveStateStore | None = None,
        state_path: Path | None = None,
        memory_store: "MemoryRetrievalApi | None" = None,
        presence: PresenceStore | None = None,
        rng: _random_module.Random | None = None,
        passive_busy_fn: Callable[[str], bool] | None = None,
        shared_tools: ToolRegistry | None = None,
        event_bus: EventBus | None = None,
        tool_hooks: list[ToolHook] | None = None,
        proactive_modules: list[object] | None = None,
        proactive_lifecycles: list[object] | None = None,
        proactive_module_factories: list[object] | None = None,
        proactive_runtime_factories: list[object] | None = None,
    ) -> None:
        self._sessions = session_manager
        self._provider = provider
        self._push = push_tool
        self._cfg = config
        self._model = config.model or model
        self._max_tokens = max_tokens
        self._state = self._build_state_store(state_store, state_path)
        self._memory = memory_store
        self._presence = presence
        self._rng = rng
        self._passive_busy_fn = passive_busy_fn
        self._shared_tools = shared_tools
        self._event_bus = event_bus
        self._tool_hooks = tool_hooks or []
        self._plugin_proactive_modules = proactive_modules or []
        self._plugin_proactive_lifecycles = proactive_lifecycles or []
        self._plugin_proactive_module_factories = proactive_module_factories or []
        self._plugin_proactive_runtime_factories = proactive_runtime_factories or []
        self._workspace_context_mtime_ns: int | None = None
        self._workspace_context_text: str = ""
        self._init_runtime_state(config)
        self._init_runtime_components()

    def _init_runtime_state(self, config: ProactiveConfig) -> None:
        self._running = False

    def _build_state_store(
        self,
        state_store: ProactiveStateStore | None,
        state_path: Path | None,
    ) -> ProactiveStateStore:
        if state_store is not None:
            return state_store
        return ProactiveStateStore(state_path or Path("proactive.db"))

    def _build_turn_orchestrator(self) -> TurnOrchestrator:
        return TurnOrchestrator(
            TurnOrchestratorDeps(
                session=SessionServices(
                    session_manager=self._sessions,
                    presence=self._presence,
                ),
                outbound=PushToolOutboundPort(self._push),
            )
        )

    def _build_sense(self) -> Sensor:
        return Sensor(
            cfg=self._cfg,
            sessions=self._sessions,
            presence=self._presence,
        )

    def _build_runtime_scope(self) -> ProactiveRuntimeScope:
        return ProactiveRuntimeScope(
            cfg=self._cfg,
            sense=self._sense,
            presence=self._presence,
            provider=self._provider,
            model=self._model,
            max_tokens=self._max_tokens,
            memory=self._memory,
            state_store=self._state,
            any_action_gate=None,
            passive_busy_fn=self._passive_busy_fn,
            turn_orchestrator=self._turn_orchestrator,
            deduper=None,
            rng=self._rng,
            workspace_context_fn=self._read_workspace_proactive_context,
            shared_tools=self._shared_tools,
            event_bus=self._event_bus,
            mcp_gateway=self._mcp_runtime.pool,
            tool_hooks=self._tool_hooks,
            schedule_fn=self._scheduler.next_interval,
        )

    def _build_plugin_runtime(self) -> object:
        selected = [
            factory
            for factory in self._plugin_proactive_runtime_factories
            if getattr(factory, "lifecycle_id", None) == self._cfg.lifecycle
        ]
        if len(selected) != 1:
            raise RuntimeError(
                f"主动 Runtime provider 数量错误: {self._cfg.lifecycle}={len(selected)}"
            )
        factory = selected[0]
        if not callable(factory):
            raise RuntimeError("插件 proactive_runtime_factories 返回了不可调用对象")
        return factory(self._build_runtime_scope())

    def _build_mcp_runtime(self) -> McpRuntimeModule:
        gateway = SharedMcpGateway(
            Path(self._sessions.workspace),
            self._shared_tools,
        )
        return McpRuntimeModule(
            cfg=self._cfg,
            gateway=gateway,
        )

    def _build_kernel(self) -> ProactiveKernel:
        runtime = self._build_plugin_runtime()
        modules = [
            self._mcp_runtime,
            *self._plugin_proactive_modules,
            *self._build_plugin_flow_modules(runtime),
        ]
        kernel = ProactiveKernel(
            modules,
            initial_slots_fn=self._build_initial_slots,
            lifecycle=self._select_lifecycle(),
        )
        logger.info("[proactive] phase graph:\n%s", kernel.inspect())
        return kernel

    def _build_plugin_flow_modules(
        self,
        runtime: object,
    ) -> list[object]:
        if not self._plugin_proactive_module_factories:
            raise RuntimeError("主动 Lifecycle 缺少 Module provider")
        modules: list[object] = []
        factories = [
            factory
            for factory in self._plugin_proactive_module_factories
            if getattr(factory, "lifecycle_id", None) == self._cfg.lifecycle
        ]
        if not factories:
            raise RuntimeError(f"主动 Lifecycle 缺少 Module provider: {self._cfg.lifecycle}")
        for factory in factories:
            if not callable(factory):
                raise RuntimeError("插件 proactive_module_factories 返回了不可调用对象")
            provided = factory(runtime)
            if not isinstance(provided, list):
                raise RuntimeError("主动 Module factory 必须返回 list")
            modules.extend(provided)
        return modules

    def _select_lifecycle(self) -> ProactiveLifecycleSpec:
        selected: list[ProactiveLifecycleSpec] = []
        for candidate in self._plugin_proactive_lifecycles:
            if not isinstance(candidate, ProactiveLifecycleSpec):
                raise RuntimeError(
                    "插件 proactive_lifecycles 返回值不是 ProactiveLifecycleSpec"
                )
            if candidate.id == self._cfg.lifecycle:
                selected.append(candidate)
        if len(selected) > 1:
            raise RuntimeError(f"主动 Lifecycle provider 冲突: {self._cfg.lifecycle}")
        if selected:
            return selected[0]
        raise RuntimeError(f"主动 Lifecycle 不存在: {self._cfg.lifecycle}")

    def _build_initial_slots(self, session_key: str) -> dict[str, Any]:
        last_user_at = (
            self._presence.get_last_user_at(session_key)
            if self._presence is not None
            else None
        )
        return {
            "proactive:cfg": self._cfg,
            "proactive:session_key": session_key,
            "proactive:started_at": datetime.now(timezone.utc),
            "proactive:last_user_at": last_user_at,
            "proactive:base_judge_send_threshold": self._cfg.judge_send_threshold,
        }

    def _init_runtime_components(self) -> None:
        # 1. 准备主动规则面板文件（PROACTIVE_CONTEXT.md）。
        self._ensure_workspace_proactive_context_file()
        # 2. 预读规则面板内容并做缓存。
        self._read_workspace_proactive_context()
        # 3. 构建发送编排器、传感器、MCP runtime 和主动链路 kernel。
        self._turn_orchestrator = self._build_turn_orchestrator()
        self._sense = self._build_sense()
        self._mcp_runtime = self._build_mcp_runtime()
        self._scheduler = ProactiveScheduler(
            cfg=self._cfg,
            presence=self._presence,
            rng=self._rng,
            target_session_key_fn=self._target_session_key,
            trace_fn=self._trace_proactive_rate_decision,
        )
        self._proactive_kernel = self._build_kernel()
        # 4. 启动时把当前 proactive 配置落一份 trace，方便回看。
        self._trace_proactive_config_snapshot()

    def _workspace_proactive_context_path(self) -> Path:
        return Path(self._sessions.workspace) / self._PROACTIVE_CONTEXT_FILE

    def _ensure_workspace_proactive_context_file(self) -> None:
        path = self._workspace_proactive_context_path()
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._PROACTIVE_CONTEXT_TEMPLATE, encoding="utf-8")

    def _read_workspace_proactive_context(self) -> str:
        path = self._workspace_proactive_context_path()
        self._ensure_workspace_proactive_context_file()
        try:
            stat = path.stat()
            mtime_ns = int(stat.st_mtime_ns)
            if self._workspace_context_mtime_ns == mtime_ns:
                return self._workspace_context_text
            text = path.read_text(encoding="utf-8").strip()
            self._workspace_context_mtime_ns = mtime_ns
            self._workspace_context_text = text
            return text
        except Exception as e:
            logger.warning("[proactive] 读取 workspace proactive context 失败: %s", e)
            return self._workspace_context_text

    def _trace_proactive_config_snapshot(self) -> None:
        payload = {
            "enabled": self._cfg.enabled,
            "tick_interval_s0": self._cfg.tick_interval_s0,
            "tick_interval_s1": self._cfg.tick_interval_s1,
            "tick_jitter": self._cfg.tick_jitter,
            "anyaction_enabled": self._cfg.anyaction_enabled,
            "anyaction_min_interval_seconds": self._cfg.anyaction_min_interval_seconds,
            "anyaction_probability_min": self._cfg.anyaction_probability_min,
            "anyaction_probability_max": self._cfg.anyaction_probability_max,
        }
        self._append_trace_line("proactive_config_trace.jsonl", payload)

    def _trace_proactive_rate_decision(
        self,
        *,
        base_score: float | None,
        interval: int,
        mode: str,
    ) -> None:
        self._append_trace_line(
            "proactive_rate_trace.jsonl",
            {
                "mode": mode,
                "base_score": round(base_score, 4) if base_score is not None else None,
                "interval_seconds": int(interval),
                "tick_interval_s0": self._cfg.tick_interval_s0,
                "tick_interval_s1": self._cfg.tick_interval_s1,
                "tick_jitter": self._cfg.tick_jitter,
            },
        )

    def _append_trace_line(self, filename: str, payload: dict[str, Any]) -> None:
        try:
            memory_dir = self._sessions.workspace / "memory"
            memory_dir.mkdir(parents=True, exist_ok=True)
            trace_file = memory_dir / filename
            if "trace_type" not in payload or "payload" not in payload:
                trace_type = "proactive_config" if "config" in filename else "proactive_rate"
                source = "proactive.config" if trace_type == "proactive_config" else "proactive.rate"
                payload = {
                    **build_strategy_trace_envelope(
                        trace_type=trace_type,  # type: ignore[arg-type]
                        source=source,
                        subject_kind="global",
                        subject_id=filename.removesuffix(".jsonl"),
                        payload=payload,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                    **payload,
                }
            with trace_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("[proactive] write trace failed %s: %s", filename, exc)

    async def run(self) -> None:
        self._running = True
        logger.info(
            f"ProactiveLoop 已启动  "
            f"目标={self._cfg.default_channel}:{self._cfg.default_chat_id}"
        )
        await self._proactive_kernel.start()
        try:
            await self._run_loop()
        finally:
            await self._proactive_kernel.stop()

    async def _run_loop(self) -> None:
        last_base_score: float | None = None
        next_interval: int | None = None
        while self._running:
            interval = (
                next_interval
                if next_interval is not None
                else self._next_interval(last_base_score)
            )
            logger.info("[proactive] 下次 tick 间隔=%ds", interval)
            await asyncio.sleep(interval)
            try:
                last_base_score = await self._tick()
                result = self._proactive_kernel.last_result
                next_interval = (
                    result.next_interval_seconds
                    if result is not None
                    else None
                )
            except Exception:
                logger.exception("ProactiveLoop tick 异常")
                last_base_score = None
                next_interval = None

    def _next_interval(self, base_score: float | None = None) -> int:
        return self._scheduler.next_interval(base_score)

    def _target_session_key(self) -> str:
        return self._sense.target_session_key()

    def stop(self) -> None:
        self._running = False

    # ── internal ──────────────────────────────────────────────────

    async def _tick(self) -> float | None:
        """执行一次 proactive v2 tick。"""
        # 给本 tick 打上 session 归属，供 observe 全局错误采集关联；
        # 纯埋点，依赖未就绪时静默跳过，绝不影响 tick 主流程。
        _ = current_session_key.set(self._target_session_key())
        # 主动回复全链路入口：Gate → Fetch → Judge → Resolve → Deliver。
        started = time.perf_counter()
        session_key = self._target_session_key()
        with diagnostic_context(session=session_key, flow="proactive", phase="tick"):
            logger.info(
                diagnostic_line(
                    "ProactiveLoop._tick",
                    event="start",
                    flow="proactive",
                    phase="tick",
                    session=session_key,
                    action="run",
                )
            )
            try:
                score = await self._proactive_kernel.run_tick(session_key)
            except Exception as exc:
                logger.exception(
                    diagnostic_line(
                        "ProactiveLoop._tick",
                        event="phase_error",
                        flow="proactive",
                        phase="tick",
                        session=session_key,
                        action="fail",
                        reason="proactive_tick_error",
                        duration_ms=int((time.perf_counter() - started) * 1000),
                        error_type=type(exc).__name__,
                        note=str(exc)[:160],
                    )
                )
                raise
            logger.info(
                diagnostic_line(
                    "ProactiveLoop._tick",
                    event="end",
                    flow="proactive",
                    phase="tick",
                    session=session_key,
                    action="done",
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
            )
            return score


def build_proactive_loop(**kwargs: Any) -> ProactiveLoop:
    return ProactiveLoop(**kwargs)
