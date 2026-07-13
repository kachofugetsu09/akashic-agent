"""运行当前插件选择的主动生命周期，并按其调度结果等待下一轮。"""

from __future__ import annotations

import asyncio
import json
import logging
import random as _random_module
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from core.memory.engine import MemoryRetrievalApi
    from agent.plugins.snapshot import (
        RuntimeSnapshot,
        RuntimeSnapshotLease,
        RuntimeSnapshotStore,
    )

from core.error_context import current_session_key
from agent.looping.ports import SessionServices
from agent.core.proactive_kernel import ProactiveKernel
from agent.provider import LLMProvider
from agent.plugins.specs import RegisteredProactiveSource
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
        proactive_sources: list[RegisteredProactiveSource] | None = None,
        runtime_snapshot_store: RuntimeSnapshotStore | None = None,
        state_store_owned: bool = False,
    ) -> None:
        self._sessions = session_manager
        self._provider = provider
        self._push = push_tool
        self._cfg = config
        self._model = config.model or model
        self._max_tokens = max_tokens
        self._state_store_owned = state_store is None or state_store_owned
        self._state_closed = False
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
        self._plugin_proactive_sources = proactive_sources or []
        self._runtime_snapshot_store = runtime_snapshot_store
        self._active_snapshot_id: str | None = None
        self._kernel_started = False
        self._active_kernel_lease: RuntimeSnapshotLease | None = None
        self._workspace_context_mtime_ns: int | None = None
        self._workspace_context_text: str = ""
        self._init_runtime_state(config)
        self._init_runtime_components()

    def _init_runtime_state(self, config: ProactiveConfig) -> None:
        self._running = False
        self._wake = asyncio.Event()
        self._reload_lock = asyncio.Lock()
        self._stopped = asyncio.Event()
        self._stopped.set()

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
            mcp_gateway=self._mcp_gateway,
            proactive_sources=self._plugin_proactive_sources,
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

    def _build_mcp_gateway(self) -> SharedMcpGateway:
        from agent.plugins.snapshot import get_current_runtime_snapshot

        snapshot = get_current_runtime_snapshot()
        tools = (
            snapshot.tool_registry
            if snapshot is not None and snapshot.tool_registry is not None
            else self._shared_tools
        )

        return SharedMcpGateway(
            Path(self._sessions.workspace),
            tools,
        )

    def _build_kernel(self) -> ProactiveKernel:
        runtime = self._build_plugin_runtime()
        modules = [
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
        # 3. 构建发送编排器、传感器、MCP 网关和主动链路 kernel。
        self._turn_orchestrator = self._build_turn_orchestrator()
        self._sense = self._build_sense()
        self._mcp_gateway: SharedMcpGateway
        self._proactive_kernel: ProactiveKernel
        self._scheduler = ProactiveScheduler(
            cfg=self._cfg,
            presence=self._presence,
            rng=self._rng,
            target_session_key_fn=self._target_session_key,
            trace_fn=self._trace_proactive_rate_decision,
        )
        if self._runtime_snapshot_store is None:
            self._mcp_gateway = self._build_mcp_gateway()
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
        self._stopped.clear()
        logger.info(
            "Proactive runtime 已启动 lifecycle=%s target=%s:%s",
            self._cfg.lifecycle,
            self._cfg.default_channel,
            self._cfg.default_chat_id,
        )
        try:
            if self._runtime_snapshot_store is not None:
                await self._start_current_snapshot()
            else:
                await self._proactive_kernel.start()
                self._kernel_started = True
            await self._run_loop()
        finally:
            stop_error: BaseException | None = None
            try:
                await self._stop_active_kernel()
            except BaseException as exc:
                stop_error = exc
            try:
                self.close()
            except BaseException as close_error:
                if stop_error is None:
                    raise
                raise stop_error from close_error
            finally:
                self._stopped.set()
            if stop_error is not None:
                raise stop_error

    async def _run_loop(self) -> None:
        last_base_score: float | None = None
        next_interval: int | None = 0
        while self._running:
            interval = (
                next_interval
                if next_interval is not None
                else self._next_interval(last_base_score)
            )
            if interval > 0:
                logger.info("[proactive] 下次运行兜底间隔=%ds", interval)
                self._wake.clear()
                try:
                    _ = await asyncio.wait_for(self._wake.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass
            if not self._running:
                return
            try:
                last_base_score = await self._tick()
                result = self._proactive_kernel.last_result
                next_interval = (
                    result.next_interval_seconds
                    if result is not None
                    else None
                )
            except Exception:
                logger.exception("[proactive] runtime tick 异常")
                last_base_score = None
                next_interval = None

    def _next_interval(self, base_score: float | None = None) -> int:
        return self._scheduler.next_interval(base_score)

    def _target_session_key(self) -> str:
        return self._sense.target_session_key()

    def stop(self) -> None:
        self._running = False
        self._wake.set()

    def close(self) -> None:
        """关闭由主动循环负责的 SQLite 状态存储。"""

        # 1. 仅关闭明确归主动循环所有的资源，外部注入的 store 由调用方负责。
        if not self._state_store_owned or self._state_closed:
            return
        self._state.close()
        self._state_closed = True

    async def wait_stopped(self) -> None:
        _ = await self._stopped.wait()

    # ── 内部方法 ──────────────────────────────────────────────────

    async def _tick(self) -> float | None:
        """执行一次 proactive v2 tick。"""
        if self._runtime_snapshot_store is None:
            async with self._reload_lock:
                return await self._tick_bound()
        lease = await self._runtime_snapshot_store.acquire()
        from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot

        async with lease:
            token = bind_runtime_snapshot(lease)
            try:
                async with self._reload_lock:
                    await self._switch_snapshot(lease.snapshot)
                    return await self._tick_bound()
            finally:
                reset_runtime_snapshot(token)

    async def _tick_admitted(self) -> float | None:
        if self._runtime_snapshot_store is not None:
            lease = await self._runtime_snapshot_store.acquire()
            from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot

            async with lease:
                token = bind_runtime_snapshot(lease)
                try:
                    await self._switch_snapshot(lease.snapshot)
                    return await self._tick_bound()
                finally:
                    reset_runtime_snapshot(token)
        return await self._tick_bound()

    async def quiesce_for_reload(self) -> None:
        async with self._reload_lock:
            await self._stop_active_kernel()

    async def resume_after_reload(self) -> None:
        self._wake.set()

    async def _tick_bound(self) -> float | None:
        session_key = self._target_session_key()
        session_token = current_session_key.set(session_key)
        try:
            # 1. 执行 Gate → Fetch → Judge → Resolve → Deliver 全链路。
            started = time.perf_counter()
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
        finally:
            # 2. 恢复父 task 上下文，避免跨 tick 残留会话归属。
            current_session_key.reset(session_token)

    async def _start_current_snapshot(self) -> None:
        assert self._runtime_snapshot_store is not None
        lease = await self._runtime_snapshot_store.acquire()
        from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot

        async with lease:
            token = bind_runtime_snapshot(lease)
            try:
                await self._switch_snapshot(lease.snapshot)
            finally:
                reset_runtime_snapshot(token)

    async def _switch_snapshot(self, snapshot: RuntimeSnapshot) -> None:
        if self._active_snapshot_id == snapshot.snapshot_id and self._kernel_started:
            return
        from agent.plugins.snapshot import get_current_runtime_lease

        source_lease = get_current_runtime_lease()
        if source_lease is None or source_lease.snapshot is not snapshot:
            raise RuntimeError("主动 kernel 切换缺少目标 Snapshot lease")
        old_kernel = self._proactive_kernel if self._kernel_started else None
        old_lease = self._active_kernel_lease
        if old_kernel is not None and old_lease is not None:
            await self._run_with_lease(old_lease, old_kernel.stop)
            self._kernel_started = False
        candidate_lease = source_lease.fork()
        try:
            candidate_kernel = await self._build_and_start_kernel(
                snapshot,
                candidate_lease,
            )
        except BaseException:
            await candidate_lease.release()
            if old_kernel is not None and old_lease is not None:
                await self._run_with_lease(old_lease, old_kernel.start)
                self._kernel_started = True
            raise
        if old_lease is not None:
            await old_lease.release()
        self._proactive_kernel = candidate_kernel
        self._active_kernel_lease = candidate_lease
        self._active_snapshot_id = snapshot.snapshot_id
        self._kernel_started = True

    async def _build_and_start_kernel(
        self,
        snapshot: RuntimeSnapshot,
        lease: RuntimeSnapshotLease,
    ) -> ProactiveKernel:
        async def build_and_start() -> ProactiveKernel:
            self._apply_snapshot(snapshot)
            self._mcp_gateway = self._build_mcp_gateway()
            kernel = self._build_kernel()
            await kernel.start()
            return kernel

        return await self._run_with_lease(lease, build_and_start)

    async def _stop_active_kernel(self) -> None:
        lease = self._active_kernel_lease
        stopped = not self._kernel_started
        try:
            if self._kernel_started:
                if lease is None:
                    await self._proactive_kernel.stop()
                else:
                    await self._run_with_lease(lease, self._proactive_kernel.stop)
                self._kernel_started = False
                stopped = True
        finally:
            if stopped:
                self._active_snapshot_id = None
                self._active_kernel_lease = None
            if stopped and lease is not None:
                await lease.release()

    async def _run_with_lease(
        self,
        lease: RuntimeSnapshotLease,
        operation: Callable[[], Awaitable[Any]],
    ) -> Any:
        from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot

        token = bind_runtime_snapshot(lease)
        try:
            return await operation()
        finally:
            reset_runtime_snapshot(token)

    def _apply_snapshot(self, snapshot: RuntimeSnapshot) -> None:
        self._plugin_proactive_modules = list(snapshot.proactive_modules)
        self._plugin_proactive_lifecycles = list(snapshot.proactive_lifecycles)
        self._plugin_proactive_module_factories = list(
            snapshot.proactive_module_factories
        )
        self._plugin_proactive_runtime_factories = list(
            snapshot.proactive_runtime_factories
        )
        self._plugin_proactive_sources = list(snapshot.proactive_sources.values())
        self._tool_hooks = list(snapshot.tool_hooks)


def build_proactive_loop(**kwargs: Any) -> ProactiveLoop:
    return ProactiveLoop(**kwargs)
