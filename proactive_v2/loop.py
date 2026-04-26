"""
ProactiveLoop — 主动触达核心循环。

独立于 AgentLoop，定期：
  1. 拉取所有内容源的最新候选事件
  2. 获取用户最近聊天上下文
  3. 调用 LLM 反思：有没有值得主动说的
  4. 产出 TurnResult 并由统一 OutboundPort 发送消息
"""

from __future__ import annotations

import asyncio
import json
import logging
import random as _random_module
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from core.memory.runtime_facade import MemoryRuntimeFacade

from agent.looping.ports import ObservabilityServices, SessionServices
from agent.provider import LLMProvider
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from agent.turns.outbound import PushToolOutboundPort
from agent.turns.orchestrator import TurnOrchestrator, TurnOrchestratorDeps
from core.common.strategy_trace import build_strategy_trace_envelope
from proactive_v2.anyaction import AnyActionGate, QuotaStore
from proactive_v2.energy import (
    compute_energy,
    d_energy,
    next_tick_from_score,
)
from proactive_v2.judge import MessageDeduper
from proactive_v2.config import ProactiveConfig
from proactive_v2.memory_sampler import sample_memory_chunks
from proactive_v2.presence import PresenceStore
from proactive_v2.sensor import Sensor
from proactive_v2.state import ProactiveStateStore
from session.manager import SessionManager

logger = logging.getLogger(__name__)


class ProactiveLoop:
    _PROACTIVE_CONTEXT_FILE = "PROACTIVE_CONTEXT.md"
    _PROACTIVE_CONTEXT_TEMPLATE = """# Proactive Context

在这里写用户当前对主动推送的明确要求和规则。

- 主 agent 负责维护这份文件。
- proactive agent 每轮都会读取它，并把它视为需要遵守的规则，不是普通参考建议。
- 这里适合写白名单、黑名单、过滤条件、优先级、必须先验证的步骤。
- 这里不提供新闻事实，不提供候选内容，只定义规则。
- 写结论即可，不要写冗长过程。
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
        memory_store: "MemoryRuntimeFacade | None" = None,
        presence: PresenceStore | None = None,
        rng: _random_module.Random | None = None,
        light_provider: LLMProvider | None = None,
        light_model: str = "",
        passive_busy_fn: Callable[[str], bool] | None = None,
        observe_writer=None,
        shared_tools: ToolRegistry | None = None,
        fitbit_enabled: bool = False,
        fitbit_url: str = "http://127.0.0.1:18765",
        fitbit_poll_interval: int = 300,
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
        self._light_provider = light_provider or provider
        self._light_model = light_model or (config.model or model)
        self._observe_writer = observe_writer
        self._passive_busy_fn = passive_busy_fn
        self._shared_tools = shared_tools
        self._fitbit_enabled = bool(fitbit_enabled)
        self._fitbit_url = str(fitbit_url or "http://127.0.0.1:18765")
        self._fitbit_poll_interval = max(1, int(fitbit_poll_interval))
        self._workspace_context_mtime_ns: int | None = None
        self._workspace_context_text: str = ""
        self._init_runtime_state(config)
        self._init_runtime_components()

    def _init_runtime_state(self, config: ProactiveConfig) -> None:
        from proactive_v2.mcp_sources import McpClientPool
        self._running = False
        self._feed_poll_lock = asyncio.Lock()
        workspace = getattr(self._sessions, "workspace", None)
        self._mcp_pool = McpClientPool(Path(workspace) if workspace else None)

    def _build_state_store(
        self,
        state_store: ProactiveStateStore | None,
        state_path: Path | None,
    ) -> ProactiveStateStore:
        if state_store is not None:
            return state_store
        return ProactiveStateStore(state_path or Path("proactive.db"))

    def _build_fitbit_provider(self):
        if not self._fitbit_enabled:
            return None
        from proactive_v2.fitbit_sleep import FitbitSleepProvider

        return FitbitSleepProvider(
            url=self._fitbit_url,
            poll_interval=self._fitbit_poll_interval,
            sleeping_modifier=self._cfg.sleep_modifier_sleeping,
        )

    def _build_turn_orchestrator(self) -> TurnOrchestrator:
        return TurnOrchestrator(
            TurnOrchestratorDeps(
                session=SessionServices(
                    session_manager=self._sessions,
                    presence=self._presence,
                ),
                trace=ObservabilityServices(
                    workspace=self._sessions.workspace,
                    observe_writer=self._observe_writer,
                ),
                outbound=PushToolOutboundPort(self._push),
            )
        )

    def _build_anyaction_gate(self) -> AnyActionGate:
        quota_path = Path(self._state.workspace_dir) / "proactive_quota.json"
        return AnyActionGate(
            cfg=self._cfg,
            quota_store=QuotaStore(quota_path),
            rng=self._rng,
        )

    def _build_sense(self, fitbit_provider) -> Sensor:
        return Sensor(
            cfg=self._cfg,
            sessions=self._sessions,
            state=self._state,
            memory=self._memory,
            presence=self._presence,
            rng=self._rng,
            fitbit=fitbit_provider,
        )

    def _build_agent_tick(self):
        from proactive_v2.agent_tick_factory import AgentTickDeps, AgentTickFactory

        # 1. 把 loop 级公共依赖收束成 AgentTickDeps。
        # 2. 交给 factory 组装出“单次 tick 执行器”。
        return AgentTickFactory(
            AgentTickDeps(
                cfg=self._cfg,
                sense=self._sense,
                presence=self._presence,
                provider=self._provider,
                model=self._model,
                max_tokens=self._max_tokens,
                memory=self._memory,
                state_store=self._state,
                any_action_gate=self._anyaction,
                passive_busy_fn=self._passive_busy_fn,
                turn_orchestrator=self._turn_orchestrator,
                deduper=self._message_deduper,
                rng=self._rng,
                workspace_context_fn=self._read_workspace_proactive_context,
                observe_writer=self._observe_writer,
                shared_tools=self._shared_tools,
                pool=self._mcp_pool,
            )
        ).build()

    def _build_message_deduper(self) -> MessageDeduper | None:
        if not self._cfg.message_dedupe_enabled:
            return None
        return MessageDeduper(
            provider=self._provider,
            model=self._model,
            max_tokens=self._max_tokens,
        )

    def _init_runtime_components(self) -> None:
        # 1. 准备主动规则面板文件（PROACTIVE_CONTEXT.md）。
        self._ensure_workspace_proactive_context_file()
        # 2. 预读规则面板内容并做缓存。
        self._read_workspace_proactive_context()
        # 3. 构建发送编排器、前置 gate、传感器、去重器和单次 tick 执行器。
        self._turn_orchestrator = self._build_turn_orchestrator()
        self._anyaction = self._build_anyaction_gate()
        self._sense = self._build_sense(self._build_fitbit_provider())
        self._message_deduper = self._build_message_deduper()
        self._agent_tick = self._build_agent_tick()
        # 4. 启动时把当前 proactive 配置落一份 trace，方便回看。
        self._trace_proactive_config_snapshot()

    def _workspace_proactive_context_path(self) -> Path | None:
        workspace = getattr(self._sessions, "workspace", None)
        if workspace is None:
            return None
        return Path(workspace) / self._PROACTIVE_CONTEXT_FILE

    def _ensure_workspace_proactive_context_file(self) -> None:
        path = self._workspace_proactive_context_path()
        if path is None or path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._PROACTIVE_CONTEXT_TEMPLATE, encoding="utf-8")

    def _read_workspace_proactive_context(self) -> str:
        path = self._workspace_proactive_context_path()
        if path is None:
            return ""
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
            "threshold": self._cfg.threshold,
            "score_llm_threshold": self._cfg.score_llm_threshold,
            "tick_interval_s0": self._cfg.tick_interval_s0,
            "tick_interval_s1": self._cfg.tick_interval_s1,
            "tick_interval_s2": self._cfg.tick_interval_s2,
            "tick_interval_s3": self._cfg.tick_interval_s3,
            "tick_jitter": self._cfg.tick_jitter,
            "anyaction_enabled": self._cfg.anyaction_enabled,
            "anyaction_min_interval_seconds": self._cfg.anyaction_min_interval_seconds,
            "anyaction_probability_min": self._cfg.anyaction_probability_min,
            "anyaction_probability_max": self._cfg.anyaction_probability_max,
            "memory_history_gate_enabled": self._cfg.memory_history_gate_enabled,
            "sleep_modifier_sleeping": self._cfg.sleep_modifier_sleeping,
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
                "threshold": self._cfg.threshold,
                "score_llm_threshold": self._cfg.score_llm_threshold,
                "tick_interval_s0": self._cfg.tick_interval_s0,
                "tick_interval_s1": self._cfg.tick_interval_s1,
                "tick_interval_s2": self._cfg.tick_interval_s2,
                "tick_interval_s3": self._cfg.tick_interval_s3,
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

    async def _poll_feeds_once(self) -> None:
        """执行一次 feed 轮询，加锁保证不并发。
        MCP tool 层已将系统级失败序列化为 "error: ..." 字符串返回，
        此处统一检测并 warning 记录，不阻断 loop 主流程。
        """
        if self._feed_poll_lock.locked():
            logger.debug("[proactive] feed poll 仍在进行，跳过本次")
            return
        async with self._feed_poll_lock:
            try:
                from proactive_v2 import mcp_sources
                await mcp_sources.poll_content_feeds_async(self._mcp_pool)
                logger.info("[proactive] feed poll 完成")
            except Exception as e:
                logger.warning("[proactive] feed poll 系统级失败: %s", e)

    async def _poll_loop(self) -> None:
        """每配置间隔秒周期性触发 feed 轮询。"""
        while self._running:
            await asyncio.sleep(max(1, int(self._cfg.feed_poller_interval_seconds)))
            if not self._running:
                break
            await self._poll_feeds_once()

    async def run(self) -> None:
        self._running = True
        logger.info(
            f"ProactiveLoop 已启动  阈值={self._cfg.threshold}  "
            f"目标={self._cfg.default_channel}:{self._cfg.default_chat_id}"
        )
        if not hasattr(self, "_mcp_pool"):
            from proactive_v2.mcp_sources import McpClientPool
            workspace = getattr(self._sessions, "workspace", None)
            self._mcp_pool = McpClientPool(Path(workspace) if workspace else None)
        await self._mcp_pool.connect_all()
        try:
            await self._run_loop()
        finally:
            await self._mcp_pool.disconnect_all()
            logger.info("[proactive] mcp pool 已关闭")

    async def _run_loop(self) -> None:
        # 启动时先同步完成首次 feed 轮询，保证首次 tick 能拿到新鲜数据
        await self._poll_feeds_once()
        # 后台周期轮询
        asyncio.create_task(self._poll_loop())
        last_base_score: float | None = None
        while self._running:
            interval = self._next_interval(last_base_score)
            logger.info("[proactive] 下次 tick 间隔=%ds", interval)
            await asyncio.sleep(interval)
            try:
                last_base_score = await self._tick()
            except Exception:
                logger.exception("ProactiveLoop tick 异常")
                last_base_score = None

    def _next_interval(self, base_score: float | None = None) -> int:
        """根据 base_score 返回自适应等待秒数。无 presence 时回退固定间隔。"""
        if not self._presence:
            interval = self._cfg.interval_seconds
            self._trace_proactive_rate_decision(
                base_score=base_score,
                interval=interval,
                mode="fixed_no_presence",
            )
            return interval
        # base_score 由 _tick 传入；首次启动时用电量估算一个初始值
        if base_score is None:
            session_key = self._target_session_key()
            last_user_at = self._presence.get_last_user_at(session_key)
            energy = compute_energy(last_user_at)
            base_score = d_energy(energy) * self._cfg.score_weight_energy
        interval = next_tick_from_score(
            base_score,
            tick_s3=self._cfg.tick_interval_s3,
            tick_s2=self._cfg.tick_interval_s2,
            tick_s1=self._cfg.tick_interval_s1,
            tick_s0=self._cfg.tick_interval_s0,
            tick_jitter=self._cfg.tick_jitter,
            rng=self._rng,
        )
        self._trace_proactive_rate_decision(
            base_score=base_score,
            interval=interval,
            mode="adaptive",
        )
        return interval

    def _target_session_key(self) -> str:
        return self._sense.target_session_key()

    def stop(self) -> None:
        self._running = False

    def _sample_random_memory(self, n: int = 2) -> list[str]:
        """随机抽取 n 条记忆片段，无记忆时返回 []。"""
        if not self._memory:
            return []
        try:
            raw = str(self._memory.read_long_term_context() or "").strip()
            return sample_memory_chunks(raw, n=n)
        except Exception as e:
            logger.warning("[proactive] 随机记忆抽取失败: %s", e)
            return []

    def _has_global_memory(self) -> bool:
        return self._sense.has_global_memory()

    def _read_memory_text(self) -> str:
        return self._sense.read_memory_text()

    def _compute_energy(self) -> float:
        """计算目标 session 的当前电量（取目标与全局较高值）。"""
        return self._sense.compute_energy()

    def _compute_interruptibility(
        self,
        *,
        now_hour: int,
        now_utc: datetime,
        recent_msg_count: int,
    ) -> tuple[float, dict[str, float]]:
        """计算软打扰系数（0~1），并注入随机探索，避免长期锁死。"""
        return self._sense.compute_interruptibility(
            now_hour=now_hour,
            now_utc=now_utc,
            recent_msg_count=recent_msg_count,
        )

    # ── internal ──────────────────────────────────────────────────

    async def _tick(self) -> float | None:
        """执行一次 proactive v2 tick。"""
        # 真正的“主动决策 + 生成回复 + 发送”主链都在 AgentTick.tick() 里。
        return await self._agent_tick.tick()


def build_proactive_loop(**kwargs: Any) -> ProactiveLoop:
    return ProactiveLoop(**kwargs)
