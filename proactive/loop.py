"""
ProactiveLoop — 主动触达核心循环。

独立于 AgentLoop，定期：
  1. 拉取所有内容源的最新候选事件
  2. 获取用户最近聊天上下文
  3. 调用 LLM 反思：有没有值得主动说的
  4. 高于阈值时通过 MessagePushTool 发送消息
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
    from core.memory.port import MemoryPort

from agent.provider import LLMProvider
from agent.tools.message_push import MessagePushTool
from core.common.strategy_trace import build_strategy_trace_envelope
from core.net.http import get_default_http_requester
from proactive.anyaction import AnyActionGate, QuotaStore
from proactive.composer import Composer
from proactive.decide import Decider
from proactive.energy import (
    compute_energy,
    d_energy,
    next_tick_from_score,
)
from proactive.judge import Judge, MessageDeduper
from proactive.config import ProactiveConfig
from proactive.loop_helpers import (
    _build_sandboxed_shell,
    _decision_with_randomized_score,
    _format_items,
    _format_recent,
    _item_id,
    _semantic_text,
    _source_key,
)
from proactive.memory_retrieval import MemoryRetrievalService
from proactive.memory_sampler import sample_memory_chunks
from proactive.presence import PresenceStore
from proactive.sender import Sender
from proactive.sensor import Sensor
from proactive.state import ProactiveStateStore
from proactive.tick import ProactiveTick
from session.manager import SessionManager

logger = logging.getLogger(__name__)


class ProactiveLoop:
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
        memory_store: "MemoryPort | None" = None,
        presence: PresenceStore | None = None,
        rng: _random_module.Random | None = None,
        light_provider: LLMProvider | None = None,
        light_model: str = "",
        passive_busy_fn: Callable[[str], bool] | None = None,
        observe_writer=None,
        tool_registry: dict | None = None,
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
        self._tool_registry = tool_registry
        self._init_runtime_state(config)
        self._init_runtime_components()

    _FEED_POLL_INTERVAL_S: int = 1800  # 固定 30min，不读远程配置

    def _init_runtime_state(self, config: ProactiveConfig) -> None:
        self._running = False
        self._manual_trigger_event = asyncio.Event()
        self._manual_trigger_lock = asyncio.Lock()
        self._feed_poll_lock = asyncio.Lock()

    def _build_state_store(
        self,
        state_store: ProactiveStateStore | None,
        state_path: Path | None,
    ) -> ProactiveStateStore:
        if state_store is not None:
            return state_store
        return ProactiveStateStore(state_path or Path("proactive_state.json"))

    def _build_fitbit_provider(self):
        if not getattr(self._cfg, "fitbit_enabled", False):
            return None
        from proactive.fitbit_sleep import FitbitSleepProvider

        return FitbitSleepProvider(
            url=self._cfg.fitbit_url,
            poll_interval=self._cfg.fitbit_poll_seconds,
            sleeping_modifier=self._cfg.sleep_modifier_sleeping,
        )

    def _build_sender(self) -> Sender:
        return Sender(
            cfg=self._cfg,
            push_tool=self._push,
            sessions=self._sessions,
            presence=self._presence,
        )

    def _build_judge(self) -> Judge:
        return Judge(
            provider=self._light_provider or self._provider,
            model=self._light_model or self._model,
            max_tokens=self._max_tokens,
            format_recent=_format_recent,
            cfg=self._cfg,
        )

    def _build_composer(self) -> Composer:
        return Composer(
            provider=self._light_provider or self._provider,
            model=self._light_model or self._model,
            max_tokens=self._max_tokens,
            format_items=_format_items,
            format_recent=_format_recent,
        )

    def _build_anyaction_gate(self) -> AnyActionGate:
        if hasattr(self._state, "path"):
            quota_path = self._state.path.parent / "proactive_quota.json"
        else:
            quota_path = Path("proactive_quota.json")
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

    def _build_decide(self) -> Decider:
        return Decider(
            randomize_fn=lambda decision: _decision_with_randomized_score(
                decision,
                strength=self._cfg.decision_score_random_strength,
                rng=self._rng,
            ),
            source_key_fn=_source_key,
            item_id_fn=_item_id,
            semantic_text_fn=_semantic_text,
            semantic_text_max_chars=self._cfg.semantic_dedupe_text_max_chars,
            judge=self._judge,
            composer=self._composer,
        )

    def _build_memory_retriever(self) -> MemoryRetrievalService:
        return MemoryRetrievalService(
            cfg=self._cfg,
            memory=self._memory,
            item_id_fn=_item_id,
            trace_writer=self._trace_proactive_memory_retrieve,
            observe_writer=self._observe_writer,
            light_provider=self._light_provider,
            light_model=self._light_model,
        )

    def _build_agent_tick(self):
        if not self._cfg.use_agent_tick:
            return None
        from proactive_v2.agent_tick import AgentTick
        from proactive_v2.tools import ToolDeps
        from proactive import mcp_sources
        from agent.tools.web_fetch import WebFetchTool

        try:
            session_key = self._sense.target_session_key()
        except Exception:
            session_key = self._cfg.default_chat_id or ""

        last_user_at_fn = (
            (lambda: self._presence.get_last_user_at(session_key))
            if self._presence
            else (lambda: None)
        )

        # ── llm_fn ─────────────────────────────────────────────────────────
        agent_model = self._cfg.agent_tick_model or self._model
        provider = self._provider

        async def _llm_fn(messages: list[dict], schemas: list[dict]) -> dict | None:
            resp = await provider.chat(
                messages=messages,
                tools=schemas,
                model=agent_model,
                max_tokens=self._max_tokens,
                tool_choice="auto",
            )
            if not resp.tool_calls:
                return None
            tc = resp.tool_calls[0]
            return {"id": tc.id, "name": tc.name, "input": tc.arguments}

        # ── data source fns ────────────────────────────────────────────────
        # 注意：loop 在 async 函数内获取（asyncio.get_running_loop），避免 build 时无 loop

        async def _alert_fn() -> list[dict]:
            return await asyncio.get_running_loop().run_in_executor(
                None, mcp_sources.fetch_alert_events
            )

        async def _feed_fn(limit: int = 5) -> list[dict]:
            events = await asyncio.get_running_loop().run_in_executor(
                None, mcp_sources.fetch_content_events
            )
            return events[:limit]

        async def _context_fn() -> list[dict]:
            return await asyncio.get_running_loop().run_in_executor(
                None, mcp_sources.fetch_context_data
            )

        sense = self._sense

        async def _recent_chat_fn(n: int = 20) -> list[dict]:
            # Sensor.collect_recent() 无参数
            return await asyncio.get_running_loop().run_in_executor(
                None, sense.collect_recent
            )

        # ── ack_fn ─────────────────────────────────────────────────────────
        async def _ack_fn(compound_key: str, ttl_hours: int) -> None:
            """compound_key 格式："{ack_server}:{id}"，如 "feed-mcp:c1"."""
            parts = compound_key.split(":", 1)
            if len(parts) != 2:
                return
            ack_server, item_id = parts
            source_key = f"mcp:{ack_server}"
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: mcp_sources.acknowledge_content_entries(
                    [(source_key, item_id)], ttl_hours=ttl_hours
                ),
            )

        # ── alert_ack_fn ───────────────────────────────────────────────────
        async def _alert_ack_fn(compound_key: str) -> None:
            """Alert 专用通道，走 acknowledge_events（非 content entries）。
            compound_key 格式："{ack_server}:{id}"，如 "alert-mcp:evt42"。
            """
            import types as _types
            parts = compound_key.split(":", 1)
            if len(parts) != 2:
                return
            ack_server, ack_id = parts
            event_proxy = _types.SimpleNamespace(_ack_server=ack_server, ack_id=ack_id)
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: mcp_sources.acknowledge_events([event_proxy]),
            )

        tool_deps = ToolDeps(
            web_fetch_tool=WebFetchTool(),
            memory=self._memory,
            alert_fn=_alert_fn,
            feed_fn=_feed_fn,
            context_fn=_context_fn,
            recent_chat_fn=_recent_chat_fn,
            ack_fn=_ack_fn,
            alert_ack_fn=_alert_ack_fn,
            max_chars=self._cfg.agent_tick_web_fetch_max_chars,
        )

        recent_n = getattr(self._cfg, "message_dedupe_recent_n", 5)
        recent_proactive_fn = (
            (lambda: self._sense.collect_recent_proactive(recent_n))
            if hasattr(self._sense, "collect_recent_proactive")
            else None
        )

        return AgentTick(
            cfg=self._cfg,
            session_key=session_key,
            state_store=self._state,
            any_action_gate=self._anyaction,
            last_user_at_fn=last_user_at_fn,
            passive_busy_fn=self._passive_busy_fn,
            sender=self._sender,
            deduper=self._message_deduper,
            tool_deps=tool_deps,
            llm_fn=_llm_fn,
            rng=self._rng,
            recent_proactive_fn=recent_proactive_fn,
        )

    def _build_message_deduper(self) -> MessageDeduper | None:
        if not self._cfg.message_dedupe_enabled:
            return None
        return MessageDeduper(
            provider=self._provider,
            model=self._model,
            max_tokens=self._max_tokens,
        )

    def _build_tick(self) -> ProactiveTick:
        return ProactiveTick(
            cfg=self._cfg,
            state=self._state,
            presence=self._presence,
            rng=self._rng,
            sensor=self._sense,
            decide=self._decide,
            composer=self._composer,
            judge=self._judge,
            sender=self._sender,
            memory_retriever=self._memory_retrieval,
            anyaction=self._anyaction,
            message_deduper=self._message_deduper,
            skill_action_runner=self._build_skill_action_runner(),
            provider=self._provider,
            model=self._model,
            light_provider=self._light_provider,
            light_model=self._light_model,
            passive_busy_fn=self._passive_busy_fn,
            stage_trace_writer=(
                lambda payload: self._append_trace_line(
                    "proactive_strategy_trace.jsonl", payload
                )
            ),
            observe_writer=self._observe_writer,
            tool_registry=self._tool_registry,
        )

    def _init_runtime_components(self) -> None:
        self._sender = self._build_sender()
        self._composer = self._build_composer()
        self._judge = self._build_judge()
        self._anyaction = self._build_anyaction_gate()
        self._sense = self._build_sense(self._build_fitbit_provider())
        self._decide = self._build_decide()
        self._memory_retrieval = self._build_memory_retriever()
        self._message_deduper = self._build_message_deduper()
        self._engine = self._build_tick()
        self._agent_tick = self._build_agent_tick()
        self._trace_proactive_config_snapshot()

    def _build_skill_action_runner(self):
        if not self._cfg.skill_actions_enabled:
            return None
        skill_path_str = (self._cfg.skill_actions_path or "").strip()
        if not skill_path_str:
            return None
        skill_path = Path(skill_path_str).expanduser()
        state_path = (
            self._state.path.parent / "skill_action_state.json"
            if hasattr(self._state, "path")
            else None
        )
        from proactive.skill_action import SkillActionRegistry, SkillActionRunner

        registry = SkillActionRegistry(skill_path)
        workspace = getattr(self._sessions, "workspace", None)
        agent_tasks_dir = (Path(workspace) / "agent-tasks") if workspace else None
        if agent_tasks_dir:
            agent_tasks_dir.mkdir(parents=True, exist_ok=True)
        return SkillActionRunner(
            registry,
            rng=self._rng,
            state_path=state_path,
            subagent_factory=self._build_subagent_factory(),
            agent_tasks_dir=agent_tasks_dir,
            memory_retrieve_fn=(
                self._memory.retrieve_related if self._memory is not None else None
            ),
            memory_format_fn=(
                self._memory.format_injection_block
                if self._memory is not None
                else None
            ),
        )

    def _build_subagent_factory(self):
        from agent.background.subagent_profiles import (
            SubagentRuntime,
            build_skill_action_spec,
        )
        from prompts.background import SKILL_ACTION_AGENT_BASE_PROMPT

        workspace = getattr(self._sessions, "workspace", None)
        if workspace is None:
            logger.warning("[proactive] 无法获取 workspace，SubAgent 不可用")
            return None
        agent_tasks_dir = Path(workspace) / "agent-tasks"
        agent_tasks_dir.mkdir(parents=True, exist_ok=True)
        runtime = SubagentRuntime(
            provider=self._provider,
            model=self._model,
            max_tokens=self._max_tokens,
            memory=self._memory,
        )
        db_path = agent_tasks_dir / "task_notes.db"
        shell_tool = _build_sandboxed_shell(agent_tasks_dir)
        fetch_requester = get_default_http_requester("external_default")

        def factory(
            action_id: str,
            system_prompt_override: str = SKILL_ACTION_AGENT_BASE_PROMPT,
        ):
            action_dir = agent_tasks_dir / action_id
            action_dir.mkdir(parents=True, exist_ok=True)
            spec = build_skill_action_spec(
                agent_tasks_dir=agent_tasks_dir,
                action_dir=action_dir,
                fetch_requester=fetch_requester,
                shell_tool=shell_tool,
                db_path=db_path,
                push_tool=self._push,
                channel=self._cfg.default_channel or "",
                chat_id=self._cfg.default_chat_id or "",
                system_prompt=system_prompt_override,
                max_iterations=40,
            )
            return spec.build(runtime)

        return factory

    async def trigger_skill_action(
        self,
        action_id: str | None = None,
    ) -> tuple[bool, str]:
        if not self._cfg.skill_actions_enabled:
            return False, "skill_actions 未启用（skill_actions_enabled=false）"
        runner = self._engine._skill_action_runner
        if runner is None:
            return False, "SkillActionRunner 未初始化"
        if self._manual_trigger_lock.locked():
            return False, "已有手动触发正在执行，请稍后再试"
        async with self._manual_trigger_lock:
            self._manual_trigger_event.set()
            now_utc = datetime.now(timezone.utc)
            action = self._pick_skill_action(runner, action_id)
            if isinstance(action, str):
                return False, action
            if self._cfg.anyaction_enabled and self._engine._anyaction:
                self._engine._anyaction.record_action(now_utc=now_utc)
            success, stdout_str = await runner.run(action)
            if success and stdout_str:
                await self._engine._try_send_proactive_text(action.id, stdout_str)
            if success:
                return True, f"skill_action {action.id!r} 已完成"
            return False, f"skill_action {action.id!r} 执行失败"

    def _pick_skill_action(self, runner, action_id: str | None):
        if action_id:
            action = runner._registry.get(action_id)
            if action is None:
                return f"找不到 action_id={action_id!r}"
            if not action.enabled:
                return f"action_id={action_id!r} 已禁用"
            return action
        action = runner.pick()
        if action is None:
            return "当前无可用 skill action（配额已满或间隔未到）"
        return action

    def _trace_proactive_memory_retrieve(self, payload: dict[str, Any]) -> None:
        line = {
            **build_strategy_trace_envelope(
                trace_type="proactive_stage",
                source="proactive.memory",
                subject_kind="global",
                subject_id="proactive-memory",
                payload=payload,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ),
            **payload,
        }
        self._append_trace_line("proactive_memory_retrieve_trace.jsonl", line)

    def _trace_proactive_config_snapshot(self) -> None:
        payload = {
            "enabled": self._cfg.enabled,
            "threshold": self._cfg.threshold,
            "score_llm_threshold": self._cfg.score_llm_threshold,
            "score_pre_threshold": self._cfg.score_pre_threshold,
            "tick_interval_s0": self._cfg.tick_interval_s0,
            "tick_interval_s1": self._cfg.tick_interval_s1,
            "tick_interval_s2": self._cfg.tick_interval_s2,
            "tick_interval_s3": self._cfg.tick_interval_s3,
            "tick_jitter": self._cfg.tick_jitter,
            "anyaction_enabled": self._cfg.anyaction_enabled,
            "anyaction_min_interval_seconds": self._cfg.anyaction_min_interval_seconds,
            "anyaction_probability_min": self._cfg.anyaction_probability_min,
            "anyaction_probability_max": self._cfg.anyaction_probability_max,
            "memory_retrieval_enabled": self._cfg.memory_retrieval_enabled,
            "memory_top_k_procedure": self._cfg.memory_top_k_procedure,
            "memory_top_k_history": self._cfg.memory_top_k_history,
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
                from proactive import mcp_sources
                await asyncio.get_event_loop().run_in_executor(
                    None, mcp_sources.poll_content_feeds
                )
                logger.info("[proactive] feed poll 完成")
            except Exception as e:
                logger.warning("[proactive] feed poll 系统级失败: %s", e)

    async def _poll_loop(self) -> None:
        """每 _FEED_POLL_INTERVAL_S 秒周期性触发 feed 轮询。"""
        while self._running:
            await asyncio.sleep(self._FEED_POLL_INTERVAL_S)
            if not self._running:
                break
            await self._poll_feeds_once()

    async def run(self) -> None:
        self._running = True
        logger.info(
            f"ProactiveLoop 已启动  阈值={self._cfg.threshold}  "
            f"目标={self._cfg.default_channel}:{self._cfg.default_chat_id}"
        )
        # 启动时先同步完成首次 feed 轮询，保证首次 tick 能拿到新鲜数据
        await self._poll_feeds_once()
        # 后台周期轮询
        asyncio.create_task(self._poll_loop())
        last_base_score: float | None = None
        while self._running:
            interval = self._next_interval(last_base_score)
            logger.info("[proactive] 下次 tick 间隔=%ds", interval)
            # 等待 interval 秒，或被手动触发事件提前唤醒
            try:
                await asyncio.wait_for(
                    self._manual_trigger_event.wait(), timeout=interval
                )
                # 事件被 set：手动触发了 skill action，不执行正常 tick，继续等下次
                self._manual_trigger_event.clear()
                logger.info("[proactive] 正常 tick 被手动触发事件中断，跳过本轮 tick")
                continue
            except asyncio.TimeoutError:
                pass  # 正常超时，执行正常 tick
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
            raw = self._memory.read_long_term().strip()
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
        """执行一次主动判断循环。
        返回 base_score 供调度器调整间隔；None 表示 gate 按能量自算（不强制最长间隔）。
        use_agent_tick=True 时走 v2 AgentTick，否则走 v1 ProactiveTick。
        """
        if self._cfg.use_agent_tick:
            return await self._agent_tick.tick()
        return await self._engine.tick()

    def _collect_recent(self) -> list[dict]:
        """取目标会话最近 N 条消息（只取 user/assistant 文本）。"""
        return self._sense.collect_recent()

    def _collect_global_memory(self) -> str:
        if not self._cfg.use_global_memory:
            logger.info("[proactive] 全局记忆已禁用（use_global_memory=false）")
            return "（全局记忆已禁用）"
        if not self._memory:
            logger.info("[proactive] 未注入 MemoryStore，跳过全局记忆")
            return "（无全局记忆）"
        try:
            raw = self._memory.get_memory_context().strip()
            if not raw:
                logger.info("[proactive] 全局记忆为空")
                return "（无全局记忆）"
            logger.info("[proactive] 已注入全局记忆 chars=%d", len(raw))
            return raw
        except Exception as e:
            logger.warning("[proactive] 读取全局记忆失败: %s", e)
            return "（读取全局记忆失败）"

    def _read_agents_md(self) -> str:
        # 与全局记忆开关保持一致：禁用 memory 时不注入 AGENTS 导航
        if not self._cfg.use_global_memory:
            return ""
        try:
            agents_md = self._sessions.workspace / "AGENTS.md"
            if not agents_md.is_file():
                return ""
            content = agents_md.read_text(encoding="utf-8").strip()[:3000]
            return content
        except Exception as e:
            logger.warning("[proactive] 读取 AGENTS.md 失败: %s", e)
            return ""

    def _build_context_block(self) -> str:
        parts = [self._collect_global_memory()]
        agents = self._read_agents_md()
        if agents:
            parts.append(f"## Workspace 导航（AGENTS.md）\n{agents}")
        return "\n\n".join(p for p in parts if p)

    async def _send(self, message: str, meta: Any | None = None) -> bool:
        return await self._sender.send(message, meta)


def build_proactive_loop(**kwargs: Any) -> ProactiveLoop:
    return ProactiveLoop(**kwargs)
