"""
ProactiveLoop — 主动触达核心循环。

独立于 AgentLoop，定期：
  1. 拉取所有订阅信息流的最新内容
  2. 获取用户最近聊天上下文
  3. 调用 LLM 反思：有没有值得主动说的
  4. 高于阈值时通过 MessagePushTool 发送消息
"""

from __future__ import annotations

import asyncio
import logging
import random as _random_module
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from core.memory.port import MemoryPort

from agent.provider import LLMProvider
from agent.tools.message_push import MessagePushTool
from proactive.energy import (
    compute_energy,
    d_energy,
    next_tick_from_score,
)
from proactive.config import ProactiveConfig
from proactive.loop_factory import ProactiveLoopFactoryMixin
from proactive.loop_helpers import (
    _Decision,
    _parse_decision,
)
from proactive.loop_runtime import ProactiveLoopRuntimeMixin
from proactive.loop_trigger import ProactiveLoopTriggerMixin
from proactive.loop_traces import ProactiveLoopTraceMixin
from proactive.memory_sampler import sample_memory_chunks
from proactive.presence import PresenceStore
from proactive.schedule import ScheduleStore
from proactive.state import ProactiveStateStore
from session.manager import SessionManager

logger = logging.getLogger(__name__)


class ProactiveLoop(
    ProactiveLoopRuntimeMixin,
    ProactiveLoopFactoryMixin,
    ProactiveLoopTriggerMixin,
    ProactiveLoopTraceMixin,
):
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
        schedule: ScheduleStore | None = None,
        rng: _random_module.Random | None = None,
        light_provider: LLMProvider | None = None,
        light_model: str = "",
        passive_busy_fn: Callable[[str], bool] | None = None,
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
        self._schedule = schedule
        self._rng = rng
        self._light_provider = light_provider or provider
        self._light_model = light_model or (config.model or model)
        self._passive_busy_fn = passive_busy_fn
        self._init_runtime_state(config)
        self._init_runtime_components()

    async def run(self) -> None:
        self._running = True
        logger.info(
            f"ProactiveLoop 已启动  阈值={self._cfg.threshold}  "
            f"目标={self._cfg.default_channel}:{self._cfg.default_chat_id}"
        )
        last_base_score: float | None = None
        try:
            # 启动后立即执行一次 tick，避免首次触达还要额外等待一个 interval。
            last_base_score = await self._tick()
        except Exception:
            logger.exception("ProactiveLoop 启动即刻 tick 异常")
            last_base_score = None
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

    def _quiet_hours(self) -> tuple[int, int, float]:
        """从 schedule.json 读取静默时段配置，缺失时回退 cfg 默认值。"""
        return self._sense.quiet_hours()

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
        """
        return await self._engine.tick()

    def _filter_new_items(
        self, items: list[FeedItem]
    ) -> tuple[list[FeedItem], list[tuple[str, str]], list[tuple[str, str]]]:
        return self._sense.filter_new_items(items)

    def _collect_recent(self) -> list[dict]:
        """取目标会话最近 N 条消息（只取 user/assistant 文本）。"""
        return self._sense.collect_recent()

    async def _reflect(
        self,
        items: list[FeedItem],
        recent: list[dict],
        energy: float = 0.0,
        urge: float = 0.0,
        is_crisis: bool = False,
        decision_signals: dict[str, object] | None = None,
    ) -> _Decision:
        return await self._reflector.reflect(
            items=items,
            recent=recent,
            energy=energy,
            urge=urge,
            is_crisis=is_crisis,
            decision_signals=decision_signals,
        )

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
