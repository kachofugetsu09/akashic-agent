"""
proactive_v2/agent_tick.py — AgentTick

结构：
  tick()
    ├── Pre-gate（全部失败直接 return None，不进 loop，不 ack）
    └── _run_loop(ctx)  → float | None
"""

from __future__ import annotations

import json
import logging
import random as _random_module
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit, urlunsplit

from agent.tool_hooks import ShellRmToRestoreHook, ToolExecutionRequest, ToolExecutor
from agent.turns.orchestrator import TurnOrchestrator
from agent.turns.result import TurnOutbound, TurnResult, TurnTrace
from core.memory.runtime_facade import MemoryRuntimeFacade
from proactive_v2.config import ProactiveConfig
from proactive_v2.contracts import (
    normalize_alert,
    normalize_content,
    normalize_context,
)
from proactive_v2.context import AgentTickContext
from proactive_v2.drift_runner import DriftRunner
from proactive_v2.gateway import DataGateway, GatewayDeps, GatewayResult
from proactive_v2.tools import TOOL_SCHEMAS, ToolDeps, dispatch, execute

logger = logging.getLogger(__name__)

# ── ACK TTL 常量 ──────────────────────────────────────────────────────────

_CITED_ACK_TTL = 168       # cited content/alert → 168h
_UNCITED_ACK_TTL = 24      # interesting uncited → 24h
_POST_GUARD_ACK_TTL = 24   # delivery/message dedupe hit → 24h
_DISCARDED_ACK_TTL = 720   # mark_not_interesting → 720h


def _read_long_term_text(memory: MemoryRuntimeFacade | None) -> str:
    if memory is not None:
        return str(memory.read_long_term_context() or "")
    return ""


def _read_self_text(memory: MemoryRuntimeFacade | None) -> str:
    if memory is not None:
        return str(memory.read_self() or "")
    return ""


# ── 模块级 delivery key + ACK 函数 ───────────────────────────────────────

def _log_content_candidates(gw: GatewayResult) -> None:
    if not gw.content_meta:
        logger.info("[proactive_v2] content candidates: 0")
        return

    lines: list[str] = []
    for index, item in enumerate(gw.content_meta, 1):
        title = str(item.get("title") or "").strip() or "(no title)"
        source = str(item.get("source") or "").strip()
        line = f"[{index}] {title}"
        if source:
            line += f" | source={source}"
        lines.append(line)

    logger.info(
        "[proactive_v2] content candidates: %d\n%s",
        len(gw.content_meta),
        "\n".join(lines),
    )

def _normalize_delivery_url(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    parts = urlsplit(text)
    path = parts.path.rstrip("/") or parts.path
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def _build_delivery_refs(ctx: AgentTickContext) -> list[str]:
    if not ctx.cited_item_ids:
        return []

    content_map = {
        f"{e.get('ack_server', '')}:{e.get('event_id') or e.get('id', '')}": e
        for e in ctx.fetched_contents
        if e.get("ack_server") and (e.get("event_id") or e.get("id"))
    }
    refs: list[str] = []

    for key in sorted(set(ctx.cited_item_ids)):
        meta = content_map.get(key)
        if meta is None:
            refs.append(f"id:{key}")
            continue

        # 1. 优先按稳定 URL 去重，挡住同一篇内容换 event_id 的重复发送。
        url = _normalize_delivery_url(str(meta.get("url") or ""))
        if url:
            refs.append(f"url:{url}")
            continue

        # 2. 没有 URL 时退化到来源+标题，仍比纯 event_id 稳定。
        source = str(meta.get("source") or meta.get("source_name") or "").strip().lower()
        title = str(meta.get("title") or "").strip().lower()
        if title:
            refs.append(f"title:{source}|{title}")
            continue

        # 3. 最后再退回原始 cited key，保持兼容。
        refs.append(f"id:{key}")

    return sorted(set(refs))


def build_delivery_key(ctx: AgentTickContext) -> str:
    """优先按 cited 内容的稳定来源标识去重；为空时退化为消息文本 hash。"""
    refs = _build_delivery_refs(ctx)
    if refs and any(not ref.startswith("id:") for ref in refs):
        key_src = json.dumps(refs)
    elif ctx.cited_item_ids:
        key_src = json.dumps(sorted(ctx.cited_item_ids))
    else:
        key_src = ctx.final_message[:500]
    return sha1(key_src.encode()).hexdigest()[:16]


async def ack_discarded(ctx: AgentTickContext, ack_fn) -> None:
    """Skip / send_fail 路径：只 ACK discarded 720h。"""
    if ack_fn is None:
        return
    for key in ctx.discarded_item_ids:
        await ack_fn(key, _DISCARDED_ACK_TTL)


async def ack_post_guard_fail(ctx: AgentTickContext, ack_fn, *, alert_ack_fn=None) -> None:
    """delivery_dedupe / message_dedupe 命中：
    content cited → 24h；alert cited → alert_ack_fn（独立通道，无 TTL）；
    alert_ack_fn=None 时回退到普通 ack_fn（24h）；
    uncited alert（本轮其余 fetched alerts）→ 同上通道，一次性清空本批 alert；
    uncited interesting（content）→ 24h；discarded → 720h。
    """
    if ack_fn is None:
        return
    fetched_alert_keys = {f"{e['ack_server']}:{e.get('event_id') or e.get('id', '')}" for e in ctx.fetched_alerts}
    cited_set = set(ctx.cited_item_ids)

    async def _ack_alert(key: str) -> None:
        if alert_ack_fn is not None:
            await alert_ack_fn(key)
        else:
            await ack_fn(key, _POST_GUARD_ACK_TTL)

    # cited content → 24h
    for key in cited_set - fetched_alert_keys:
        await ack_fn(key, _POST_GUARD_ACK_TTL)
    # cited alert → alert_ack_fn（独立通道）；无时回退到 ack_fn 24h
    for key in cited_set & fetched_alert_keys:
        await _ack_alert(key)
    # uncited fetched alerts → 同一通道，一次性清空，防止逐条 tick 循环
    for key in fetched_alert_keys - cited_set:
        await _ack_alert(key)
    # uncited interesting（content，alert 排除）→ 24h
    for key in (ctx.interesting_item_ids - cited_set) - fetched_alert_keys:
        await ack_fn(key, _POST_GUARD_ACK_TTL)
    for key in ctx.discarded_item_ids:
        await ack_fn(key, _DISCARDED_ACK_TTL)


async def ack_on_success(ctx: AgentTickContext, ack_fn, *, alert_ack_fn=None) -> None:
    """发送成功：
    cited content → 168h；cited alert → alert_ack_fn（独立通道，无 TTL）；
    alert_ack_fn=None 时回退到普通 ack_fn（168h）；
    uncited interesting（content）→ 24h；discarded → 720h。
    """
    if ack_fn is None:
        return
    fetched_alert_keys = {f"{e['ack_server']}:{e.get('event_id') or e.get('id', '')}" for e in ctx.fetched_alerts}
    fetched_content_keys = {f"{e['ack_server']}:{e.get('event_id') or e.get('id', '')}" for e in ctx.fetched_contents}
    cited_set = set(ctx.cited_item_ids)

    # cited content → 168h
    for key in cited_set & fetched_content_keys:
        await ack_fn(key, _CITED_ACK_TTL)

    # cited alert → 独立 alert_ack_fn（无 TTL）；无时回退到普通 ack_fn（168h）
    for key in cited_set & fetched_alert_keys:
        if alert_ack_fn is not None:
            await alert_ack_fn(key)
        else:
            await ack_fn(key, _CITED_ACK_TTL)

    # uncited interesting（content，alert 已被 fetched_alert_keys 排除）→ 24h
    for key in (ctx.interesting_item_ids - cited_set) - fetched_alert_keys:
        await ack_fn(key, _UNCITED_ACK_TTL)

    for key in ctx.discarded_item_ids:
        await ack_fn(key, _DISCARDED_ACK_TTL)


class AgentTick:
    def __init__(
        self,
        *,
        cfg: ProactiveConfig,
        session_key: str,
        state_store: Any,
        any_action_gate: Any | None,
        last_user_at_fn: Callable[[], datetime | None],
        passive_busy_fn: Callable[[str], bool] | None,
        turn_orchestrator: TurnOrchestrator | None = None,
        deduper: Any,
        tool_deps: ToolDeps,
        gateway_deps: GatewayDeps | None = None,
        workspace_context_fn: Callable[[], str] | None = None,
        llm_fn: Any | None = None,
        rng: Any | None = None,
        recent_proactive_fn: Callable[[], list] | None = None,
        drift_runner: DriftRunner | None = None,
    ) -> None:
        self._cfg = cfg
        self._session_key = session_key
        self._state_store = state_store
        self._any_action_gate = any_action_gate
        self._last_user_at_fn = last_user_at_fn
        self._passive_busy_fn = passive_busy_fn
        self._turn_orchestrator = turn_orchestrator
        self._deduper = deduper
        self._tool_deps = tool_deps
        self._gateway_deps = gateway_deps
        self._workspace_context_fn = workspace_context_fn
        self._llm_fn = llm_fn
        self._rng = rng if rng is not None else _random_module.Random()
        self._recent_proactive_fn = recent_proactive_fn
        self._drift_runner = drift_runner
        if self._drift_runner is not None and getattr(self._drift_runner, "step_recorder", None) is None:
            self._drift_runner.step_recorder = (
                lambda ctx, phase, tool_name, tool_call_id, tool_args, tool_result_text: (
                    self._record_tick_step(
                        ctx,
                        phase=phase,
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        tool_args=tool_args,
                        tool_result_text=tool_result_text,
                    )
                )
            )
        self._tool_executor = ToolExecutor([ShellRmToRestoreHook()])
        self.last_ctx: AgentTickContext | None = None  # 供测试检查

    async def tick(self) -> float | None:
        # 1. 每次 tick 先创建一个新的上下文容器。
        #    后续 gateway 输入、分类结果、最终消息、ack 相关状态都写在这里。
        ctx = AgentTickContext(
            session_key=self._session_key,
            now_utc=datetime.now(timezone.utc),
        )

        # ── Pre-gate ──────────────────────────────────────────────────────

        # 5.1 passive_busy（系统硬 veto）
        if self._passive_busy_fn and self._passive_busy_fn(self._session_key):
            logger.debug("[proactive_v2] pre-gate: passive_busy → return None")
            self._record_tick_log_finish(ctx, gate_exit="busy")
            return None

        # 5.2 delivery_cooldown
        if self._state_store.count_deliveries_in_window(
            self._session_key,
            self._cfg.agent_tick_delivery_cooldown_hours,
        ) > 0:
            logger.debug("[proactive_v2] pre-gate: delivery_cooldown → return None")
            self._record_tick_log_finish(ctx, gate_exit="cooldown")
            return None

        # 5.3 AnyAction gate
        if self._any_action_gate is not None:
            should_act, meta = self._any_action_gate.should_act(
                now_utc=ctx.now_utc,
                last_user_at=self._last_user_at_fn(),
            )
            if not should_act:
                logger.debug("[proactive_v2] pre-gate: anyaction gate → return None meta=%s", meta)
                self._record_tick_log_finish(ctx, gate_exit="presence")
                return None

        # 5.4 context gate（概率 + 配额）
        context_as_fallback_open = self._rng.random() < self._cfg.agent_tick_context_prob
        if context_as_fallback_open:
            last_at = self._state_store.get_last_context_only_at(self._session_key)
            count_24h = self._state_store.count_context_only_in_window(
                self._session_key, window_hours=24
            )
            if (
                (
                    last_at is not None
                    and (ctx.now_utc - last_at).total_seconds()
                    < self._cfg.context_only_min_interval_hours * 3600
                )
                or count_24h >= self._cfg.context_only_daily_max
            ):
                context_as_fallback_open = False

        ctx.context_as_fallback_open = context_as_fallback_open
        self.last_ctx = ctx
        self._record_tick_log_start(ctx)

        # 2. 通过 pre-gate 后，才真正进入“预取数据 -> agent loop -> post_loop”主流程。
        logger.info("[proactive_v2] tick: pre-gate passed, starting loop (context_fallback=%s)", ctx.context_as_fallback_open)
        entered_execution = await self._run_loop(ctx)
        if (entered_execution or ctx.drift_entered) and self._any_action_gate is not None:
            self._any_action_gate.record_action(now_utc=ctx.now_utc)
        if ctx.drift_entered:
            logger.info(
                "[proactive_v2] tick: drift entered, skipping normal post_loop message_sent=%s finished=%s",
                ctx.drift_message_sent,
                ctx.drift_finished,
            )
            self._record_tick_log_finish(ctx)
            ctx.content_store.clear()
            return 0.0
        result = await self._post_loop(ctx)
        ctx.content_store.clear()  # 清理 hashmap，防止内存泄漏
        return result

    def _record_tick_log_start(self, ctx: AgentTickContext) -> None:
        self._state_store.record_tick_log_start(
            tick_id=ctx.tick_id,
            session_key=self._session_key,
            started_at=ctx.now_utc.isoformat(),
            gate_exit=None,
        )

    def _record_tick_log_finish(
        self,
        ctx: AgentTickContext,
        *,
        gate_exit: str | None = None,
        result: TurnResult | None = None,
    ) -> None:
        decision = result.decision if result is not None else ctx.terminal_action
        if ctx.drift_entered and result is None and decision is None:
            decision = "reply" if ctx.drift_message_sent else "skip"
        trace_extra = result.trace.extra if result is not None and result.trace is not None else {}
        skip_reason = str(trace_extra.get("skip_reason") or ctx.skip_reason or "")
        final_message = ""
        if result is not None and result.outbound is not None:
            final_message = str(result.outbound.content or "")
        elif ctx.final_message:
            final_message = ctx.final_message
        self._state_store.record_tick_log_finish(
            tick_id=ctx.tick_id,
            session_key=self._session_key,
            started_at=ctx.now_utc.isoformat(),
            finished_at=datetime.now(timezone.utc).isoformat(),
            gate_exit=gate_exit,
            terminal_action=decision,
            skip_reason=skip_reason,
            steps_taken=ctx.steps_taken,
            alert_count=len(ctx.fetched_alerts),
            content_count=len(ctx.fetched_contents),
            context_count=len(ctx.fetched_context),
            interesting_ids=sorted(ctx.interesting_item_ids),
            discarded_ids=sorted(ctx.discarded_item_ids),
            cited_ids=list(ctx.cited_item_ids),
            drift_entered=ctx.drift_entered,
            final_message=final_message,
        )

    def _record_tick_step(
        self,
        ctx: AgentTickContext,
        *,
        phase: str,
        tool_name: str,
        tool_call_id: str,
        tool_args: dict[str, Any],
        tool_result_text: str,
    ) -> None:
        self._state_store.record_tick_step_log(
            tick_id=ctx.tick_id,
            step_index=ctx.steps_taken,
            phase=phase,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_args=tool_args,
            tool_result_text=tool_result_text,
            terminal_action_after=ctx.terminal_action,
            skip_reason_after=ctx.skip_reason,
            interesting_ids_after=sorted(ctx.interesting_item_ids),
            discarded_ids_after=sorted(ctx.discarded_item_ids),
            cited_ids_after=list(ctx.cited_item_ids),
            final_message_after=ctx.final_message,
        )

    def _build_system_prompt(self, ctx: AgentTickContext, gw: GatewayResult) -> str:
        fallback_status = "允许" if ctx.context_as_fallback_open else "不允许"

        memory_block = ""
        self_block = ""
        recent_context_block = ""
        if self._tool_deps.memory is not None:
            try:
                raw = _read_long_term_text(self._tool_deps.memory).strip()
                if raw:
                    memory_block = "\n【用户长期记忆】\n" + raw + "\n"
            except Exception:
                pass
            try:
                self_content = _read_self_text(self._tool_deps.memory).strip()
                if self_content:
                    self_block = f"## Akashic 自我认知\n\n{self_content}\n\n"
            except Exception:
                pass
            try:
                rc = str(self._tool_deps.memory.read_recent_context() or "").strip()
                if rc:
                    recent_context_block = "【近期交互上下文】\n" + rc + "\n\n"
            except Exception:
                pass

        alert_block = self._render_alert_block(gw.alerts)
        context_block = self._render_context_block(gw.context)

        workspace_context_block = ""
        if self._workspace_context_fn is not None:
            try:
                raw = (self._workspace_context_fn() or "").strip()
                if raw:
                    workspace_context_block = (
                        "【Workspace 主动上下文（主/被动 loop 共享规则面板，不是内容源）】\n"
                        + raw[:3000]
                        + "\n\n"
                    )
            except Exception:
                pass

        content_block = self._render_content_block(gw.content_meta, gw.content_store)

        from agent.persona import AKASHIC_IDENTITY, PERSONALITY_RULES

        return (
            f"{AKASHIC_IDENTITY}\n\n"
            f"{PERSONALITY_RULES}\n\n"
            f"{self_block}"
            "你现在处于主动推送决策模式：判断现在是否该给用户发一条消息，以及发什么。\n"
            "数据已预取完毕，基于下方数据直接决策。\n\n"
            f"{alert_block}"
            f"{content_block}"
            f"{context_block}"
            f"{workspace_context_block}"
            f"{memory_block}\n"
            f"{recent_context_block}"
            f"【优先级】Alert > Content > Context-fallback（本轮：{fallback_status}）\n\n"
            "【你的任务】\n"
            "⚡ 如果本轮有 Alert：把本轮所有 Alert 整合成一条消息，调用 message_push 并填写本轮全部 Alert 的 id 作为 evidence，然后 finish_turn(decision=reply) 结束。Alert 是系统触发的高优先级通知，不走内容筛选流程。\n"
            "1. 对本轮 Content 逐条判断：这条内容是否可能让用户不感兴趣，是否可能不符合规则，是否值得进入 interesting。\n"
            "2. 你的主工作是分类，不是主动研究新题材，不是主动扩展候选池。\n"
            "3. 你要基于规则和用户偏好，把本轮 Content 分成 interesting 和 not_interesting。\n\n"
            "【你的输出】\n"
            "1. 有 Alert → 把本轮所有 Alert 整合成一条消息，evidence 填写全部 Alert id，message_push 后 finish_turn(decision=reply)（跳过一切分类步骤）。\n"
            "2. 无 Alert：对每条 Content 给出最终分类：mark_interesting 或 mark_not_interesting。\n"
            "3. 如果最终没有 interesting，调用 finish_turn(decision=skip, reason=no_content)。\n"
            "4. 如果最终有 interesting，生成一条最终消息并按 message_push + finish_turn(decision=reply) 收尾。\n\n"
            "【工具职责】\n"
            "1. Workspace 主动上下文：这是用户当前明确提出并要求你遵守的规则集合。它定义你该怎么筛、哪些要先验证、哪些必须过滤；它不提供新闻事实。\n"
            "2. recall_memory：仅用于 Content 评估——判断单条内容是否可能是用户雷点，或是否可能让用户感兴趣。Alert 不需要调用此工具。\n"
            "   ⚠️ 当内容标题稀疏（如 'RT @xxx'、'Image'、转推无正文）时，必须把 source（来源/作者名）作为关键词纳入 query，不要只靠标题查询。\n"
            "   例：source=terasumc (Artist) 时，query 应包含 'terasumc' 而非只用推文标题。\n"
            "3. get_content：给当前候选条目补正文。\n"
            "4. web_fetch：优先用于抓取当前候选条目的直接来源页面或正文；当条目已经有明确 URL，且你需要补正文、核实细节、核实规则时，先用它。\n"
            "5. get_recent_chat：只用于最后判断现在是否适合打扰用户。\n"
            "6. mark_interesting / mark_not_interesting：写入最终分类结果。\n"
            "7. message_push：暂存草稿，不终止 loop。\n"
            "8. finish_turn(decision=reply) 或 finish_turn(decision=skip, reason=...)：提交或放弃，终止 loop。\n\n"
            "【规则优先级】\n"
            "1. Workspace 主动上下文代表用户当前对主动推送的明确要求，应视为规则而不是建议。\n"
            "2. 当 Workspace 主动上下文规定了过滤条件、白名单、黑名单、必须先验证的步骤时，你必须遵守，不要凭常识跳过。\n"
            "3. recall_memory 只能帮助你判断用户兴趣和雷点，不能替代规则校验。\n"
            "4. 如果规则判断和你的常识直觉冲突，以 Workspace 主动上下文为准。\n"
            "5. 如果某条内容是否 interesting 取决于规则校验结果，就先完成规则校验，再决定 mark_interesting 或 mark_not_interesting。\n"
            "6. 如果 Workspace 主动上下文不仅规定了结论标准，还规定了确认方式或确认来源，你必须按那个方式确认，不能换成你自己的猜测、记忆或随意搜索。\n"
            "7. 当当前候选条目已经有直接 URL 时，优先用 web_fetch 按直接来源确认；不要跳过直接来源确认。\n"
            "8. 「仅凭常识无法确认」中的「常识」不包含你的训练数据记忆。排名、赛况、阵容归属等实时变化的数据，你的训练知识已过时，不能用来代替规则要求的 web_fetch 验证。当 Workspace 主动上下文规定了时效性数据的 web_fetch 查询方式，该查询是必须步骤，不是可选项。\n\n"
            "【信息源规则】\n"
            "1. 主信息源只有本轮已提供的 Alerts / Content / Context。只有这些来源里的事实才能进入最终发送内容。\n"
            "2. 用户长期记忆、Workspace 主动上下文、recent_chat 只用于过滤、排序、同步规则、判断是否打扰；它们不是新的事实来源，也不是新的候选主题列表。\n"
            "3. Workspace 主动上下文的作用是同步主动 loop 与被动回复 loop 的运行规则，例如白名单、黑名单、关注范围、过滤条件、优先级；它提供规则，不提供本轮新闻事实。\n"
            "4. 即使 Workspace 主动上下文里出现了队伍名、选手名、游戏名、技术主题，也不能把这些名字直接当作本轮候选内容去展开、补全或脑补。\n"
            "5. 严禁根据长期记忆或 Workspace 主动上下文自行脑补具体新闻、比赛结果、转会、更新或其他外部事件。\n"
            "6. 当候选条目已自带来源 URL 时，先直接 web_fetch 该来源页面；不要凭记忆补细节，也不要跳过来源确认。\n"
            "7. 当本轮 alert 和 content 都为空时，你只有两条路：\n"
            "   a. finish_turn(decision=skip, reason=no_content)（默认，大多数情况选这条）\n"
            "   b. get_recent_chat → 若最近对话有自然延伸的未完成话题，可先 message_push 再 finish_turn(decision=reply) 轻松挑起对话；\n"
            "      此时 evidence 必须为空 []，消息里不得引用任何外部事件或可验证事实。\n"
            "   禁止在这两条路之外做任何事：不允许 recall_memory、不允许 get_content、\n"
            "   不允许 web_fetch、严禁捏造任何 item_id（包括 'feed:xxx' 格式）。\n"
            "   路径 b 是低概率选项——若 recent_chat 没有明显未完成话题，必须选 a。\n\n"
            "【决策流程】\n\n"
            "【Alert 快速路径】本轮如有 Alert：\n"
            "  → get_recent_chat 确认用户不在忙\n"
            "  → 把本轮所有 Alert 的内容整合成一条消息，evidence 必须填写本轮全部 Alert 的 id\n"
            "  → message_push → finish_turn(decision=reply) 结束\n"
            "  → 结束，可以不调用 recall_memory / mark_* / get_content / web_fetch\n\n"
            "【Content 路径】本轮无 Alert 时，Content 的主要任务不是做研究，而是把本轮候选逐条分成 interesting 或 not_interesting。\n"
            "Content 评估必须逐条进行，不能把不同主题的多条内容打包成一次统一判断。\n"
            "每条 Content 必须单独给出 mark_interesting 或 mark_not_interesting 结论，不能因为先评估的条目不感兴趣就跳过剩余条目直接 skip。\n"
            "你只能对本轮 Content 列表里真实存在的条目做 recall_memory / get_content / mark_*；不要对列表外的假想标题、假想比赛、假想转会或假想更新调用 recall_memory。\n"
            "只有当某一条内容本身与你已知的用户兴趣明显匹配时，才能把这一条标记为 interesting。\n"
            "如果一批条目里只有部分相关，必须只标记相关的那几条，其他条目继续判断或标记为 not_interesting。\n"
            "严禁因为其中 1-2 条命中兴趣，就把整批 item_ids 一次性 mark_interesting。\n"
            "调用 mark_interesting / mark_not_interesting 时，尽量附带一句简短 reason，说明是规则过滤、用户雷点、明显相关、边界验证失败或其他哪一种原因。\n"
            "reason 可以写得具体，方便观测；但如果 reason 中出现具体排名、Top N 结论、具体归属、具体日期等可验证事实，这些事实必须是你本轮按规则指定方式验证过的。\n"
            "如果还没完成验证，可以在 reason 里明确写“未验证”或“疑似”，但不要把未验证事实写成确定结论。\n\n"
            "推荐的最小流程（仅适用于 Content 路径，Alert 路径见上）：\n"
            "  1. 先看标题和来源，做快速初筛。\n"
            "  2. 用 recall_memory 判断这条内容是否可能是用户雷点，或是否可能让用户感兴趣。\n"
            "  3. 只有当条目看起来可能相关、或需要更多细节时，再调用 get_content。\n"
            "  4. web_fetch 只在必要时使用：当前候选已有直接 URL 时，先抓直接来源页面或正文；规则确认、细节核实都优先走它。\n"
            "     ⚠️ web_fetch 失败（404/超时/二进制图片）不能直接 mark_not_interesting；应退回 recall_memory 以 source/作者名为关键词判断用户兴趣。\n"
            "  5. 最终把每条内容分类为 mark_interesting 或 mark_not_interesting。\n"
            "  6. 所有条目分类完毕后：有 interesting → get_recent_chat 判断是否打扰 → message_push + finish_turn(decision=reply)；全部不感兴趣 → finish_turn(decision=skip, reason=no_content)\n"
            "  ⚠️ mark_* 不是终止动作，之后必须调 finish_turn\n\n"
            "Context-fallback（本轮允许且 alert/content 均无结果）：\n"
            "  context 数据已在上方，有亮点 → message_push + finish_turn(decision=reply)，否则 finish_turn(decision=skip, reason=no_content)\n\n"
            "【发送要求】\n"
            "- 语气自然，像朋友分享，不是推送通知\n"
            "- message_push 必须带非空 message；finish_turn(decision=skip, reason=...) 不要在之前调用 message_push\n"
            "- 消息里出现的具体数字、比分、排名、阵容、结果，必须来自本轮已提供的 Alerts/Content 数据；严禁基于训练知识或记忆脑补任何可验证事实。\n"
            "- 当某段内容基于外部来源且该来源有可靠链接时，在这段内容结束后自然附上对应原始链接，方便用户立即溯源\n"
            "- 链接要紧跟相关内容，不要把所有链接集中堆到整条消息末尾，也不要做成生硬的参考文献区\n"
            "- 如果一段内容对应多个来源，可以在该段后连续附上多个链接；没有可靠链接时不要强行补链接\n"
            "- 链接直接使用原始 url，不要杜撰、不要改写、不要省略协议头\n"
            "- evidence 格式：\"{ack_server}:{event_id}\"，如 \"feed:fmcp_abc123\"\n"
            "- 当本轮 content 和 alerts 均为空时，evidence 必须为 []；任何 'feed:xxx' 格式的 id 只能来自本轮真实提供的候选列表，不能自行捏造\n"
            "- 没有实质内容时 finish_turn(decision=skip, reason=no_content) 是正确选择\n\n"
            "【finish_turn.reason】no_content | user_busy | already_sent_similar | other"
        )

    def _render_alert_block(self, alerts: list[dict]) -> str:
        if not alerts:
            return ""
        lines = [
            normalize_alert(raw).to_prompt_line(index=i)
            for i, raw in enumerate(alerts, 1)
        ]
        return "【Alerts（时效性高，优先处理）】\n" + "\n".join(lines) + "\n\n"

    def _render_content_block(
        self, content_meta: list[dict], content_store: dict[str, str]
    ) -> str:
        if not content_meta:
            return ""
        lines: list[str] = []
        for i, raw in enumerate(content_meta, 1):
            contract = normalize_content(raw)
            has_content = bool(content_store.get(contract.item_id))
            lines.append(contract.to_prompt_line(index=i, has_content=has_content))
        return "【Content 列表（正文通过 get_content 按需获取）】\n" + "\n".join(lines) + "\n\n"

    def _render_context_block(self, context: list[dict]) -> str:
        if not context:
            return ""
        annotated_context = [normalize_context(item).to_prompt_item() for item in context]
        return (
            "【背景上下文】\n"
            "注：sleep_prob=睡眠概率，awake_prob=清醒概率（= 1 - sleep_prob）；"
            "若同时存在 `*_local` 与原始时间字段，判断早晚和相对时间时优先看 `*_local`，原始字段可能是 UTC。\n"
            + json.dumps(annotated_context, ensure_ascii=False)[:900]
            + "\n\n"
        )

    async def _run_loop(self, ctx: AgentTickContext) -> bool:
        """Agent loop（P5）。先调 DataGateway 预取数据，再启动 agent loop。"""
        if self._llm_fn is None:
            self.last_ctx = ctx
            return False

        # ── Gateway 预取 ──────────────────────────────────────────────────
        # 1. 先把 alerts / content / context 在 loop 外一次性预取完，
        #    避免模型在 loop 内自己反复拉源。
        gateway_deps = self._gateway_deps or GatewayDeps(
            alert_fn=None,
            feed_fn=None,
            context_fn=None,
            web_fetch_tool=self._tool_deps.web_fetch_tool,
            max_chars=self._tool_deps.max_chars,
            content_limit=self._cfg.agent_tick_content_limit,
        )
        gw = DataGateway(
            alert_fn=gateway_deps.alert_fn,
            feed_fn=gateway_deps.feed_fn,
            context_fn=gateway_deps.context_fn,
            web_fetch_tool=gateway_deps.web_fetch_tool,
            max_chars=gateway_deps.max_chars,
            content_limit=gateway_deps.content_limit,
        )
        gw_result = await gw.run()
        _log_content_candidates(gw_result)

        # 2. 把 gateway 的输入快照灌进 ctx。
        #    后续 tools、post-guard、ack 都只读 ctx，不再回头碰 gateway。
        ctx.mark_alerts_prefetched(gw_result.alerts)
        fetched_contents = [
            {
                "id": m["id"].split(":", 1)[1] if ":" in m["id"] else m["id"],
                "event_id": m["id"].split(":", 1)[1] if ":" in m["id"] else m["id"],
                "ack_server": m["id"].split(":", 1)[0],
                "title": m.get("title") or "",
                "source": m.get("source") or "",
                "url": m.get("url") or "",
                "published_at": m.get("published_at") or "",
            }
            for m in gw_result.content_meta
        ]
        ctx.mark_contents_prefetched(fetched_contents, gw_result.content_store)
        ctx.mark_context_prefetched(gw_result.context)

        # 2.5 快速 skip：无 alert、无 content、且 context_fallback 未开启时，
        #     直接跳过 LLM，避免空转。
        if not gw_result.alerts and not gw_result.content_meta and not ctx.context_as_fallback_open:
            if self._drift_runner is not None and self._cfg.drift_enabled:
                last_drift_at = self._state_store.get_last_drift_at(self._session_key)
                min_interval_hours = max(0, int(getattr(self._cfg, "drift_min_interval_hours", 0) or 0))
                if (
                    last_drift_at is not None
                    and min_interval_hours > 0
                    and (ctx.now_utc - last_drift_at).total_seconds() < min_interval_hours * 3600
                ):
                    logger.info(
                        "[proactive_v2] _run_loop: drift blocked by interval last_drift_at=%s min_interval_hours=%d",
                        last_drift_at.isoformat(),
                        min_interval_hours,
                    )
                    ctx.terminal_action = "skip"
                    ctx.skip_reason = "no_content"
                    self.last_ctx = ctx
                    return False
                logger.info("[proactive_v2] _run_loop: empty gateway result, attempting drift")
                entered_drift = await self._drift_runner.run(ctx, self._llm_fn)
                if entered_drift:
                    self._state_store.mark_drift_run(self._session_key, ctx.now_utc)
                    logger.info("[proactive_v2] _run_loop: drift entered, message_sent=%s", ctx.drift_message_sent)
                    self.last_ctx = ctx
                    return bool(ctx.drift_message_sent)
                logger.info("[proactive_v2] _run_loop: drift not entered")
            logger.info("[proactive_v2] _run_loop: no alerts/content and context_fallback=False → skip LLM")
            ctx.terminal_action = "skip"
            ctx.skip_reason = "no_content"
            self.last_ctx = ctx
            return False

        # 3. 构造本轮 proactive 专用 system prompt，把预取数据一次性注入给模型。
        system_msg = {"role": "system", "content": self._build_system_prompt(ctx, gw_result)}
        kickoff_msg = {
            "role": "user",
            "content": (
                "开始本轮 proactive 处理。"
                "请基于上面的候选内容和规则，必须通过工具逐步完成分类，"
                "最后通过 message_push + finish_turn(decision=reply)，或 finish_turn(decision=skip, reason=...) 收尾。"
            ),
        }
        messages: list[dict] = [system_msg, kickoff_msg]

        # 4. 主 loop：每轮允许模型自行决定是否调用工具。
        #    直到 finish_turn 写入 terminal_action，或达到步数上限。
        while ctx.steps_taken < self._cfg.agent_tick_max_steps:
            ok = await self._run_tool_step(
                messages,
                ctx,
                loop_tag="loop",
                tool_choice="auto",
            )
            if not ok:
                break

            if ctx.terminal_action is not None:
                break

        # ── Classification completeness check ─────────────────────────────
        # 若 agent 已 finish_skip 但仍有未分类 content 条目，重置并强制补完。
        # 这捕捉的场景：agent 评完部分条目后急着结束，剩余条目从未被评估。
        if ctx.terminal_action == "skip" and gw_result.content_meta:
            all_content_ids = {m["id"] for m in gw_result.content_meta}
            classified_ids = ctx.interesting_item_ids | ctx.discarded_item_ids
            unclassified_ids = all_content_ids - classified_ids
            if unclassified_ids:
                ctx.terminal_action = None
                ctx.skip_reason = ""
                ctx.skip_note = ""
                titles_hint = "; ".join(
                    f"{m['id']}（{m['title'][:40]}）"
                    for m in gw_result.content_meta
                    if m["id"] in unclassified_ids
                )
                completeness_msg = (
                    f"【系统提示】以下 {len(unclassified_ids)} 个条目尚未完成分类：\n"
                    f"{titles_hint}\n"
                    "请对每条调用 mark_interesting 或 mark_not_interesting，"
                    "全部分类完毕后再调用 message_push + finish_turn(decision=reply)，或 finish_turn(decision=skip, reason=...)。"
                )
                logger.info(
                    "[proactive_v2] completeness-check: %d unclassified items, resetting terminal_action → %s",
                    len(unclassified_ids),
                    sorted(unclassified_ids),
                )
                messages.append({"role": "user", "content": completeness_msg})
                for _ in range(5):
                    if ctx.terminal_action is not None or ctx.steps_taken >= self._cfg.agent_tick_max_steps:
                        break
                    ok = await self._run_tool_step(
                        messages,
                        ctx,
                        loop_tag="complete",
                    )
                    if not ok:
                        break

        # ── Reflection pass ───────────────────────────────────────────────
        # 5. 若 agent 已经把 interesting 标好了，但还没 finish_turn，
        #    就注入一条确定性反思提示，逼它在下一轮完成 reply/skip 收尾。
        if ctx.terminal_action is None and ctx.interesting_item_ids and ctx.steps_taken < self._cfg.agent_tick_max_steps:
            ids_str = ", ".join(sorted(ctx.interesting_item_ids))
            reflection = (
                f"【系统提示】你已将以下条目标记为 interesting：{ids_str}。\n"
                "所有条目均已分类完毕。你必须现在调用 message_push 撰写推送，然后调用 finish_turn(decision=reply)；"
                "或直接调用 finish_turn(decision=skip, reason=...)。不允许直接结束。"
            )
            logger.info("[proactive_v2] reflection: interesting=%d, injecting send prompt", len(ctx.interesting_item_ids))
            messages.append({"role": "user", "content": reflection})
            for _ in range(3):
                if ctx.terminal_action is not None or ctx.steps_taken >= self._cfg.agent_tick_max_steps:
                    break
                ok = await self._run_tool_step(
                    messages,
                    ctx,
                    loop_tag="reflect",
                    tool_choice="auto",
                )
                if not ok:
                    break

        self.last_ctx = ctx
        return ctx.terminal_action == "reply"

    async def _run_tool_step(
        self,
        messages: list[dict],
        ctx: AgentTickContext,
        *,
        loop_tag: str,
        tool_choice: str | dict = "auto",
        schemas: list[dict] | None = None,
    ) -> bool:
        # 1. 用当前 messages + TOOL_SCHEMAS 调一次模型，拿到本轮唯一的 tool call。
        active_schemas = schemas or TOOL_SCHEMAS
        llm_fn = self._llm_fn
        if llm_fn is None:
            return False
        tool_call = await llm_fn(messages, active_schemas, tool_choice)
        if tool_call is None:
            logger.warning(
                "[proactive_v2] %s: llm_fn returned None at step %d, stopping",
                loop_tag,
                ctx.steps_taken,
            )
            return False
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("input", {})
        arg_summary = json.dumps(tool_args, ensure_ascii=False)[:200]
        logger.info(
            "[proactive_v2] %s step %d: %s  args=%s",
            loop_tag,
            ctx.steps_taken,
            tool_name,
            arg_summary,
        )
        ctx.steps_taken += 1
        exec_result = await self._tool_executor.execute(
            ToolExecutionRequest(
                call_id=str(tool_call.get("id") or f"call_{ctx.steps_taken}"),
                tool_name=tool_name,
                arguments=tool_args,
                source="proactive",
                session_key=self._session_key,
            ),
            lambda name, args: dispatch(name, args, ctx, self._tool_deps),
        )
        if exec_result.status == "error":
            logger.warning("[proactive_v2] %s: tool error: %s", loop_tag, exec_result.output)
            result = str(exec_result.output)
            call_id = tool_call.get("id") or f"call_{ctx.steps_taken}"
            self._record_tick_step(
                ctx,
                phase=f"{loop_tag}:error",
                tool_name=tool_name,
                tool_call_id=str(call_id),
                tool_args=tool_args,
                tool_result_text=result,
            )
            self._append_tool_messages(
                messages,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=call_id,
                result=result,
            )
            return False
        result = str(exec_result.output)
        call_id = tool_call.get("id") or f"call_{ctx.steps_taken}"
        self._record_tick_step(
            ctx,
            phase=loop_tag,
            tool_name=tool_name,
            tool_call_id=str(call_id),
            tool_args=tool_args,
            tool_result_text=result,
        )
        # 3. 把 assistant tool_call + tool result 都回写到 messages。
        #    下一轮模型就能看到上一轮工具做了什么、返回了什么。
        self._append_tool_messages(
            messages,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=call_id,
            result=result,
        )
        return True

    @staticmethod
    def _append_tool_messages(
        messages: list[dict],
        *,
        tool_name: str,
        tool_args: dict,
        tool_call_id: str,
        result: str,
    ) -> None:
        messages.append(
            {
                "role": "assistant",
                "content": f"调用工具 {tool_name}",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args, ensure_ascii=False),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result,
            }
        )

    async def _post_loop(self, ctx: AgentTickContext) -> float:
        """收口到 TurnResult；发送与副作用交给 orchestrator。"""
        # 1. 先把 ctx 归并成 TurnResult（reply/skip、evidence、副作用）。
        result = await self._build_turn_result(ctx)
        self._record_tick_log_finish(ctx, result=result)
        if self._turn_orchestrator is None:
            raise RuntimeError("proactive turn_orchestrator is required")
        # 2. 再统一交给 TurnOrchestrator 落会话、发送消息、执行 side effects。
        await self._turn_orchestrator.handle_proactive_turn(
            result=result,
            session_key=self._session_key,
            channel=str(self._cfg.default_channel or "").strip(),
            chat_id=str(self._cfg.default_chat_id or "").strip(),
        )
        return 0.0

    async def _build_turn_result(self, ctx: AgentTickContext) -> TurnResult:
        ack_fn = self._tool_deps.ack_fn
        # 1. 如果最终不是 reply，直接构造成 skip，并只保留 skip 路径需要的副作用。
        if ctx.terminal_action != "reply":
            logger.info(
                "[proactive_v2] post-loop: action=%s steps=%d discarded=%d interesting=%d skip_reason=%s note=%s",
                ctx.terminal_action or "none",
                ctx.steps_taken,
                len(ctx.discarded_item_ids),
                len(ctx.interesting_item_ids),
                getattr(ctx, "skip_reason", ""),
                getattr(ctx, "skip_note", ""),
            )
            return TurnResult(
                decision="skip",
                outbound=None,
                trace=TurnTrace(
                    source="proactive",
                    extra={
                        "steps_taken": ctx.steps_taken,
                        "skip_reason": getattr(ctx, "skip_reason", ""),
                        "skip_note": getattr(ctx, "skip_note", ""),
                    },
                ),
                side_effects=[
                    _CallbackSideEffect(
                        callback=lambda: ack_discarded(ctx, ack_fn),
                        name="ack_discarded_skip",
                    )
                ],
            )

        # 2. 先做 delivery 去重：同一批来源内容短时间内不重复发。
        delivery_key = build_delivery_key(ctx)
        if self._state_store.is_delivery_duplicate(
            self._session_key, delivery_key, self._cfg.delivery_dedupe_hours
        ):
            logger.info("[proactive_v2] delivery_dedupe hit")
            return TurnResult(
                decision="skip",
                outbound=None,
                evidence=list(ctx.cited_item_ids),
                trace=TurnTrace(
                    source="proactive",
                    extra={
                        "steps_taken": ctx.steps_taken,
                        "skip_reason": "already_sent_similar",
                        "dedupe": "delivery",
                    },
                ),
                side_effects=[
                    _CallbackSideEffect(
                        callback=lambda: ack_post_guard_fail(
                            ctx, ack_fn, alert_ack_fn=self._tool_deps.alert_ack_fn
                        ),
                        name="ack_post_guard_delivery",
                    )
                ],
            )

        # 3. 再做 message 语义去重：新消息和最近主动消息如果实质重复，也跳过。
        if self._cfg.message_dedupe_enabled and self._deduper is not None:
            recent_proactive = (
                self._recent_proactive_fn()
                if self._recent_proactive_fn is not None
                else []
            )
            is_dup, reason = await self._deduper.is_duplicate(
                new_message=ctx.final_message,
                recent_proactive=recent_proactive,
                new_state_summary_tag="none",
            )
            if is_dup:
                logger.info("[proactive_v2] message_dedupe hit: %s", reason)
                return TurnResult(
                    decision="skip",
                    outbound=None,
                    evidence=list(ctx.cited_item_ids),
                    trace=TurnTrace(
                        source="proactive",
                        extra={
                            "steps_taken": ctx.steps_taken,
                            "skip_reason": "already_sent_similar",
                            "dedupe": "message",
                            "dedupe_note": str(reason or ""),
                        },
                    ),
                    side_effects=[
                        _CallbackSideEffect(
                            callback=lambda: ack_post_guard_fail(
                                ctx, ack_fn, alert_ack_fn=self._tool_deps.alert_ack_fn
                            ),
                            name="ack_post_guard_message",
                        )
                    ],
                )

        # 4. 两层 post-guard 都通过后，才真正产出 reply 类型 TurnResult。
        #    发送成功/失败后的状态更新和 ACK 都以 side effect 形式挂在这里。
        return TurnResult(
            decision="reply",
            outbound=TurnOutbound(session_key=self._session_key, content=ctx.final_message),
            evidence=list(ctx.cited_item_ids),
            trace=TurnTrace(
                source="proactive",
                extra={
                    "steps_taken": ctx.steps_taken,
                    "skip_reason": "",
                    "state_summary_tag": "none",
                },
            ),
            success_side_effects=[
                _CallbackSideEffect(
                    callback=lambda: _mark_delivery(
                        state_store=self._state_store,
                        session_key=self._session_key,
                        delivery_key=delivery_key,
                    ),
                    name="mark_delivery",
                ),
                _CallbackSideEffect(
                    callback=lambda: _mark_context_only_send(
                        state_store=self._state_store,
                        session_key=self._session_key,
                        context_as_fallback_open=ctx.context_as_fallback_open,
                        has_cited=bool(ctx.cited_item_ids),
                    ),
                    name="mark_context_only_send",
                ),
                _CallbackSideEffect(
                    callback=lambda: ack_on_success(
                        ctx,
                        ack_fn,
                        alert_ack_fn=self._tool_deps.alert_ack_fn,
                    ),
                    name="ack_on_success",
                ),
            ],
            failure_side_effects=[
                _CallbackSideEffect(
                    callback=lambda: ack_discarded(ctx, ack_fn),
                    name="ack_discarded_send_fail",
                )
            ],
        )

@dataclass
class _CallbackSideEffect:
    callback: Callable[[], Awaitable[None]]
    name: str = "callback"

    async def run(self) -> None:
        await self.callback()


async def _mark_delivery(*, state_store: Any, session_key: str, delivery_key: str) -> None:
    state_store.mark_delivery(session_key, delivery_key)


async def _mark_context_only_send(
    *,
    state_store: Any,
    session_key: str,
    context_as_fallback_open: bool,
    has_cited: bool,
) -> None:
    if context_as_fallback_open and not has_cited:
        state_store.mark_context_only_send(session_key)
