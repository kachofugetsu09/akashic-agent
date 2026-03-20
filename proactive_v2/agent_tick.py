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
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any, Callable

from proactive.config import ProactiveConfig
from proactive_v2.context import AgentTickContext
from proactive_v2.tools import TOOL_SCHEMAS, ToolDeps, execute

logger = logging.getLogger(__name__)

# ── ACK TTL 常量 ──────────────────────────────────────────────────────────

_CITED_ACK_TTL = 168       # cited content/alert → 168h
_UNCITED_ACK_TTL = 24      # interesting uncited → 24h
_POST_GUARD_ACK_TTL = 24   # delivery/message dedupe hit → 24h
_DISCARDED_ACK_TTL = 720   # mark_not_interesting → 720h


# ── 模块级 delivery key + ACK 函数 ───────────────────────────────────────

def build_delivery_key(ctx: AgentTickContext) -> str:
    """cited_item_ids 排序后 hash 为主键；为空时退化为消息文本 hash（context-fallback）。"""
    if ctx.cited_item_ids:
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


async def ack_post_guard_fail(ctx: AgentTickContext, ack_fn) -> None:
    """delivery_dedupe / message_dedupe 命中：
    content cited → 24h；alert cited → 不 ACK（§20）；
    uncited interesting（content）→ 24h；discarded → 720h。
    """
    if ack_fn is None:
        return
    fetched_alert_keys = {f"{e['ack_server']}:{e.get('event_id') or e.get('id', '')}" for e in ctx.fetched_alerts}
    cited_set = set(ctx.cited_item_ids)
    content_cited = cited_set - fetched_alert_keys
    for key in content_cited:
        await ack_fn(key, _POST_GUARD_ACK_TTL)
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
        sender: Any,
        deduper: Any,
        tool_deps: ToolDeps,
        llm_fn: Any | None = None,
        rng: Any | None = None,
        recent_proactive_fn: Callable[[], list] | None = None,
    ) -> None:
        self._cfg = cfg
        self._session_key = session_key
        self._state_store = state_store
        self._any_action_gate = any_action_gate
        self._last_user_at_fn = last_user_at_fn
        self._passive_busy_fn = passive_busy_fn
        self._sender = sender
        self._deduper = deduper
        self._tool_deps = tool_deps
        self._llm_fn = llm_fn
        self._rng = rng if rng is not None else _random_module.Random()
        self._recent_proactive_fn = recent_proactive_fn
        self.last_ctx: AgentTickContext | None = None  # 供测试检查

    async def tick(self) -> float | None:
        ctx = AgentTickContext(
            session_key=self._session_key,
            now_utc=datetime.now(timezone.utc),
        )

        # ── Pre-gate ──────────────────────────────────────────────────────

        # 5.1 passive_busy（系统硬 veto）
        if self._passive_busy_fn and self._passive_busy_fn(self._session_key):
            logger.debug("[proactive_v2] pre-gate: passive_busy → return None")
            return None

        # 5.2 delivery_cooldown
        if self._state_store.count_deliveries_in_window(
            self._session_key,
            self._cfg.agent_tick_delivery_cooldown_hours,
        ) > 0:
            logger.debug("[proactive_v2] pre-gate: delivery_cooldown → return None")
            return None

        # 5.3 AnyAction gate
        if self._any_action_gate is not None:
            should_act, meta = self._any_action_gate.should_act(
                now_utc=ctx.now_utc,
                last_user_at=self._last_user_at_fn(),
            )
            if not should_act:
                logger.debug("[proactive_v2] pre-gate: anyaction gate → return None meta=%s", meta)
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

        logger.info("[proactive_v2] tick: pre-gate passed, starting loop (context_fallback=%s)", ctx.context_as_fallback_open)
        await self._run_loop(ctx)
        return await self._post_loop(ctx)

    def _build_system_prompt(self, ctx: AgentTickContext) -> str:
        fallback_status = "允许" if ctx.context_as_fallback_open else "不允许"

        # 长期记忆注入（截断至 2000 chars）
        memory_block = ""
        if self._tool_deps.memory is not None:
            try:
                raw = self._tool_deps.memory.read_long_term().strip()
                if raw:
                    memory_block = (
                        "\n【用户记忆与偏好】\n"
                        + raw[:2000]
                        + "\n── 仅为快速参考；细粒度偏好在向量库中，务必通过 recall_memory 检索。\n"
                    )
            except Exception:
                pass

        return (
            "你是主动关怀型 AI 的决策核心，判断现在是否该给用户发一条消息。\n\n"
            "【当前状态】\n"
            f"- Context fallback 本轮：{fallback_status}\n"
            f"{memory_block}\n"
            "【优先级规则】Alert > Content > Context-fallback"
            "（仅本轮允许 且 alert/content 均无结果时才考虑）\n\n"
            "【各路径行为】\n\n"
            "Alert：\n"
            "  get_alert_events → 有 alert → [可选] get_context_data 补充背景 → send_message\n\n"
            "Content：\n"
            "  1. get_content_events（最多 5 条）\n"
            "  2. 对每条无条件调用 web_fetch(url) 阅读正文\n"
            "  3. 阅读后立即调用 recall_memory(标题+正文摘要)，确认偏好是否匹配、是否触碰雷点\n"
            "  4. 每条内容必须明确分类（不能留未分类）：\n"
            "     - 感兴趣 → mark_interesting\n"
            "     - 明确不感兴趣 → mark_not_interesting\n"
            "     ⚠️ 判断原则：recall_memory 无命中≠不感兴趣。用户没有明确记录的话题，若内容本身有价值，\n"
            "        倾向于标记 interesting 并推送，让用户自己判断。只有明确触碰雷点（如手机游戏、营销脚本）才 mark_not_interesting。\n"
            "  5. 所有条目分类完毕后：\n"
            "     - 有 interesting 条目 → 必须立即调用 send_message(text, cited_ids) 发送消息\n"
            "     - 无 interesting 条目 → skip(no_content)\n"
            "     ⚠️ mark_interesting / mark_not_interesting 不是终止动作，必须在之后调用 send_message 或 skip\n"
            "  6. get_recent_chat：判断用户当前是否真的在忙\n"
            "     忙碌信号：连续工作对话、正在处理紧急事项；无明显信号时不要以\u300c可能在忙\u300d为由跳过\n"
            "  7. 所有感兴趣条目聚合成一条消息 → send_message(text, cited_ids)\n\n"
            "Context-fallback（本轮允许 且 alert/content 均无结果时）：\n"
            "  get_context_data → 有亮点 → send_message / 否则 skip(no_content)\n\n"
            "【发送要求】\n"
            "- 消息语气自然，像朋友分享，不是推送通知\n"
            "- 没有实质内容时，skip 是正确选择，不要为发而发\n"
            "- web_fetch 失败时，可基于标题和记忆判断；失败本身不代表不感兴趣，不要 mark_not_interesting\n"
            "- cited_ids 只填实际引用的条目，格式：\"{ack_server}:{id}\"，如 \"feed-mcp:abc123\"\n\n"
            "【skip reason 枚举】no_content | user_busy | already_sent_similar | other"
        )

    async def _run_loop(self, ctx: AgentTickContext) -> float | None:
        """Agent loop（P5）。P6 将在此之后追加 post-guard + ACK + send。"""
        if self._llm_fn is None:
            self.last_ctx = ctx
            return 0.0

        system_msg = {"role": "system", "content": self._build_system_prompt(ctx)}
        messages: list[dict] = [system_msg]

        while ctx.steps_taken < self._cfg.agent_tick_max_steps:
            tool_call = await self._llm_fn(messages, TOOL_SCHEMAS)
            if tool_call is None:
                break

            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("input", {})
            # 打印工具名 + 关键参数（截断避免日志过长）
            _arg_summary = json.dumps(tool_args, ensure_ascii=False)[:200]
            logger.info("[proactive_v2] step %d: %s  args=%s", ctx.steps_taken, tool_name, _arg_summary)

            try:
                result = await execute(tool_name, tool_args, ctx, self._tool_deps)
            except ValueError as e:
                logger.warning("[proactive_v2] loop: tool error: %s", e)
                break

            # 追加工具调用和结果到 messages（OpenAI Chat Completions 格式）
            call_id = tool_call.get("id") or f"call_{ctx.steps_taken}"
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args, ensure_ascii=False),
                    },
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": result,
            })

            if ctx.terminal_action is not None:
                break

        # ── Reflection pass ───────────────────────────────────────────────
        # 若 agent 已标记 interesting 条目但忘记调用 send_message / skip，
        # 注入一条确定性提示，强制其完成终止动作（最多再跑 3 步）。
        if ctx.terminal_action is None and ctx.interesting_item_ids and ctx.steps_taken < self._cfg.agent_tick_max_steps:
            ids_str = ", ".join(sorted(ctx.interesting_item_ids))
            reflection = (
                f"【系统提示】你已将以下条目标记为 interesting：{ids_str}。\n"
                "所有条目均已分类完毕。你必须现在调用 send_message 撰写并发送推送，"
                "或调用 skip（若你认为不应发送）。不允许直接结束。"
            )
            logger.info("[proactive_v2] reflection: interesting=%d, injecting send prompt", len(ctx.interesting_item_ids))
            messages.append({"role": "user", "content": reflection})
            for _ in range(3):
                if ctx.terminal_action is not None or ctx.steps_taken >= self._cfg.agent_tick_max_steps:
                    break
                tool_call = await self._llm_fn(messages, TOOL_SCHEMAS)
                if tool_call is None:
                    break
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("input", {})
                _arg_summary = json.dumps(tool_args, ensure_ascii=False)[:200]
                logger.info("[proactive_v2] reflect step %d: %s  args=%s", ctx.steps_taken, tool_name, _arg_summary)
                try:
                    result = await execute(tool_name, tool_args, ctx, self._tool_deps)
                except ValueError as e:
                    logger.warning("[proactive_v2] reflect: tool error: %s", e)
                    break
                call_id = tool_call.get("id") or f"call_{ctx.steps_taken}"
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": call_id, "type": "function", "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args, ensure_ascii=False),
                    }}],
                })
                messages.append({"role": "tool", "tool_call_id": call_id, "content": result})

        self.last_ctx = ctx

    async def _post_loop(self, ctx: AgentTickContext) -> float:
        """Post-composition guards + send + ACK（P6）。"""
        ack_fn = self._tool_deps.ack_fn

        # skip 或无 terminal → 只 ACK discarded
        if ctx.terminal_action != "send":
            # 兜底：fetched 但未分类的条目自动 discard（720h），避免下次 tick 重复评估
            all_fetched_keys = {
                f"{e.get('ack_server', '')}:{e.get('event_id', '')}"
                for e in ctx.fetched_contents
                if e.get("ack_server") and e.get("event_id")
            }
            unclassified = all_fetched_keys - ctx.interesting_item_ids - ctx.discarded_item_ids
            if unclassified:
                ctx.discarded_item_ids.update(unclassified)
                logger.info("[proactive_v2] post-loop: auto-discarded %d unclassified items", len(unclassified))
            logger.info("[proactive_v2] post-loop: action=%s steps=%d discarded=%d interesting=%d skip_reason=%s note=%s",
                        ctx.terminal_action or "none", ctx.steps_taken, len(ctx.discarded_item_ids),
                        len(ctx.interesting_item_ids),
                        getattr(ctx, "skip_reason", ""), getattr(ctx, "skip_note", ""))
            await ack_discarded(ctx, ack_fn)
            return 0.0

        # ── Post-composition guards ───────────────────────────────────────

        delivery_key = build_delivery_key(ctx)

        # delivery_dedupe
        if self._state_store.is_delivery_duplicate(
            self._session_key, delivery_key, self._cfg.delivery_dedupe_hours
        ):
            logger.info("[proactive_v2] delivery_dedupe hit")
            await ack_post_guard_fail(ctx, ack_fn)
            return 0.0

        # message_dedupe
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
                await ack_post_guard_fail(ctx, ack_fn)
                return 0.0

        # ── Send ──────────────────────────────────────────────────────────

        send_ok = await self._sender.send(ctx.final_message)
        if not send_ok:
            await ack_discarded(ctx, ack_fn)
            return 0.0

        # ── Send success ──────────────────────────────────────────────────

        self._state_store.mark_delivery(self._session_key, delivery_key)

        # context-only 配额：cited_ids 为空说明是纯 context-fallback 路径
        if ctx.context_as_fallback_open and not ctx.cited_item_ids:
            self._state_store.mark_context_only_send(self._session_key)

        await ack_on_success(ctx, ack_fn, alert_ack_fn=self._tool_deps.alert_ack_fn)
        return 0.0
