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
from proactive_v2.gateway import DataGateway, GatewayResult
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
        result = await self._post_loop(ctx)
        ctx.content_store.clear()  # 清理 hashmap，防止内存泄漏
        return result

    def _build_system_prompt(self, ctx: AgentTickContext, gw: GatewayResult) -> str:
        fallback_status = "允许" if ctx.context_as_fallback_open else "不允许"

        memory_block = ""
        if self._tool_deps.memory is not None:
            try:
                raw = self._tool_deps.memory.read_long_term().strip()
                if raw:
                    memory_block = (
                        "\n【用户长期记忆（快速参考）】\n"
                        + raw[:2000]
                        + "\n── 细粒度偏好请用 recall_memory 检索。\n"
                    )
            except Exception:
                pass

        alert_block = ""
        if gw.alerts:
            lines = []
            for i, a in enumerate(gw.alerts):
                aid = f"{a.get('ack_server','?')}:{a.get('event_id') or a.get('id','?')}"
                lines.append(f"  [{i+1}] id={aid}  {a.get('title','')}")
            alert_block = "【Alerts（时效性高，优先处理）】\n" + "\n".join(lines) + "\n\n"

        context_block = ""
        if gw.context:
            context_block = (
                "【背景上下文】\n"
                + json.dumps(gw.context, ensure_ascii=False)[:800]
                + "\n\n"
            )

        content_block = ""
        if gw.content_meta:
            lines = []
            for i, m in enumerate(gw.content_meta):
                has_content = bool(gw.content_store.get(m["id"]))
                status = "✓" if has_content else "✗(预取失败)"
                lines.append(
                    f"  [{i+1}] id={m['id']}\n"
                    f"       title={m['title']}\n"
                    f"       source={m['source']}  正文:{status}"
                )
            content_block = (
                "【Content 列表（正文通过 get_content 按需获取）】\n"
                + "\n".join(lines)
                + "\n\n"
            )

        return (
            "你是主动关怀型 AI 的决策核心，判断现在是否该给用户发一条消息。\n"
            "数据已预取完毕，基于下方数据直接决策。\n\n"
            f"{alert_block}"
            f"{content_block}"
            f"{context_block}"
            f"{memory_block}\n"
            f"【优先级】Alert > Content > Context-fallback（本轮：{fallback_status}）\n\n"
            "【决策流程】\n\n"
            "Alert（若有）→ 直接 send_message\n\n"
            "Content如果recall不能回答用户对该内容的偏好：\n"
            "  1. recall_memory × 2（负向雷点假设 + 正向兴趣假设）\n"
            "     - 负向 query：「用户对《标题》完全不感兴趣甚至厌恶」\n"
            "       命中雷点 → mark_not_interesting，不读正文\n"
            "     - 正向 query：「用户对《标题》很感兴趣」\n"
            "       有信号 → 继续读正文\n"
            "     - 无命中 ≠ 不感兴趣，结合标题和来源常识判断\n"
            "  2. 对感兴趣条目批量 get_content([id,...]) 读正文\n"
            "     正文为空（预取失败）→ 可用 web_fetch 降级，或凭标题判断\n"
            "  3. 读完正文后最终分类：mark_interesting / mark_not_interesting\n"
            "  4. 所有条目分类完毕：\n"
            "     有 interesting → get_recent_chat 判断是否打扰 → send_message\n"
            "     全部不感兴趣 → skip(no_content)\n"
            "  ⚠️ mark_* 不是终止动作，之后必须调 send_message 或 skip\n\n"
            "Context-fallback（本轮允许且 alert/content 均无结果）：\n"
            "  context 数据已在上方，有亮点 → send_message，否则 skip\n\n"
            "【发送要求】\n"
            "- 语气自然，像朋友分享，不是推送通知\n"
            "- cited_ids 格式：\"{ack_server}:{event_id}\"，如 \"feed:fmcp_abc123\"\n"
            "- 没有实质内容时 skip 是正确选择\n\n"
            "【skip reason】no_content | user_busy | already_sent_similar | other"
        )

    async def _run_loop(self, ctx: AgentTickContext) -> float | None:
        """Agent loop（P5）。先调 DataGateway 预取数据，再启动 agent loop。"""
        if self._llm_fn is None:
            self.last_ctx = ctx
            return 0.0

        # ── Gateway 预取 ──────────────────────────────────────────────────
        gw = DataGateway(
            alert_fn=self._tool_deps.alert_fn,
            feed_fn=self._tool_deps.feed_fn,
            context_fn=self._tool_deps.context_fn,
            web_fetch_tool=self._tool_deps.web_fetch_tool,
            max_chars=self._tool_deps.max_chars,
            content_limit=self._cfg.agent_tick_content_limit,
        )
        gw_result = await gw.run()

        # 填充 ctx（供 ACK 路径使用）
        ctx.fetched_alerts = gw_result.alerts
        ctx.fetched_contents = [
            {"ack_server": m["id"].split(":", 1)[0], "event_id": m["id"].split(":", 1)[1] if ":" in m["id"] else m["id"]}
            for m in gw_result.content_meta
        ]
        ctx.fetched_context = gw_result.context
        ctx.content_store = gw_result.content_store

        system_msg = {"role": "system", "content": self._build_system_prompt(ctx, gw_result)}
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
