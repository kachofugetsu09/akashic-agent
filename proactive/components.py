from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from core.memory.port import MemoryPort

from agent.persona import AKASHIC_IDENTITY, PERSONALITY_RULES
from agent.provider import LLMProvider, LLMResponse
from agent.tool_bundles import build_fitbit_tools, build_readonly_research_tools
from agent.tool_runtime import (
    append_assistant_tool_calls,
    append_tool_result,
    prepare_toolset,
    tool_call_signature,
)
from agent.tools.base import Tool
from agent.tools.message_push import MessagePushTool
from core.net.http import get_default_http_requester
from feeds.base import FeedItem
from proactive.json_utils import extract_json_object
from proactive.presence import PresenceStore
from proactive.state import ProactiveStateStore
from session.manager import SessionManager

logger = logging.getLogger(__name__)

_TOOL_LOOP_REPEAT_LIMIT = 3
_SUMMARY_MAX_TOKENS = 384


def build_proactive_memory_query(
    *,
    items: list[FeedItem],
    recent: list[dict],
    decision_signals: dict[str, object],
    is_crisis: bool,
    max_items: int = 3,
    max_recent: int = 3,
) -> str:
    """Build a compact query for proactive memory retrieval."""
    lines: list[str] = ["主动触达主题"]

    lines.append("候选内容：")
    for item in items[: max(1, int(max_items))]:
        title = (item.title or "").strip() or "(无标题)"
        snippet = re.sub(r"\s+", " ", (item.content or "").strip())[:120]
        source = (item.source_name or "").strip()
        source_type = (item.source_type or "").strip().lower()
        source_key = f"{source_type}:{source.lower()}" if source_type or source else ""
        domain = ""
        if item.url:
            try:
                domain = (urlsplit(item.url).netloc or "").strip().lower()
            except Exception:
                domain = ""
        lines.append(f"- {title}" + (f"（{source}）" if source else ""))
        if source_key:
            lines.append(f"  来源标签: {source_key}")
        if domain:
            lines.append(f"  来源域名: {domain}")
        if snippet:
            lines.append(f"  {snippet}")

    lines.append("近期对话：")
    for msg in recent[-max(1, int(max_recent)) :]:
        role = "用户" if str(msg.get("role", "")) == "user" else "助手"
        text = re.sub(r"\s+", " ", str(msg.get("content", "")).strip())[:120]
        if text:
            lines.append(f"- {role}: {text}")

    health_events = decision_signals.get("health_events")
    if isinstance(health_events, list) and health_events:
        lines.append("健康事件：")
        for event in health_events[:2]:
            message = re.sub(
                r"\s+",
                " ",
                str((event or {}).get("message", "")).strip(),
            )[:120]
            if message:
                lines.append(f"- {message}")

    lines.append("触达目标：")
    lines.append("- 基于当前内容生成自然主动消息")
    lines.append("- 优先遵循用户偏好与过往事件")
    if is_crisis:
        lines.append("- 当前是重连场景，优先自然开场")
    return "\n".join(lines)


def _build_optional_fitbit_tool_runtime(
    fitbit_tools: list[Tool],
) -> tuple[list[dict[str, Any]], dict[str, Tool]]:
    if not fitbit_tools:
        return [], {}
    prepared = prepare_toolset(fitbit_tools)
    return prepared.schemas, prepared.tool_map


def _resolve_active_tool_runtime(
    *,
    base_schemas: list[dict[str, Any]],
    base_tool_map: dict[str, Tool],
    include_fitbit: bool,
    fitbit_schemas: list[dict[str, Any]],
    fitbit_tool_map: dict[str, Tool],
) -> tuple[list[dict[str, Any]], dict[str, Tool]]:
    active_tool_schemas = list(base_schemas)
    active_tool_map = dict(base_tool_map)
    if include_fitbit and fitbit_schemas:
        active_tool_schemas.extend(fitbit_schemas)
        active_tool_map.update(fitbit_tool_map)
    return active_tool_schemas, active_tool_map


async def _execute_tool_calls(
    *,
    messages: list[dict[str, Any]],
    tool_calls: list[Any],
    active_tool_map: dict[str, Tool],
    log_prefix: str,
    log_each_call: bool = False,
    args_preview_chars: int = 80,
    result_preview_chars: int = 100,
) -> None:
    for tc in tool_calls:
        tool = active_tool_map.get(tc.name)
        if tool:
            if log_each_call:
                logger.info(
                    "%s 调用工具 %s args=%s",
                    log_prefix,
                    tc.name,
                    str(tc.arguments)[:args_preview_chars],
                )
            result = await tool.execute(**tc.arguments)
            if log_each_call:
                logger.info(
                    "%s 工具结果 %s: %s",
                    log_prefix,
                    tc.name,
                    result[:result_preview_chars],
                )
        else:
            result = f"未知工具：{tc.name}"
        append_tool_result(messages, tool_call_id=tc.id, content=result)


def _format_recent_proactive_entries(recent_proactive: list[object]) -> str:
    def field(raw: object, name: str, default: object = "") -> object:
        if isinstance(raw, dict):
            return raw.get(name, default)
        return getattr(raw, name, default)

    lines: list[str] = []
    for i, msg in enumerate(recent_proactive, 1):
        content = str(field(msg, "content", "") or "").strip()
        if not content:
            continue
        tag = str(field(msg, "state_summary_tag", "none") or "none")
        ts = field(msg, "timestamp", None)
        ts_text = ""
        if ts is not None:
            try:
                ts_text = ts.isoformat()
            except Exception:
                ts_text = str(ts)
        source_refs = field(msg, "source_refs", []) or []
        source_bits: list[str] = []
        for raw in source_refs[:1]:
            source_name = str(field(raw, "source_name", "") or "").strip()
            title = str(field(raw, "title", "") or "").strip()
            url = str(field(raw, "url", "") or "").strip()
            parts = [p for p in [source_name, title, url] if p]
            if parts:
                source_bits.append(" | ".join(parts))
        meta = []
        if ts_text:
            meta.append(f"time={ts_text}")
        if tag and tag != "none":
            meta.append(f"state_tag={tag}")
        if source_bits:
            meta.append(f"source={source_bits[0]}")
        meta_text = f" ({'; '.join(meta)})" if meta else ""
        lines.append(f"[{i}]{meta_text} {content}")
    return "\n---\n".join(lines)


@dataclass
class ReflectHooks:
    format_items: Callable[[list[FeedItem]], str]
    format_recent: Callable[[list[dict]], str]
    parse_decision: Callable[[str], Any]
    collect_global_memory: Callable[[], str]
    sample_random_memory: Callable[[int], list[str]]
    target_session_key: Callable[[], str]
    on_reflect_error: Callable[[Exception], Any]


@dataclass(frozen=True)
class ProactivePromptContext:
    now_str: str
    now_iso: str
    feed_text: str
    chat_text: str
    memory_text: str


def _build_proactive_prompt_context(
    *,
    items: list[FeedItem],
    recent: list[dict],
    format_items: Callable[[list[FeedItem]], str],
    format_recent: Callable[[list[dict]], str],
    collect_global_memory: Callable[[], str],
) -> ProactivePromptContext:
    now = datetime.now().astimezone()
    return ProactivePromptContext(
        now_str=now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        now_iso=now.isoformat(),
        feed_text=format_items(items) or "（暂无订阅内容）",
        chat_text=format_recent(recent) or "（无近期对话记录）",
        memory_text=collect_global_memory(),
    )


class ProactiveReflector:
    def __init__(
        self,
        *,
        provider: LLMProvider,
        model: str,
        max_tokens: int,
        cfg: Any,
        memory_store: "MemoryPort | None",
        presence: PresenceStore | None,
        hooks: ReflectHooks,
        fitbit_url: str = "",
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens
        self._cfg = cfg
        self._memory = memory_store
        self._presence = presence
        self._hooks = hooks
        self._fitbit_url = fitbit_url
        self._fitbit_tools: list[Tool] = build_fitbit_tools(
            fitbit_url=fitbit_url,
            requester=get_default_http_requester("local_service"),
        )
        self._fitbit_tool_schemas, self._fitbit_tool_map = (
            _build_optional_fitbit_tool_runtime(self._fitbit_tools)
        )

    async def reflect(
        self,
        items: list[FeedItem],
        recent: list[dict],
        energy: float = 0.0,
        urge: float = 0.0,
        is_crisis: bool = False,
        decision_signals: dict[str, object] | None = None,
        retrieved_memory_block: str = "",
    ) -> Any:
        prompt_context = _build_proactive_prompt_context(
            items=items,
            recent=recent,
            format_items=self._hooks.format_items,
            format_recent=self._hooks.format_recent,
            collect_global_memory=self._hooks.collect_global_memory,
        )

        self_text = ""
        now_ongoing_text = ""
        if self._memory:
            try:
                self_text = self._memory.read_self().strip()
            except Exception:
                pass
            try:
                now_ongoing_text = self._memory.read_now_ongoing().strip()
            except Exception:
                pass

        crisis_hint = ""
        if is_crisis:
            topic_chunks = self._hooks.sample_random_memory(1)
            topic_hint = topic_chunks[0] if topic_chunks else ""
            session_key = self._hooks.target_session_key()
            last_at = (
                self._presence.get_last_user_at(session_key) if self._presence else None
            )
            elapsed = ""
            if last_at:
                hours = (datetime.now(timezone.utc) - last_at).total_seconds() / 3600
                elapsed = f"距离上次对话已超过 {hours:.0f} 小时。"
            topic_section = (
                f"\n\n## 随机话题建议（危机开场用）\n\n{topic_hint}"
                if topic_hint
                else ""
            )
            crisis_hint = (
                f"\n[危机模式] {elapsed}"
                "用户可能已忘记你的存在，需要主动找一个自然的切入点重新联系。"
                "可以从下方随机话题出发，或用关心/有趣内容开场。"
                f"{topic_section}"
            )

        system_msg = (
            "你是 Akashic，正在决定是否主动联系你的用户。"
            "你了解用户订阅的信息流和最近的对话内容。"
            "你的目标是在恰当的时机出现，而不是频繁打扰。"
            "\n\n## 身份\n"
            f"{AKASHIC_IDENTITY}"
            "\n\n## 性格\n"
            f"{PERSONALITY_RULES}"
            + (f"\n\n## 自我认知\n\n{self_text}" if self_text else "")
        )
        user_msg = f"""当前时间：{prompt_context.now_str}
（ISO格式：{prompt_context.now_iso}）

## 主动性上下文

当前电量（与用户的互动新鲜度）: {energy:.2f}  (0=完全冷却, 1=刚刚对话)
主动冲动指数: {urge:.2f}  (0=不需要说, 1=非常需要联系){crisis_hint}
{f"## 决策信号（系统计算）\n\n```json\n{json.dumps(decision_signals, ensure_ascii=False, indent=2)}\n```\n" if decision_signals else ""}

## 订阅信息流（最新内容）

{prompt_context.feed_text}

## 长期记忆（用户画像/偏好）

{prompt_context.memory_text}
{f"## 相关记忆（本次触达召回）\n\n{retrieved_memory_block}\n" if retrieved_memory_block else ""}
{f"## 用户近期状态\n\n{now_ongoing_text}\n" if now_ongoing_text else ""}
## 近期对话

{prompt_context.chat_text}

## 任务

综合以上信息，判断是否值得主动联系用户。考虑：
- 信息流里有没有用户可能感兴趣的内容
- 现在说点什么是否自然、不唐突
- 与近期对话有无关联或延伸；但这只是加分项，不是前置条件
- 若某条内容与用户长期兴趣高度匹配，即使和近期对话无关，也可以自然开启一个新话题
- 若有多条候选内容，一次只围绕一个最值得说的主题，不要把多条资讯拼成摘要
- 电量越低越需要主动联系，危机模式时哪怕简单关心也有价值
- 若存在 health_events，优先考虑健康提醒；可调用 fitbit_health_snapshot 校验当前实时状态（注意 data_lag_min 判断数据是否新鲜）
- 若不存在 health_events，禁止在消息正文中引用具体健康数值（心率、血氧等）
- 提及健康时只转述 health_events[*].message，不编造数值
- 若最近主动消息已经表达过对用户当前处境的总结、安慰或判断，而用户此后还没有回复，则新消息禁止重复这一层；若本次只是新资讯，直接进入新内容

只输出 JSON，不要其他内容：
{{
  "reasoning": "内心独白（不会显示给用户，说清楚你的判断依据）",
  "score": 0.0,
  "should_send": false,
  "message": "",
  "evidence_item_ids": []
}}

score 说明：0.0=完全没必要  0.5=有点想说  0.7=比较值得  1.0=非常值得立刻说
只有在你能指出消息依据的确切证据时，should_send 才能为 true：
- 若引用信息流，必须能在上方“订阅信息流”里定位到对应条目
- 若要提来源，必须使用条目里真实出现的来源名 / 原文链接，禁止编造“官方群/官网消息源/朋友转述”等来源
- 若你找不到确切证据或来源不清，应降低 score，并把 should_send 设为 false
message 若 should_send=true，写要发给用户的话（口语化，不要像系统通知）
写 message 时必须直接表达你的判断/观点，不要把结尾写成征求用户看法的反问句。
禁止使用“你怎么看/你觉得呢/你怎么想/要不要我继续”这类收尾。
除非必须让用户做明确选择（如确认日程、权限、付款），否则不要主动提问。
若 message 引用了 RSS / 网页信息流里的具体内容，优先自然带上“来源名 + 可点击原文链接”；若链接过长，可只保留一个最关键 URL。
对非网页来源（如 novel-kb）不要伪造外链；若没有确切 URL，就只写来源名，不要编造链接。
系统不会在发送前替你自动补来源，所以你在 message 里写出的来源信息必须是最终版本。
如果 message 引用了信息流，请在 evidence_item_ids 中只返回 1 个最关键的 item_id；没有引用时可为空数组。"""

        try:
            active_tools, active_tool_map = _resolve_active_tool_runtime(
                base_schemas=[],
                base_tool_map={},
                include_fitbit=bool(
                    decision_signals and decision_signals.get("health_events")
                ),
                fitbit_schemas=self._fitbit_tool_schemas,
                fitbit_tool_map=self._fitbit_tool_map,
            )

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
            resp: LLMResponse | None = None
            for _ in range(2):
                resp = await self._provider.chat(
                    messages=messages,
                    tools=active_tools,
                    model=self._model,
                    max_tokens=self._max_tokens,
                )
                tool_calls = getattr(resp, "tool_calls", []) or []
                if not tool_calls:
                    break

                append_assistant_tool_calls(
                    messages,
                    content=resp.content,
                    tool_calls=tool_calls,
                )
                await _execute_tool_calls(
                    messages=messages,
                    tool_calls=tool_calls,
                    active_tool_map=active_tool_map,
                    log_prefix="[proactive]",
                )

            if resp is None:
                return self._hooks.parse_decision("")
            content = resp.content or ""
            logger.info("[proactive] LLM 原始输出预览: %r", content[:240])
            return self._hooks.parse_decision(content)
        except Exception as e:
            logger.error(f"[proactive] LLM 反思失败: {e}")
            return self._hooks.on_reflect_error(e)


class ProactiveSender:
    def __init__(
        self,
        *,
        cfg: Any,
        push_tool: MessagePushTool,
        sessions: SessionManager,
        presence: PresenceStore | None,
    ) -> None:
        self._cfg = cfg
        self._push = push_tool
        self._sessions = sessions
        self._presence = presence

    async def send(self, message: str, meta: Any | None = None) -> bool:
        message = (message or "").strip()
        channel = (self._cfg.default_channel or "").strip()
        chat_id = self._cfg.default_chat_id.strip()
        if not channel or not chat_id:
            logger.warning(
                "[proactive] default_channel/default_chat_id 未配置，跳过发送"
            )
            return False
        logger.info(
            "[proactive] 准备发送主动消息 channel=%s chat_id=%s message_len=%d",
            channel,
            chat_id,
            len(message),
        )
        try:
            result = await self._push.execute(
                channel=channel,
                chat_id=chat_id,
                message=message,
            )
            logger.info("[proactive] message_push 返回: %r", result[:200])
            if "已发送" not in result:
                logger.warning(f"[proactive] 发送未成功: {result}")
                return False
            key = f"{channel}:{chat_id}"
            session = self._sessions.get_or_create(key)
            source_refs_payload: list[dict[str, Any]] = []
            evidence_item_ids: list[str] = []
            state_summary_tag = "none"
            if meta is not None:
                evidence_item_ids = [
                    str(x)
                    for x in (getattr(meta, "evidence_item_ids", None) or [])
                    if str(x).strip()
                ]
                state_summary_tag = str(
                    getattr(meta, "state_summary_tag", "none") or "none"
                )
                for raw in getattr(meta, "source_refs", None) or []:
                    source_refs_payload.append(
                        {
                            "item_id": str(getattr(raw, "item_id", "") or ""),
                            "source_type": str(getattr(raw, "source_type", "") or ""),
                            "source_name": str(getattr(raw, "source_name", "") or ""),
                            "title": str(getattr(raw, "title", "") or ""),
                            "url": getattr(raw, "url", None),
                            "published_at": getattr(raw, "published_at", None),
                        }
                    )
            session.add_message(
                "assistant",
                message,
                proactive=True,
                tools_used=["message_push"],
                evidence_item_ids=evidence_item_ids,
                source_refs=source_refs_payload,
                state_summary_tag=state_summary_tag,
            )
            await self._sessions.save_async(session)
            if self._presence:
                self._presence.record_proactive_sent(key)
            logger.info(f"[proactive] 已发送主动消息并写入会话 → {channel}:{chat_id}")
            return True
        except Exception as e:
            logger.error(f"[proactive] 发送失败: {e}")
            return False


class ProactiveItemFilter:
    def __init__(
        self,
        *,
        cfg: Any,
        state: ProactiveStateStore,
        source_key_fn: Callable[[FeedItem], str],
        item_id_fn: Callable[[FeedItem], str],
        semantic_text_fn: Callable[[FeedItem, int], str],
        build_tfidf_vectors_fn: Callable[[list[str], int], list[dict[str, float]]],
        cosine_fn: Callable[[dict[str, float], dict[str, float]], float],
    ) -> None:
        self._cfg = cfg
        self._state = state
        self._source_key = source_key_fn
        self._item_id = item_id_fn
        self._semantic_text = semantic_text_fn
        self._build_tfidf_vectors = build_tfidf_vectors_fn
        self._cosine = cosine_fn

    def filter_new_items(
        self, items: list[FeedItem]
    ) -> tuple[list[FeedItem], list[tuple[str, str]], list[tuple[str, str]]]:
        if not items:
            logger.info("[proactive] 本轮无 item，去重过滤跳过")
            return [], [], []
        now = datetime.now(timezone.utc)
        source_fresh: list[FeedItem] = []
        source_entries: list[tuple[str, str]] = []
        cooldown_hours = getattr(self._cfg, "llm_reject_cooldown_hours", 0)
        for item in items:
            source_key = self._source_key(item)
            item_id = self._item_id(item)
            seen = self._state.is_item_seen(
                source_key=source_key,
                item_id=item_id,
                ttl_hours=self._cfg.dedupe_seen_ttl_hours,
                now=now,
            )
            logger.debug(
                "[proactive] item 去重检查 source=%s item_id=%s seen=%s title=%r",
                source_key,
                item_id[:16],
                seen,
                (item.title or "")[:60],
            )
            if seen:
                continue
            # LLM 拒绝冷却检查（独立短 TTL，不影响 seen_items 的 14天去重）
            if cooldown_hours > 0 and self._state.is_rejection_cooled(
                source_key=source_key,
                item_id=item_id,
                ttl_hours=cooldown_hours,
                now=now,
            ):
                logger.debug(
                    "[proactive] rejection_cooldown 跳过 source=%s item_id=%s ttl_hours=%d",
                    source_key,
                    item_id[:16],
                    cooldown_hours,
                )
                continue
            source_fresh.append(item)
            source_entries.append((source_key, item_id))
        if not self._cfg.semantic_dedupe_enabled or not source_fresh:
            return source_fresh, source_entries, []
        return self._semantic_dedupe(source_fresh, source_entries, now)

    def _semantic_dedupe(
        self,
        source_fresh: list[FeedItem],
        source_entries: list[tuple[str, str]],
        now: datetime,
    ) -> tuple[list[FeedItem], list[tuple[str, str]], list[tuple[str, str]]]:
        history = self._state.get_semantic_items(
            window_hours=self._cfg.semantic_dedupe_window_hours,
            max_candidates=self._cfg.semantic_dedupe_max_candidates,
            now=now,
        )
        payload = [
            {
                "item": item,
                "source_key": source_key,
                "item_id": item_id,
                "text": self._semantic_text(
                    item, self._cfg.semantic_dedupe_text_max_chars
                ),
            }
            for item, (source_key, item_id) in zip(source_fresh, source_entries)
        ]
        if not payload:
            return [], [], []
        docs = [h["text"] for h in history] + [p["text"] for p in payload]
        vectors = self._build_tfidf_vectors(docs, self._cfg.semantic_dedupe_ngram)
        history_vectors = vectors[: len(history)]
        payload_vectors = vectors[len(history) :]

        keep_items: list[FeedItem] = []
        keep_entries: list[tuple[str, str]] = []
        duplicate_entries: list[tuple[str, str]] = []
        accepted_vectors: list[dict[str, float]] = []
        accepted_meta: list[dict[str, str]] = []
        threshold = self._cfg.semantic_dedupe_threshold
        for idx, p in enumerate(payload):
            vec = payload_vectors[idx]
            best_sim = 0.0
            best_kind = ""
            best_source = ""
            best_item_id = ""

            for h_idx, h_vec in enumerate(history_vectors):
                sim = self._cosine(vec, h_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_kind = "history"
                    best_source = history[h_idx].get("source_key", "")
                    best_item_id = history[h_idx].get("item_id", "")

            for a_idx, a_vec in enumerate(accepted_vectors):
                sim = self._cosine(vec, a_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_kind = "batch"
                    best_source = accepted_meta[a_idx].get("source_key", "")
                    best_item_id = accepted_meta[a_idx].get("item_id", "")

            logger.debug(
                "[proactive] 语义去重检查 source=%s item_id=%s best_sim=%.4f threshold=%.2f matched_kind=%s matched_source=%s matched_item=%s title=%r",
                p["source_key"],
                p["item_id"][:16],
                best_sim,
                threshold,
                best_kind or "-",
                best_source or "-",
                (best_item_id or "-")[:16],
                (p["item"].title or "")[:80],
            )

            if best_sim >= threshold:
                duplicate_entries.append((p["source_key"], p["item_id"]))
                logger.debug(
                    "[proactive] 语义去重命中，过滤 item source=%s item_id=%s sim=%.4f",
                    p["source_key"],
                    p["item_id"][:16],
                    best_sim,
                )
                continue

            keep_items.append(p["item"])
            keep_entries.append((p["source_key"], p["item_id"]))
            accepted_vectors.append(vec)
            accepted_meta.append(
                {"source_key": p["source_key"], "item_id": p["item_id"]}
            )
        logger.debug(
            "[proactive] 语义去重结果 keep=%d duplicate=%d history_candidates=%d",
            len(keep_items),
            len(duplicate_entries),
            len(history),
        )
        return keep_items, keep_entries, duplicate_entries


class ProactiveFeatureScorer:
    """AI 特征打分器：仅输出固定特征分（0~1），不做最终决策。"""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        model: str,
        max_tokens: int,
        format_items: Callable[[list[FeedItem]], str],
        format_recent: Callable[[list[dict]], str],
        collect_global_memory: Callable[[], str],
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens
        self._format_items = format_items
        self._format_recent = format_recent
        self._collect_global_memory = collect_global_memory

    async def score_features(
        self,
        *,
        items: list[FeedItem],
        recent: list[dict],
        decision_signals: dict[str, object],
        retrieved_memory_block: str = "",
    ) -> dict[str, float | str]:
        prompt_context = _build_proactive_prompt_context(
            items=items,
            recent=recent,
            format_items=self._format_items,
            format_recent=self._format_recent,
            collect_global_memory=self._collect_global_memory,
        )
        system_msg = (
            "你是主动触达特征评估器。只输出固定JSON字段。"
            "每个分数字段必须是0到1的小数；同时给每个字段一句简短理由。"
            "若决策信号含 health_events，健康相关触达优先级高于普通资讯触达。"
            "message_readiness_reason 应基于用户整体状态（时间、活跃度、对话节奏等）综合判断，无需引用具体健康数值。"
            "若决策信号不含 health_events，不得用健康状况作为触达理由。"
            "topic_continuity 代表与近期对话的连续性，它是加分项而不是硬门槛。"
            "如果订阅内容与用户长期兴趣明显匹配，即使与近期对话无关，也可以给出高 interest_match 和合理的 reconnect_value。"
            "不要给最终决策，不要输出额外文本。"
        )
        user_msg = f"""当前时间：{prompt_context.now_str}

## 决策信号（系统计算）
```json
{json.dumps(decision_signals, ensure_ascii=False, indent=2)}
```

## 订阅信息流
{prompt_context.feed_text}

## 长期记忆
{prompt_context.memory_text}
{f"## 相关记忆（本次触达召回）\n{retrieved_memory_block}\n" if retrieved_memory_block else ""}

## 近期对话
{prompt_context.chat_text}

补充原则：
- 不要求消息必须承接近期对话
- 若某条信息流与用户长期兴趣明显匹配，可把它视为一个自然的新切入点
- 只有当完全无关、打扰风险高、或内容不够新鲜时，才应压低分数

只输出 JSON，且仅包含以下键：
{{
  "topic_continuity": 0.0,
  "topic_continuity_reason": "",
  "interest_match": 0.0,
  "interest_match_reason": "",
  "content_novelty": 0.0,
  "content_novelty_reason": "",
  "reconnect_value": 0.0,
  "reconnect_value_reason": "",
  "disturb_risk": 0.0,
  "disturb_risk_reason": "",
  "message_readiness": 0.0,
  "message_readiness_reason": "",
  "confidence": 0.0,
  "confidence_reason": ""
}}
"""
        try:
            resp = await self._provider.chat(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                tools=[],
                model=self._model,
                max_tokens=min(512, self._max_tokens),
            )
            raw = extract_json_object(resp.content or "")
            return self._sanitize(raw)
        except Exception:
            return self._sanitize({})

    @staticmethod
    def _sanitize(raw: dict) -> dict[str, float | str]:
        def get(name: str, default: float) -> float:
            try:
                v = float(raw.get(name, default))
            except Exception:
                v = default
            return max(0.0, min(1.0, v))

        def reason(name: str) -> str:
            try:
                text = str(raw.get(name, "")).strip()
            except Exception:
                text = ""
            return text[:120]

        return {
            "topic_continuity": get("topic_continuity", 0.5),
            "topic_continuity_reason": reason("topic_continuity_reason"),
            "interest_match": get("interest_match", 0.5),
            "interest_match_reason": reason("interest_match_reason"),
            "content_novelty": get("content_novelty", 0.5),
            "content_novelty_reason": reason("content_novelty_reason"),
            "reconnect_value": get("reconnect_value", 0.5),
            "reconnect_value_reason": reason("reconnect_value_reason"),
            "disturb_risk": get("disturb_risk", 0.5),
            "disturb_risk_reason": reason("disturb_risk_reason"),
            "message_readiness": get("message_readiness", 0.5),
            "message_readiness_reason": reason("message_readiness_reason"),
            "confidence": get("confidence", 0.5),
            "confidence_reason": reason("confidence_reason"),
        }


class ProactiveMessageDeduper:
    """发送前语义去重：兼顾新闻重复和用户状态/安慰框架重复。"""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        model: str,
        max_tokens: int,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens

    async def is_duplicate(
        self,
        new_message: str,
        recent_proactive: list[object],
        new_state_summary_tag: str = "none",
    ) -> tuple[bool, str]:
        """返回 (is_duplicate, reason)。recent_proactive 为空时直接放行。"""
        if not recent_proactive:
            return False, "无近期主动消息，放行"

        recent_text = _format_recent_proactive_entries(recent_proactive)
        system_msg = (
            "你是消息重复检测器。判断【新消息】是否与【近期已发消息】在实质信息上重复。\n"
            "重复定义包括两类：\n"
            "1. 同一事件/新闻/信息，即使措辞、语气、细节不同，本质信息相同。\n"
            "2. 同一用户状态总结或安慰框架重复，例如再次概括用户的压力、焦虑、别太逼自己、底子还在等，即使后半段换了新资讯，也算重复或低价值重复。\n"
            "不重复定义：话题相同但有真正的新进展、新内容或明显不同角度。\n"
            "只输出 JSON，不要其他内容。"
        )
        user_msg = (
            f"近期已发消息（最多5条）：\n{recent_text}\n\n"
            f"---\n新消息：{new_message}\n"
            f"新消息 state_summary_tag：{new_state_summary_tag}\n\n"
            "---\n只输出 JSON：\n"
            '{"is_duplicate": false, "reason": "简短说明"}'
        )
        try:
            resp = await self._provider.chat(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                tools=[],
                model=self._model,
                max_tokens=min(128, self._max_tokens),
            )
            content = (resp.content or "").strip()
            try:
                d = extract_json_object(content)
            except Exception:
                logger.warning("[proactive.deduper] 无法解析 JSON: %r", content[:100])
                return False, "解析失败，放行"
            is_dup = bool(d.get("is_duplicate", False))
            reason = str(d.get("reason", ""))
            logger.info(
                "[proactive.deduper] is_duplicate=%s reason=%r", is_dup, reason[:80]
            )
            return is_dup, reason
        except Exception as e:
            logger.warning("[proactive.deduper] 检测失败，放行: %s", e)
            return False, str(e)


class ProactiveMessageComposer:
    """特征模式下的消息生成器：只负责生成 message，不做是否发送决策。

    工具能力保持只读（read_file / web_fetch / web_search），避免主动链路拥有写入能力。
    """

    # PRE_FLIGHT：生成前强制自检，对齐主循环策略
    _PRE_FLIGHT = (
        "【生成消息前必须完成以下自检，无需在消息中说明】\n"
        "1. 消息中是否涉及用户的实时状态数据（游戏时长、订阅源列表等）？"
        "若涉及，必须先调用对应工具获取真实数据，禁止凭记忆臆断。\n"
        "   ⚠️ 健康数据：若 decision_signals 含 health_events，可调用 fitbit_health_snapshot 校验当前实时状态；"
        "若不含 health_events，禁止调用 fitbit_* 工具，禁止引用健康数值。\n"
        "2. 消息中是否有可能已过期的事实（如某游戏发布日期、某 DLC 上线状态）？"
        "若不确定，优先用 web_search/web_fetch 校验；仍无证据时用推测语气，避免给出具体日期。\n"
        "3. 如果消息引用了信息流或网页内容，你是否已经掌握确切来源名与可点击链接？"
        "若有，优先在正文自然带上“来源名 + URL”；若没有确切来源或链接，禁止编造。\n"
        "4. 若以上都不适用，直接生成消息即可。"
    )

    # REFLECT：工具结果后反思
    _REFLECT = (
        "根据上述工具结果，决定下一步操作。\n"
        "【自检】即将生成的消息中，有无工具结果支撑的事实？"
        "无支撑的必须用推测语气（'我猜'/'可能'/'好像'），禁止强断言。\n"
        "⚠️ 若任何工具调用失败（文件不存在、网络错误等），禁止在消息中提及失败本身；"
        "直接忽略该话题，转向其他可靠内容生成消息。"
    )

    def __init__(
        self,
        *,
        provider: LLMProvider,
        model: str,
        max_tokens: int,
        format_items: Callable[[list[FeedItem]], str],
        format_recent: Callable[[list[dict]], str],
        collect_global_memory: Callable[[], str],
        max_tool_iterations: int = 10,
        fitbit_url: str = "",
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens
        self._format_items = format_items
        self._format_recent = format_recent
        self._collect_global_memory = collect_global_memory
        self._max_tool_iterations = max_tool_iterations
        self._fitbit_tools: list[Tool] = build_fitbit_tools(
            fitbit_url=fitbit_url,
            requester=get_default_http_requester("local_service"),
        )
        self._fitbit_tool_schemas, self._fitbit_tool_map = (
            _build_optional_fitbit_tool_runtime(self._fitbit_tools)
        )
        # 工具实例：只读（不写文件、不执行系统动作）
        self._tools: list[Tool] = build_readonly_research_tools(
            fetch_requester=get_default_http_requester("external_default"),
        )
        prepared = prepare_toolset(self._tools)
        self._tool_schemas = prepared.schemas
        self._tool_map = prepared.tool_map

    async def compose_message(
        self,
        *,
        items: list[FeedItem],
        recent: list[dict],
        decision_signals: dict[str, object],
        retrieved_memory_block: str = "",
    ) -> str:
        prompt_context = _build_proactive_prompt_context(
            items=items,
            recent=recent,
            format_items=self._format_items,
            format_recent=self._format_recent,
            collect_global_memory=self._collect_global_memory,
        )

        system_msg = (
            "你是陪伴型助手。系统已经决定可以主动发送消息给用户。\n"
            "你只能调用只读工具（read_file / web_fetch / web_search，若存在 health_events 时可额外调用 fitbit_*）；"
            "不能写文件，不能执行外部动作。\n"
            "最终只输出一条自然、简短、可直接发送给用户的中文消息，不超过120字。\n"
            "消息要自然表达你的判断，不要用“你怎么看/你觉得呢”等征求看法的反问句收尾。\n"
            "除非必须让用户做明确选择，否则不要主动提问。\n"
            "若决策信号含 health_events，优先围绕健康事件给出关怀，再考虑资讯话题；只能引用 health_events[*].message，不要编造数值。\n"
            "若决策信号不含 health_events，禁止在消息正文中引用具体健康数值（心率、血氧等）。\n"
            "若有多条候选信息流，一次只围绕一个主题生成消息，不要写成摘要。\n"
            "消息不必强行承接近期对话；如果某条信息流本身就很贴合用户兴趣，可以自然地开启一个新话题。\n"
            "如果用户尚未回复最近一次主动关怀，不要重复总结用户当前处境，不要再次使用同类安慰前缀；若本次只是新资讯，直接进入新内容。\n"
            "若你引用了某条信息流里的具体消息，必须确保正文里有证据支撑；找不到确切来源时，不要硬写。\n"
            "## 身份（与主循环一致）\n"
            f"{AKASHIC_IDENTITY}\n"
            "## 性格（与主循环一致）\n"
            f"{PERSONALITY_RULES}\n"
            "如果内容来自 RSS / 网页，请优先自然点出来源名，并在合适时附上一个可点击的原文 URL；非网页来源不要伪造链接。\n"
            "系统不会替你自动补来源，所以正文里的来源信息必须由你自己写完整。\n"
            "不要输出JSON，不要解释，不要前缀。"
        )
        user_msg = f"""当前时间：{prompt_context.now_str}
（ISO格式：{prompt_context.now_iso}）

## 决策信号
```json
{json.dumps(decision_signals, ensure_ascii=False, indent=2)}
```
## 信息流
{prompt_context.feed_text}
## 长期记忆
{prompt_context.memory_text}
{f"## 相关记忆（本次触达召回）\n{retrieved_memory_block}\n" if retrieved_memory_block else ""}
## 近期对话
{prompt_context.chat_text}

请生成一条可发送给用户的消息（不超过120字）。
若近期对话里没有合适衔接，也可以直接从最相关的那条信息流自然起题。
若引用了某条信息流，请优先在正文里自然带上来源名；如果该条有“原文链接”，最好附上一个最关键的 URL，方便用户直接点进去。
若没有确切证据、来源名或链接，就不要把相关事实写成确定表述。"""

        messages: list[dict] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "user", "content": self._PRE_FLIGHT},
        ]
        last_tool_signature = ""
        repeat_count = 0
        active_tool_schemas, active_tool_map = _resolve_active_tool_runtime(
            base_schemas=self._tool_schemas,
            base_tool_map=self._tool_map,
            include_fitbit=bool(decision_signals.get("health_events")),
            fitbit_schemas=self._fitbit_tool_schemas,
            fitbit_tool_map=self._fitbit_tool_map,
        )

        try:
            for iteration in range(self._max_tool_iterations):
                resp = await self._provider.chat(
                    messages=messages,
                    tools=active_tool_schemas,
                    model=self._model,
                    max_tokens=self._max_tokens,
                )
                tool_calls = getattr(resp, "tool_calls", []) or []

                if not tool_calls:
                    # 无工具调用，最终消息
                    result = (resp.content or "").strip()
                    logger.info(
                        "[composer] 消息生成完成 iterations=%d chars=%d",
                        iteration + 1,
                        len(result),
                    )
                    return result

                signature = tool_call_signature(tool_calls)
                if signature and signature == last_tool_signature:
                    repeat_count += 1
                else:
                    repeat_count = 1
                    last_tool_signature = signature

                if repeat_count >= _TOOL_LOOP_REPEAT_LIMIT:
                    logger.warning(
                        "[composer] 检测到工具调用循环 signature=%s repeat=%d，提前收尾",
                        signature[:160],
                        repeat_count,
                    )
                    return await self._summarize_incomplete_progress(
                        messages,
                        reason="tool_call_loop",
                        iteration=iteration + 1,
                    )

                # 执行工具调用
                append_assistant_tool_calls(
                    messages,
                    content=resp.content,
                    tool_calls=tool_calls,
                )

                await _execute_tool_calls(
                    messages=messages,
                    tool_calls=tool_calls,
                    active_tool_map=active_tool_map,
                    log_prefix="[composer]",
                    log_each_call=True,
                )

                messages.append({"role": "user", "content": self._REFLECT})

            logger.warning(
                "[composer] 已达到最大工具迭代次数 %d", self._max_tool_iterations
            )
            return await self._summarize_incomplete_progress(
                messages,
                reason="max_iterations",
                iteration=self._max_tool_iterations,
            )
        except Exception as e:
            logger.warning("[composer] 消息生成失败: %s", e)
            return ""

    async def _summarize_incomplete_progress(
        self,
        messages: list[dict],
        *,
        reason: str,
        iteration: int,
    ) -> str:
        prompt = (
            f"[收尾原因] {reason}\n"
            f"[已执行轮次] {iteration}\n"
            "你没能在预算内完成全部工具链，请输出一条可直接发给用户的中文消息。\n"
            "要求：先简短说明你已核对到的进展，再给出下一步会继续做什么；"
            "不要出现“达到最大迭代次数”等模板句，不要 JSON。"
        )
        try:
            resp = await self._provider.chat(
                messages=messages + [{"role": "user", "content": prompt}],
                tools=[],
                model=self._model,
                max_tokens=min(_SUMMARY_MAX_TOKENS, self._max_tokens),
            )
            text = (resp.content or "").strip()
            if text:
                return text
        except Exception as e:
            logger.warning("[composer] 生成收尾总结失败: %s", e)
        return "我先整理到当前可确认的信息，后续补齐核验后再给你更完整的结论。"
