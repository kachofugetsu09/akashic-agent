from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urlsplit

from agent.persona import AKASHIC_IDENTITY, PERSONALITY_RULES
from agent.provider import LLMProvider
from agent.tool_bundles import build_fitbit_tools, build_readonly_research_tools
from agent.tool_runtime import (
    append_assistant_tool_calls,
    append_tool_result,
    prepare_toolset,
    tool_call_signature,
)
from agent.tools.base import Tool
from agent.tools.message_push import MessagePushTool
from agent.tools.web_fetch import WebFetchTool
from core.net.http import get_default_http_requester
from feeds.base import FeedItem
from prompts.proactive import (
    build_compose_prompt_messages,
    build_feature_scoring_prompt_messages,
    build_post_judge_prompt_messages,
)
from proactive.json_utils import extract_json_object
from proactive.presence import PresenceStore
from proactive.state import ProactiveStateStore
from session.manager import SessionManager

logger = logging.getLogger(__name__)

_TOOL_LOOP_REPEAT_LIMIT = 3
_SUMMARY_MAX_TOKENS = 384


def classify_content_quality(item: object) -> str:
    """粗粒度判断内容质量，决定是否需要补抓原文。"""
    content = str(getattr(item, "content", "") or "").strip()
    title = str(getattr(item, "title", "") or "").strip()
    if len(content) > 300:
        return "full"
    if len(content) > 60:
        return "snippet"
    if title:
        return "title_only"
    return "empty"


def build_proactive_preference_query(
    *,
    items: list[FeedItem],
    max_items: int = 3,
) -> str:
    """构建针对候选 item 来源/话题的偏好专项查询。

    用于独立检索 preference 类型记忆，问题是“用户对这些内容来源的态度/偏好是什么”，
    而非通用记忆查询。返回的查询字符串应能命中向量库中类似“用户只关注 TeamAtlas 和 PlayerNova”
    的偏好记忆。
    """
    lines: list[str] = ["用户偏好 兴趣 关注"]
    seen_sources: set[str] = set()
    for item in items[: max(1, max_items)]:
        source = (item.source_name or "").strip()
        source_type = (item.source_type or "").strip().lower()
        title = (item.title or "").strip()
        if source and source not in seen_sources:
            seen_sources.add(source)
            lines.append(f"来源: {source}")
            if source_type:
                lines.append(f"来源类型: {source_type}:{source.lower()}")
        if title:
            snippet = re.sub(r"\s+", " ", title)[:60]
            lines.append(f"话题: {snippet}")
    lines.append("用户是否喜欢/关注/不关心该来源或话题")
    return "\n".join(lines)


def build_proactive_preference_hyde_prompt(query: str, context: str = "") -> str:
    """为 preference 检索生成更像真实偏好记忆的 HyDE prompt。"""
    context_section = f"\n候选上下文：\n{context}\n" if context else ""
    return (
        "你是个人助手的偏好记忆系统。根据当前候选内容与偏好检索问题，生成一条"
        "如果这类长期偏好已经存入记忆库时会长什么样的假想偏好记忆条目。\n"
        f"{context_section}"
        "规则：\n"
        "- 输出风格贴近 preference 记忆 summary：使用“用户明确... / 用户不喜欢... / 主动消息...”这类第三人称陈述\n"
        "- 优先生成最可能命中长期偏好的那一条记忆；如果候选更像用户会反感、过滤、排斥或不想被推送的内容，就生成负向偏好记忆，而不是勉强生成正向兴趣\n"
        "- 聚焦长期偏好、反感、过滤倾向或关注方向，不要总结新闻事实本身\n"
        "- 特别注意用户对平台、设备生态、题材、内容来源的长期厌恶或排斥，这类负向偏好和正向关注同样重要\n"
        "- 不要提问，不要解释，不要输出多条\n"
        "- 只输出一条简洁中文文本\n\n"
        f"偏好检索问题：{query}\n"
        "假想偏好记忆条目："
    )


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
    candidate_message = re.sub(
        r"\s+",
        " ",
        str(decision_signals.get("candidate_message", "")).strip(),
    )[:120]
    if candidate_message:
        lines.append(f"拟发送消息: {candidate_message}")

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

    # alert_events 包含所有告警类型；health_events 是其健康专属子集。
    # memory query 使用 alert_events，确保非健康告警（如湿度报警）也能触发相关记忆检索。
    alert_events = decision_signals.get("alert_events") or decision_signals.get("health_events")
    if isinstance(alert_events, list) and alert_events:
        lines.append("告警事件：")
        for event in alert_events[:2]:
            message = re.sub(
                r"\s+",
                " ",
                str((event or {}).get("message", "") or (event or {}).get("content", "")).strip(),
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


@dataclass(frozen=True)
class ProactiveJudgeResult:
    final_score: float
    should_send: bool
    vetoed_by: str | None
    dims_deterministic: dict[str, float]
    dims_llm: dict[str, float]
    dims_llm_raw: dict[str, int]


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
        preference_block: str = "",
    ) -> dict[str, float | str]:
        prompt_context = _build_proactive_prompt_context(
            items=items,
            recent=recent,
            format_items=self._format_items,
            format_recent=self._format_recent,
            collect_global_memory=self._collect_global_memory,
        )
        combined_block = retrieved_memory_block
        if preference_block:
            pref_section = "## 用户偏好（强约束）\n" + preference_block
            combined_block = (
                pref_section + "\n\n" + retrieved_memory_block
                if retrieved_memory_block
                else pref_section
            )
        system_msg, user_msg = build_feature_scoring_prompt_messages(
            prompt_context=prompt_context,
            decision_signals=decision_signals,
            retrieved_memory_block=combined_block,
        )
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


class ProactiveJudge:
    """compose 后置评分器：聚合确定性维度与 LLM 维度。"""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        model: str,
        max_tokens: int,
        format_items: Callable[[list[FeedItem]], str],
        format_recent: Callable[[list[dict]], str],
        cfg: Any,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens
        self._format_items = format_items
        self._format_recent = format_recent
        self._cfg = cfg

    async def compose_for_judge(
        self,
        *,
        items: list[FeedItem],
        recent: list[dict],
        preference_block: str = "",
        no_content_token: str = "<no_content/>",
    ) -> str:
        logger.info(
            "[compose] 开始生成消息 items=%d pref_block=%d字符",
            len(items),
            len(preference_block),
        )
        # 1. 先尽量为最终选中的条目补抓正文；抓不到时保留原标题/摘要继续生成。
        items = await self._enrich_items_for_compose(items)
        # 1. 基于候选内容、近期对话、偏好构造最小 compose prompt。
        prompt_context = _build_proactive_prompt_context(
            items=items,
            recent=recent,
            format_items=self._format_items,
            format_recent=self._format_recent,
            collect_global_memory=lambda: "",
        )
        system_msg, user_msg = build_compose_prompt_messages(
            prompt_context=prompt_context,
            preference_block=preference_block,
            no_content_token=no_content_token,
        )
        # 2. 仅做一次无工具调用，让模型专注生成内容本身。
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
        except Exception as e:
            logger.warning("[compose] compose_for_judge 失败: %s", e)
            return ""
        # 3. 规范化输出，显式支持 no_content token。
        text = (resp.content or "").strip()
        if text.startswith(no_content_token):
            logger.info("[compose] → <no_content/>，内容无价值，提前退出")
            return no_content_token
        logger.info("[compose] → 生成成功 %d字符: %s", len(text), text[:80].replace("\n", " "))
        return text

    async def _enrich_items_for_compose(self, items: list[FeedItem]) -> list[FeedItem]:
        # 1. 只补抓最终 compose 组，最多 2 条，避免主动链路过重。
        # 2. 只在标题/摘要过短时抓取，已有较完整正文就直接复用。
        # 3. 抓取失败时保持原内容，不影响后续生成。
        selected = items[:]
        candidates = [
            item
            for item in selected[:2]
            if item.url and classify_content_quality(item) != "full"
        ]
        if not candidates:
            return selected
        fetcher = WebFetchTool(get_default_http_requester("external_default"))
        for item in candidates:
            try:
                raw = await fetcher.execute(url=item.url, format="text", timeout=8)
                data = json.loads(raw or "{}")
                text = str(data.get("text", "") or "").strip()
                if len(text) > 400:
                    item.content = text[:4000]
                    setattr(item, "content_status", "fetched")
            except Exception as e:
                logger.info("[compose] enrich_item_failed url=%s err=%s", item.url, e)
                setattr(item, "content_status", "fetch_failed")
        return selected

    async def judge_message(
        self,
        *,
        message: str,
        recent: list[dict],
        recent_proactive_text: str,
        preference_block: str = "",
        age_hours: float,
        sent_24h: int,
        interrupt_factor: float,
    ) -> ProactiveJudgeResult:
        # 1. 先计算确定性维度并执行硬否决。
        deterministic = self._build_deterministic_dims(
            age_hours=age_hours,
            sent_24h=sent_24h,
            interrupt_factor=interrupt_factor,
        )
        logger.info(
            "[judge] 确定性维度 urgency=%.3f balance=%.3f dynamics=%.3f"
            "  (age_hours=%.1f sent_24h=%d interrupt=%.3f)",
            deterministic["urgency"],
            deterministic["balance"],
            deterministic["dynamics"],
            age_hours,
            sent_24h,
            interrupt_factor,
        )
        vetoed = self._deterministic_veto(deterministic)
        if vetoed:
            logger.info("[judge] 确定性否决 vetoed_by=%s → 不发送", vetoed)
            return ProactiveJudgeResult(0.0, False, vetoed, deterministic, {}, {})
        # 2. 再请求 LLM 三维打分并校验维度下限。
        llm_dims, llm_dims_raw = await self._score_llm_dims(
            message=message,
            recent=recent,
            recent_proactive_text=recent_proactive_text,
            preference_block=preference_block,
        )
        logger.info(
            "[judge] LLM维度 info_gap=%d relevance=%d impact=%d (归一化 %.2f/%.2f/%.2f)",
            llm_dims_raw.get("information_gap", 0),
            llm_dims_raw.get("relevance", 0),
            llm_dims_raw.get("expected_impact", 0),
            llm_dims.get("information_gap", 0.0),
            llm_dims.get("relevance", 0.0),
            llm_dims.get("expected_impact", 0.0),
        )
        llm_veto_min = (int(getattr(self._cfg, "judge_veto_llm_dim_min", 2)) - 1) / 4.0
        if any(v < llm_veto_min for v in llm_dims.values()):
            low_dims = [k for k, v in llm_dims.items() if v < llm_veto_min]
            logger.info("[judge] LLM维度否决 low_dims=%s → 不发送", low_dims)
            return ProactiveJudgeResult(
                0.0,
                False,
                "llm_dim",
                deterministic,
                llm_dims,
                llm_dims_raw,
            )
        # 3. 最后按权重汇总总分并输出是否发送。
        final_score = self._compute_final_score(deterministic, llm_dims)
        threshold = float(getattr(self._cfg, "judge_send_threshold", 0.60))
        should_send = final_score >= threshold
        logger.info(
            "[judge] final_score=%.3f 阈值=%.2f → %s",
            final_score,
            threshold,
            "发送" if should_send else "不发送",
        )
        return ProactiveJudgeResult(
            final_score,
            should_send,
            None,
            deterministic,
            llm_dims,
            llm_dims_raw,
        )

    def _build_deterministic_dims(
        self,
        *,
        age_hours: float,
        sent_24h: int,
        interrupt_factor: float,
    ) -> dict[str, float]:
        daily_max = max(1, int(getattr(self._cfg, "judge_balance_daily_max", 8)))
        urgency_horizon = max(
            1.0, float(getattr(self._cfg, "judge_urgency_horizon_hours", 12.0))
        )
        urgency = max(0.0, 1.0 - (max(age_hours, 0.0) / urgency_horizon))
        balance = max(0.0, 1.0 - (max(sent_24h, 0) / float(daily_max)))
        dynamics = 0.6 + 0.4 * max(0.0, min(1.0, float(interrupt_factor)))
        return {"urgency": urgency, "balance": balance, "dynamics": dynamics}

    def pre_compose_veto(
        self,
        *,
        age_hours: float,
        sent_24h: int,
        interrupt_factor: float,
    ) -> str | None:
        """compose 前仅凭确定性维度做快速否决，避免 LLM 调用浪费。"""
        deterministic = self._build_deterministic_dims(
            age_hours=age_hours,
            sent_24h=sent_24h,
            interrupt_factor=interrupt_factor,
        )
        vetoed = self._deterministic_veto(deterministic)
        if vetoed:
            logger.info(
                "[judge] pre-compose 确定性否决 vetoed_by=%s"
                " urgency=%.3f balance=%.3f (age_hours=%.1f sent_24h=%d)",
                vetoed,
                deterministic["urgency"],
                deterministic["balance"],
                age_hours,
                sent_24h,
            )
        return vetoed

    def _deterministic_veto(self, deterministic: dict[str, float]) -> str | None:
        # 1. MVP 只保留 balance 硬否决，避免旧内容因 urgency 过低被挡在 compose 前。
        if deterministic["balance"] < float(getattr(self._cfg, "judge_veto_balance_min", 0.1)):
            return "balance"
        return None

    async def _score_llm_dims(
        self,
        *,
        message: str,
        recent: list[dict],
        recent_proactive_text: str,
        preference_block: str = "",
    ) -> tuple[dict[str, float], dict[str, int]]:
        system_msg, user_msg = build_post_judge_prompt_messages(
            recent_summary=self._format_recent(recent) or "（无近期对话）",
            last_proactive=recent_proactive_text or "（无近期主动消息）",
            composed_message=message,
            preference_block=preference_block,
        )
        try:
            resp = await self._provider.chat(
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                tools=[],
                model=self._model,
                max_tokens=min(256, self._max_tokens),
            )
            raw = extract_json_object(resp.content or "")
        except Exception:
            raw = {}
        raw_dims = {
            "information_gap": self._clamp_dim(raw.get("information_gap")),
            "relevance": self._clamp_dim(raw.get("relevance")),
            "expected_impact": self._clamp_dim(raw.get("expected_impact")),
        }
        normalized = {k: (v - 1) / 4.0 for k, v in raw_dims.items()}
        return normalized, raw_dims

    @staticmethod
    def _clamp_dim(raw: object) -> int:
        try:
            value = int(raw)
        except Exception:
            value = 2
        return max(1, min(5, value))

    def _compute_final_score(
        self,
        deterministic: dict[str, float],
        llm_dims: dict[str, float],
    ) -> float:
        weights = {
            "urgency": float(getattr(self._cfg, "judge_weight_urgency", 0.15)),
            "balance": float(getattr(self._cfg, "judge_weight_balance", 0.10)),
            "dynamics": float(getattr(self._cfg, "judge_weight_dynamics", 0.10)),
            "information_gap": float(getattr(self._cfg, "judge_weight_information_gap", 0.25)),
            "relevance": float(getattr(self._cfg, "judge_weight_relevance", 0.20)),
            "expected_impact": float(getattr(self._cfg, "judge_weight_expected_impact", 0.20)),
        }
        all_dims: dict[str, float] = dict(deterministic)
        all_dims.update(llm_dims)
        weight_sum = sum(weights.get(k, 0.0) for k in all_dims.keys())
        if weight_sum <= 0:
            return 0.0
        return sum(weights.get(k, 0.0) * all_dims[k] for k in all_dims.keys()) / weight_sum


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
        "   ⚠️ 告警处理：优先围绕 alert_events 中最值得处理的一条告警生成消息；"
        "若该告警涉及健康来源，可调用 fitbit_health_snapshot 校验当前实时状态。\n"
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
        preference_block: str = "",
    ) -> str:
        prompt_context = _build_proactive_prompt_context(
            items=items,
            recent=recent,
            format_items=self._format_items,
            format_recent=self._format_recent,
            collect_global_memory=self._collect_global_memory,
        )

        preference_constraint = ""
        if preference_block:
            preference_constraint = (
                "## 用户偏好（硬约束，生成时必须遵守）\n"
                + preference_block
                + "\n"
                + "若上述偏好中明确表示不关注某来源、话题或内容，"
                + "禁止在消息中提及该内容，禁止出现逆偏好措辞。\n"
            )

        system_msg = (
            "你是陪伴型助手。系统已经决定可以主动发送消息给用户。\n"
            "你只能调用只读工具（read_file / web_fetch / web_search / fitbit_*）；"
            "不能写文件，不能执行外部动作。\n"
            "最终只输出一条自然、可直接发送给用户的中文消息，不超过400字。\n"
            "消息要自然表达你的判断，不要用“你怎么看/你觉得呢”等征求看法的反问句收尾。\n"
            "除非必须让用户做明确选择，否则不要主动提问。\n"
            "若决策信号含 alert_events，优先围绕其中最值得处理的一条告警生成消息，再考虑资讯话题；健康告警、日历告警等来源在告警优先级上同级。\n"
            "若该告警涉及健康来源，只能引用 health_events[*].message，不要编造数值；若不是健康来源，就基于对应 alert_events[*] 的 content/title/source_name 来写。\n"
            "若有多条候选信息流，一次只围绕一个主题生成消息；同一主题下若有 2-3 条连续更新，优先合并成一条更完整的主动消息，允许自然串联多个进展，不必压成单条快讯。\n"
            "消息不必强行承接近期对话；如果某条信息流本身就很贴合用户兴趣，可以自然地开启一个新话题。\n"
            "如果用户尚未回复最近一次主动关怀，不要重复总结用户当前处境，不要再次使用同类安慰前缀；若本次只是新资讯，直接进入新内容。\n"
            "若你引用了某条信息流里的具体消息，必须确保正文里有证据支撑；找不到确切来源时，不要硬写。\n"
            + preference_constraint
            + "## 身份（与主循环一致）\n"
            f"{AKASHIC_IDENTITY}\n"
            "## 性格（与主循环一致）\n"
            f"{PERSONALITY_RULES}\n"
            "如果内容来自 RSS / 网页，请优先自然点出来源名，并在合适时附上可点击的原文 URL；非网页来源不要伪造链接。\n"
            "若同一主题聚合了 2-3 条更新，正文里每个被你明确提到的具体进展，都应附上对应链接；允许把多个链接集中放在消息末尾逐行列出。\n"
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

请生成一条可发送给用户的消息（不超过400字）。
若近期对话里没有合适衔接，也可以直接从最相关的那条信息流自然起题。
如果 items 里本来就是同一主题的连续更新，应优先把 2-3 条自然整合进一条消息里，允许按“先说结论，再补充两三个关键进展”的方式展开，但不要扩展到不同主题。
若引用了某条信息流，请优先在正文里自然带上来源名；如果你在一条消息里聚合了多条更新，则每条被提到的更新都应带上对应链接，必要时可把多个 URL 集中放在结尾。
若没有确切证据、来源名或链接，就不要把相关事实写成确定表述。"""

        logger.debug("[prompt:compose] user_msg=\n%s", user_msg)
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
            include_fitbit=True,
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
