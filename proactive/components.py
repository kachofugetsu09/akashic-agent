from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urlsplit

from agent.provider import LLMProvider
from agent.tools.message_push import MessagePushTool
from agent.tools.web_fetch import WebFetchTool
from core.net.http import get_default_http_requester
from feeds.base import FeedItem
from prompts.proactive import (
    build_compose_prompt_messages,
    build_post_judge_prompt_messages,
)
from proactive.json_utils import extract_json_object
from proactive.presence import PresenceStore
from proactive.state import ProactiveStateStore
from session.manager import SessionManager

logger = logging.getLogger(__name__)


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
