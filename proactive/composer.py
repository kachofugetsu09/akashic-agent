from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable
from urllib.parse import urlsplit

from agent.provider import LLMProvider
from agent.tools.web_fetch import WebFetchTool
from core.net.http import get_default_http_requester
from feeds.base import FeedItem
from prompts.proactive import build_compose_prompt_messages

logger = logging.getLogger(__name__)


def classify_content_quality(item: object) -> str:
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
            lines.append(f"话题: {re.sub(r'\\s+', ' ', title)[:60]}")
    lines.append("用户是否喜欢/关注/不关心该来源或话题")
    return "\n".join(lines)


def build_proactive_preference_hyde_prompt(query: str, context: str = "") -> str:
    context_section = f"\n候选上下文：\n{context}\n" if context else ""
    return (
        "你是个人助手的偏好记忆系统。根据当前候选内容与偏好检索问题，生成一条"
        "如果这类长期偏好已经存入记忆库时会长什么样的假想偏好记忆条目。\n"
        f"{context_section}"
        "规则：\n"
        "- 输出风格贴近 preference 记忆 summary：使用“用户明确... / 用户不喜欢...”这类第三人称陈述\n"
        "- 优先生成最可能命中长期偏好的那一条记忆\n"
        "- 聚焦长期偏好、反感、过滤倾向或关注方向，不要总结新闻事实本身\n"
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
        lines.extend(_memory_query_item_lines(item))
    lines.append("近期对话：")
    for message in recent[-max(1, int(max_recent)) :]:
        text = re.sub(r"\s+", " ", str(message.get("content", "")).strip())[:120]
        if not text:
            continue
        role = "用户" if str(message.get("role", "")) == "user" else "助手"
        lines.append(f"- {role}: {text}")
    lines.extend(_memory_query_alert_lines(decision_signals))
    lines.append("触达目标：")
    lines.append("- 基于当前内容生成自然主动消息")
    lines.append("- 优先遵循用户偏好与过往事件")
    if is_crisis:
        lines.append("- 当前是重连场景，优先自然开场")
    return "\n".join(lines)


def _memory_query_item_lines(item: FeedItem) -> list[str]:
    title = (item.title or "").strip() or "(无标题)"
    snippet = re.sub(r"\s+", " ", (item.content or "").strip())[:120]
    source = (item.source_name or "").strip()
    source_type = (item.source_type or "").strip().lower()
    source_key = f"{source_type}:{source.lower()}" if source_type or source else ""
    lines = [f"- {title}" + (f"（{source}）" if source else "")]
    if source_key:
        lines.append(f"  来源标签: {source_key}")
    domain = _source_domain(item.url)
    if domain:
        lines.append(f"  来源域名: {domain}")
    if snippet:
        lines.append(f"  {snippet}")
    return lines


def _memory_query_alert_lines(decision_signals: dict[str, object]) -> list[str]:
    alert_events = decision_signals.get("alert_events") or decision_signals.get(
        "health_events"
    )
    if not isinstance(alert_events, list) or not alert_events:
        return []
    lines = ["告警事件："]
    for event in alert_events[:2]:
        if not isinstance(event, dict):
            continue
        message = re.sub(
            r"\s+",
            " ",
            str(event.get("message", "") or event.get("content", "")).strip(),
        )[:120]
        if message:
            lines.append(f"- {message}")
    return lines


def _source_domain(url: str | None) -> str:
    if not url:
        return ""
    try:
        return (urlsplit(url).netloc or "").strip().lower()
    except Exception:
        return ""


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
) -> ProactivePromptContext:
    now = datetime.now().astimezone()
    return ProactivePromptContext(
        now_str=now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        now_iso=now.isoformat(),
        feed_text=format_items(items) or "（暂无订阅内容）",
        chat_text=format_recent(recent) or "（无近期对话记录）",
        memory_text="",
    )


class Composer:
    def __init__(
        self,
        *,
        provider: LLMProvider,
        model: str,
        max_tokens: int,
        format_items: Callable[[list[FeedItem]], str],
        format_recent: Callable[[list[dict]], str],
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens
        self._format_items = format_items
        self._format_recent = format_recent
        self._fetcher: WebFetchTool | None = None

    async def compose_for_judge(
        self,
        *,
        items: list[FeedItem],
        recent: list[dict],
        preference_block: str = "",
        no_content_token: str = "<no_content/>",
    ) -> str:
        # 1. 先统一补抓 compose 候选正文，避免上游和这里重复抓取。
        items = await self._enrich_items(items)
        # 2. 再构造最小 compose prompt，只做生成，不做额外判断。
        prompt_context = _build_proactive_prompt_context(
            items=items,
            recent=recent,
            format_items=self._format_items,
            format_recent=self._format_recent,
        )
        system_msg, user_msg = build_compose_prompt_messages(
            prompt_context=prompt_context,
            preference_block=preference_block,
            no_content_token=no_content_token,
        )
        # 3. 最后请求模型，显式支持 no_content token。
        try:
            response = await self._provider.chat(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                tools=[],
                model=self._model,
                max_tokens=min(512, self._max_tokens),
            )
        except Exception as exc:
            logger.warning("[compose] compose_for_judge 失败: %s", exc)
            return ""
        text = (response.content or "").strip()
        if text.startswith(no_content_token):
            logger.info("[compose] → <no_content/>，内容无价值，提前退出")
            return no_content_token
        logger.info(
            "[compose] → 生成成功 %d字符: %s",
            len(text),
            text[:80].replace("\n", " "),
        )
        return text

    async def _enrich_items(self, items: list[FeedItem]) -> list[FeedItem]:
        candidates = [
            item
            for item in items[:2]
            if item.url and classify_content_quality(item) != "full"
        ]
        if not candidates:
            return items
        fetcher = self._ensure_fetcher()
        if fetcher is None:
            return items
        for item in candidates:
            await self._enrich_one_item(fetcher, item)
        return items

    def _ensure_fetcher(self) -> WebFetchTool | None:
        if self._fetcher is not None:
            return self._fetcher
        try:
            self._fetcher = WebFetchTool(
                get_default_http_requester("external_default")
            )
        except Exception as exc:
            logger.info("[compose] fetcher init failed: %s", exc)
            return None
        return self._fetcher

    async def _enrich_one_item(
        self,
        fetcher: WebFetchTool,
        item: FeedItem,
    ) -> None:
        try:
            raw = await fetcher.execute(url=item.url, format="text", timeout=8)
            data = json.loads(raw or "{}")
        except Exception as exc:
            logger.info("[compose] enrich_item_failed url=%s err=%s", item.url, exc)
            setattr(item, "content_status", "fetch_failed")
            return
        text = str(data.get("text", "") or "").strip()
        if len(text) <= 400:
            setattr(item, "content_status", "fetch_failed")
            return
        item.content = text[:4000]
        setattr(item, "content_status", "fetched")
