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
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from agent.provider import LLMProvider
from agent.tools.message_push import MessagePushTool
from feeds.base import FeedItem
from feeds.registry import FeedRegistry
from proactive.state import ProactiveStateStore
from session.manager import SessionManager

logger = logging.getLogger(__name__)


@dataclass
class ProactiveConfig:
    enabled: bool = False
    interval_seconds: int = 1800    # 两次 tick 间隔（秒）
    threshold: float = 0.70         # score 高于此值才发送
    items_per_source: int = 3       # 每个信息源取几条
    recent_chat_messages: int = 20  # 回顾最近 N 条对话
    model: str = ""                 # 留空则继承全局 model
    default_channel: str = "telegram"
    default_chat_id: str = ""
    dedupe_seen_ttl_hours: int = 24 * 14
    delivery_dedupe_hours: int = 24
    only_new_items_trigger: bool = True


@dataclass
class _Decision:
    score: float
    should_send: bool
    message: str
    reasoning: str
    evidence_item_ids: list[str] = field(default_factory=list)


class ProactiveLoop:
    def __init__(
        self,
        feed_registry: FeedRegistry,
        session_manager: SessionManager,
        provider: LLMProvider,
        push_tool: MessagePushTool,
        config: ProactiveConfig,
        model: str,
        max_tokens: int = 1024,
        state_store: ProactiveStateStore | None = None,
        state_path: Path | None = None,
    ) -> None:
        self._feeds = feed_registry
        self._sessions = session_manager
        self._provider = provider
        self._push = push_tool
        self._cfg = config
        self._model = config.model or model
        self._max_tokens = max_tokens
        self._state = state_store or ProactiveStateStore(state_path or Path("proactive_state.json"))
        self._running = False
        logger.info(
            "[proactive] 去重配置 seen_ttl=%dh delivery_window=%dh only_new_items_trigger=%s",
            self._cfg.dedupe_seen_ttl_hours,
            self._cfg.delivery_dedupe_hours,
            self._cfg.only_new_items_trigger,
        )

    async def run(self) -> None:
        self._running = True
        logger.info(
            f"ProactiveLoop 已启动  间隔={self._cfg.interval_seconds}s  "
            f"阈值={self._cfg.threshold}  "
            f"目标={self._cfg.default_channel}:{self._cfg.default_chat_id}"
        )
        while self._running:
            await asyncio.sleep(self._cfg.interval_seconds)
            try:
                await self._tick()
            except Exception:
                logger.exception("ProactiveLoop tick 异常")

    def stop(self) -> None:
        self._running = False

    # ── internal ──────────────────────────────────────────────────

    async def _tick(self) -> None:
        logger.info("[proactive] tick 开始")
        self._state.cleanup(
            seen_ttl_hours=self._cfg.dedupe_seen_ttl_hours,
            delivery_ttl_hours=self._cfg.delivery_dedupe_hours,
        )

        # 1. 并发拉取信息流
        items = await self._feeds.fetch_all(self._cfg.items_per_source)
        logger.info(f"[proactive] 拉取到 {len(items)} 条信息")
        new_items, new_entries = self._filter_new_items(items)
        logger.info(
            "[proactive] 去重后剩余新信息 %d 条（过滤重复 %d 条）",
            len(new_items),
            len(items) - len(new_items),
        )
        if not new_items and self._cfg.only_new_items_trigger:
            logger.info("[proactive] 无新信息，按配置跳过本轮反思")
            return

        # 2. 最近聊天上下文
        recent = self._collect_recent()
        logger.info("[proactive] 最近会话消息条数=%d", len(recent))

        # 3. LLM 反思
        reflect_items = new_items if self._cfg.only_new_items_trigger else items
        decision = await self._reflect(reflect_items, recent)
        logger.info(
            f"[proactive] score={decision.score:.2f}  "
            f"send={decision.should_send}  "
            f"reasoning={decision.reasoning[:80]!r}"
        )

        # 4. 阈值判断
        if decision.should_send and decision.score >= self._cfg.threshold:
            channel = (self._cfg.default_channel or "").strip()
            chat_id = self._cfg.default_chat_id.strip()
            session_key = f"{channel}:{chat_id}" if channel and chat_id else ""
            evidence_ids = _resolve_evidence_item_ids(decision, new_items if new_items else reflect_items)
            delivery_key = _build_delivery_key(evidence_ids, decision.message)
            logger.info(
                "[proactive] 发送前去重检查 session=%s evidence_count=%d delivery_key=%s",
                session_key or "（未配置）",
                len(evidence_ids),
                delivery_key[:16],
            )
            if session_key and self._state.is_delivery_duplicate(
                session_key=session_key,
                delivery_key=delivery_key,
                window_hours=self._cfg.delivery_dedupe_hours,
            ):
                logger.info("[proactive] 命中发送去重，跳过发送")
                self._state.mark_items_seen(new_entries)
                logger.info("[proactive] 已按去重命中标记本轮条目为 seen（视为已送达过同等内容）")
                return
            sent = await self._send(decision.message)
            if sent and session_key:
                self._state.mark_delivery(session_key, delivery_key)
                self._state.mark_items_seen(new_entries)
                logger.info("[proactive] 已发送成功并标记本轮条目为 seen")
            else:
                logger.info("[proactive] 本轮发送未成功，不标记 seen，后续可再次尝试")
        else:
            logger.info("[proactive] 决定不主动发送")
            logger.info("[proactive] 本轮未发送，不标记 seen，后续可再次尝试")

    def _filter_new_items(self, items: list[FeedItem]) -> tuple[list[FeedItem], list[tuple[str, str]]]:
        if not items:
            logger.info("[proactive] 本轮无 item，去重过滤跳过")
            return [], []
        now = datetime.now(timezone.utc)
        new_items: list[FeedItem] = []
        new_entries: list[tuple[str, str]] = []
        for item in items:
            source_key = _source_key(item)
            item_id = _item_id(item)
            seen = self._state.is_item_seen(
                source_key=source_key,
                item_id=item_id,
                ttl_hours=self._cfg.dedupe_seen_ttl_hours,
                now=now,
            )
            logger.info(
                "[proactive] item 去重检查 source=%s item_id=%s seen=%s title=%r",
                source_key,
                item_id[:16],
                seen,
                (item.title or "")[:60],
            )
            if seen:
                continue
            new_items.append(item)
            new_entries.append((source_key, item_id))
        return new_items, new_entries

    def _collect_recent(self) -> list[dict]:
        """取目标会话最近 N 条消息（只取 user/assistant 文本）。"""
        channel = (self._cfg.default_channel or "").strip()
        chat_id = self._cfg.default_chat_id.strip()
        if not channel or not chat_id:
            logger.info("[proactive] collect_recent 跳过：目标 channel/chat_id 未配置")
            return []
        key = f"{channel}:{chat_id}"

        try:
            session = self._sessions.get_or_create(key)
            msgs = session.messages[-self._cfg.recent_chat_messages:]
            logger.info(
                "[proactive] collect_recent 成功 key=%s total=%d selected=%d",
                key,
                len(session.messages),
                len(msgs),
            )
            return [
                {"role": m["role"], "content": str(m.get("content", ""))[:200]}
                for m in msgs
                if m.get("role") in ("user", "assistant") and m.get("content")
            ]
        except Exception as e:
            logger.warning(f"[proactive] 加载 session {key!r} 失败: {e}")
            return []

    async def _reflect(self, items: list[FeedItem], recent: list[dict]) -> _Decision:
        now_str = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
        feed_text = _format_items(items) or "（暂无订阅内容）"
        chat_text = _format_recent(recent) or "（无近期对话记录）"

        system_msg = (
            "你是一个陪伴型 AI 助手，正在决定是否主动联系用户。"
            "你了解用户订阅的信息流和最近的对话内容。"
            "你的目标是在恰当的时机分享有价值的信息，而不是频繁打扰用户。"
        )

        user_msg = f"""当前时间：{now_str}

## 订阅信息流（最新内容）

{feed_text}

## 近期对话

{chat_text}

## 任务

综合以上信息，判断是否值得主动联系用户。考虑：
- 信息流里有没有用户可能感兴趣的内容
- 现在说点什么是否自然、不唐突
- 与近期对话有无关联或延伸

只输出 JSON，不要其他内容：
{{
  "reasoning": "内心独白（不会显示给用户，说清楚你的判断依据）",
  "score": 0.0,
  "should_send": false,
  "message": "",
  "evidence_item_ids": []
}}

score 说明：0.0=完全没必要  0.5=有点想说  0.7=比较值得  1.0=非常值得立刻说
message 若 should_send=true，写要发给用户的话（口语化，不要像系统通知）
evidence_item_ids 从订阅信息流里挑选支持你判断的 item_id（可为空数组）"""

        try:
            resp = await self._provider.chat(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                tools=[],
                model=self._model,
                max_tokens=self._max_tokens,
            )
            content = resp.content or ""
            logger.info("[proactive] LLM 原始输出预览: %r", content[:240])
            return _parse_decision(content)
        except Exception as e:
            logger.error(f"[proactive] LLM 反思失败: {e}")
            return _Decision(score=0.0, should_send=False, message="", reasoning=str(e))

    async def _send(self, message: str) -> bool:
        channel = (self._cfg.default_channel or "").strip()
        chat_id = self._cfg.default_chat_id.strip()
        if not channel or not chat_id:
            logger.warning("[proactive] default_channel/default_chat_id 未配置，跳过发送")
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
            session.add_message(
                "assistant",
                message,
                proactive=True,
                tools_used=["message_push"],
            )
            self._sessions.save(session)
            logger.info(f"[proactive] 已发送主动消息并写入会话 → {channel}:{chat_id}")
            return True
        except Exception as e:
            logger.error(f"[proactive] 发送失败: {e}")
            return False


# ── helpers ──────────────────────────────────────────────────────

def _format_items(items: list[FeedItem]) -> str:
    if not items:
        return ""
    lines = []
    for item in items:
        pub = ""
        if item.published_at:
            try:
                pub = " (" + item.published_at.astimezone().strftime("%m-%d %H:%M") + ")"
            except Exception:
                pass
        title = item.title or "(无标题)"
        lines.append(f"[{item.source_name}|item_id={_item_id(item)}]{pub} {title}")
        if item.content:
            lines.append(f"  {item.content[:200]}")
        if item.url:
            lines.append(f"  {item.url}")
    return "\n".join(lines)


def _format_recent(msgs: list[dict]) -> str:
    if not msgs:
        return ""
    lines = []
    for m in msgs[-10:]:   # 最多展示最近 10 条
        role = "用户" if m["role"] == "user" else "助手"
        content = str(m.get("content", ""))[:150]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _parse_decision(text: str) -> _Decision:
    """从 LLM 输出中提取 JSON 决策。"""
    # 先尝试提取 ```json ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    raw = match.group(1) if match else text

    # 找第一个完整的 { ... }
    brace_match = re.search(r"\{[\s\S]*\}", raw)
    if not brace_match:
        logger.warning(f"[proactive] 无法提取 JSON: {text[:200]!r}")
        return _Decision(score=0.0, should_send=False, message="", reasoning="parse error")

    try:
        d = json.loads(brace_match.group())
        evidence = d.get("evidence_item_ids", [])
        if not isinstance(evidence, list):
            evidence = []
        return _Decision(
            score=float(d.get("score", 0.0)),
            should_send=_strict_bool(d.get("should_send", False)),
            message=str(d.get("message", "")),
            reasoning=str(d.get("reasoning", "")),
            evidence_item_ids=[str(x).strip() for x in evidence if str(x).strip()],
        )
    except Exception as e:
        logger.warning(f"[proactive] JSON 解析失败: {e}  raw={raw[:200]!r}")
        return _Decision(score=0.0, should_send=False, message="", reasoning=str(e))


def _strict_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "true":
            return True
        if text == "false":
            return False
    return False


def _source_key(item: FeedItem) -> str:
    return f"{(item.source_type or '').strip().lower()}:{(item.source_name or '').strip().lower()}"


def _normalize_url(url: str | None) -> str:
    if not url:
        return ""
    try:
        p = urlsplit(url.strip())
        scheme = (p.scheme or "").lower()
        netloc = (p.netloc or "").lower()
        path = p.path.rstrip("/")
        return urlunsplit((scheme, netloc, path, p.query, ""))
    except Exception:
        return (url or "").strip()


def _item_id(item: FeedItem) -> str:
    url = _normalize_url(item.url)
    if url:
        return "u_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    raw = "|".join([
        (item.source_type or "").strip().lower(),
        (item.source_name or "").strip().lower(),
        (item.title or "").strip().lower(),
        (item.content or "").strip().lower()[:200],
        item.published_at.isoformat() if item.published_at else "",
    ])
    return "h_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _resolve_evidence_item_ids(decision: _Decision, items: list[FeedItem]) -> list[str]:
    valid = {_item_id(i) for i in items}
    selected = [x for x in decision.evidence_item_ids if x in valid]
    if selected:
        return sorted(set(selected))
    fallback = sorted(valid)
    return fallback[:5]


def _build_delivery_key(item_ids: list[str], message: str) -> str:
    canonical_ids = "|".join(sorted(set(item_ids)))
    canonical_msg = re.sub(r"\s+", " ", (message or "").strip().lower())
    raw = f"{canonical_ids}::{canonical_msg}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()
