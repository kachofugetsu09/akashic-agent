"""
ProactiveStateStore — 主动消息流的去重状态持久化。

状态文件包含五类信息：
1) seen_items: 每个 source 下已处理过的 item_id（长 TTL，14天）
2) deliveries: 每个 session 下已发送过的 delivery_key
3) semantic_items: 语义去重历史（文本与时间戳，72h TTL）
4) rejection_cooldown: LLM 拒绝后的软冷却（可配置 TTL，默认 12h）
   独立于 seen_items，防止拒绝误判造成永久压制。
5) pending_items: 已发现但尚未被真正消费的候选条目 backlog
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.common.timekit import parse_iso as _parse_iso, utcnow as _utcnow
from feeds.base import FeedItem
from infra.persistence.json_store import load_json, save_json
from proactive.item_id import compute_item_id, compute_source_key

logger = logging.getLogger(__name__)


class ProactiveStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load()
        logger.info(
            "[proactive.state] 初始化完成 path=%s seen_sources=%d delivery_sessions=%d semantic_items=%d reject_cool=%d pending_sources=%d",
            self.path,
            len(self._state["seen_items"]),
            len(self._state["deliveries"]),
            len(self._state["semantic_items"]),
            sum(len(v) for v in self._state["rejection_cooldown"].values()),
            len(self._state["pending_items"]),
        )

    def is_item_seen(
        self,
        source_key: str,
        item_id: str,
        ttl_hours: int,
        now: datetime | None = None,
    ) -> bool:
        now = now or _utcnow()
        source_map = self._state["seen_items"].get(source_key, {})
        ts = _parse_iso(source_map.get(item_id))
        if ts is None:
            return False
        if ts < now - timedelta(hours=max(ttl_hours, 1)):
            logger.info(
                "[proactive.state] item 过期，视为未见 source=%s item_id=%s ts=%s ttl_hours=%d",
                source_key,
                item_id[:16],
                source_map.get(item_id),
                ttl_hours,
            )
            return False
        return True

    def mark_items_seen(
        self,
        entries: list[tuple[str, str]],
        now: datetime | None = None,
    ) -> None:
        if not entries:
            logger.info("[proactive.state] mark_items_seen: entries 为空，跳过")
            return
        now = now or _utcnow()
        ts = now.isoformat()
        added = 0
        for source_key, item_id in entries:
            source_map = self._state["seen_items"].setdefault(source_key, {})
            if item_id not in source_map:
                added += 1
            source_map[item_id] = ts
        self._save()
        logger.debug(
            "[proactive.state] 已记录已见条目 count=%d newly_added=%d ts=%s",
            len(entries),
            added,
            ts,
        )

    def is_delivery_duplicate(
        self,
        session_key: str,
        delivery_key: str,
        window_hours: int,
        now: datetime | None = None,
    ) -> bool:
        now = now or _utcnow()
        sess = self._state["deliveries"].get(session_key, {})
        ts = _parse_iso(sess.get(delivery_key))
        if ts is None:
            return False
        if ts >= now - timedelta(hours=max(window_hours, 1)):
            logger.info(
                "[proactive.state] 命中发送去重 session=%s delivery_key=%s ts=%s window_hours=%d",
                session_key,
                delivery_key[:16],
                sess.get(delivery_key),
                window_hours,
            )
            return True
        return False

    def mark_delivery(
        self,
        session_key: str,
        delivery_key: str,
        now: datetime | None = None,
    ) -> None:
        now = now or _utcnow()
        ts = now.isoformat()
        sess = self._state["deliveries"].setdefault(session_key, {})
        sess[delivery_key] = ts
        self._save()
        logger.info(
            "[proactive.state] 已记录发送 session=%s delivery_key=%s ts=%s",
            session_key,
            delivery_key[:16],
            ts,
        )

    def count_deliveries_in_window(
        self,
        session_key: str,
        window_hours: int,
        now: datetime | None = None,
    ) -> int:
        """统计指定 session 在最近窗口内的发送次数。"""
        now = now or _utcnow()
        cutoff = now - timedelta(hours=max(window_hours, 1))
        sess = self._state["deliveries"].get(session_key, {})
        count = 0
        for raw_ts in sess.values():
            ts = _parse_iso(raw_ts)
            if ts and ts >= cutoff:
                count += 1
        return count

    def get_semantic_items(
        self,
        window_hours: int,
        max_candidates: int,
        now: datetime | None = None,
    ) -> list[dict[str, str]]:
        now = now or _utcnow()
        cutoff = now - timedelta(hours=max(window_hours, 1))
        items: list[dict[str, str]] = []
        for raw in self._state["semantic_items"]:
            ts = _parse_iso(str(raw.get("ts", "")))
            if ts is None or ts < cutoff:
                continue
            text = str(raw.get("text", "")).strip()
            if not text:
                continue
            items.append(
                {
                    "source_key": str(raw.get("source_key", "")),
                    "item_id": str(raw.get("item_id", "")),
                    "text": text,
                    "ts": ts.isoformat(),
                }
            )
        items.sort(key=lambda x: x["ts"], reverse=True)
        limited = items[: max(max_candidates, 1)]
        logger.info(
            "[proactive.state] 语义候选加载 total=%d within_window=%d returned=%d window_hours=%d",
            len(self._state["semantic_items"]),
            len(items),
            len(limited),
            window_hours,
        )
        return limited

    def mark_semantic_items(
        self,
        entries: list[dict[str, str]],
        now: datetime | None = None,
    ) -> None:
        if not entries:
            logger.info("[proactive.state] mark_semantic_items: entries 为空，跳过")
            return
        now = now or _utcnow()
        ts = now.isoformat()
        base = self._state["semantic_items"]
        added = 0
        for entry in entries:
            text = str(entry.get("text", "")).strip()
            if not text:
                continue
            base.append(
                {
                    "source_key": str(entry.get("source_key", "")),
                    "item_id": str(entry.get("item_id", "")),
                    "text": text,
                    "ts": ts,
                }
            )
            added += 1
        if added <= 0:
            logger.info("[proactive.state] mark_semantic_items: 无有效文本，跳过")
            return
        self._save()
        logger.info(
            "[proactive.state] 已记录语义条目 count=%d ts=%s",
            added,
            ts,
        )

    # ── pending backlog ───────────────────────────────────────────

    def upsert_pending_items(
        self,
        items: list[FeedItem],
        *,
        max_per_source: int = 20,
        max_total: int = 200,
        now: datetime | None = None,
    ) -> None:
        if not items:
            logger.info("[proactive.state] upsert_pending_items: items 为空，跳过")
            return
        now = now or _utcnow()
        ts = now.isoformat()
        inserted = 0
        updated = 0
        for item in items:
            source_key = compute_source_key(item)
            item_id = compute_item_id(item)
            source_map = self._state["pending_items"].setdefault(source_key, {})
            record = source_map.get(item_id)
            if record is None:
                source_map[item_id] = {
                    "payload": self._serialize_item(item),
                    "first_seen_at": ts,
                    "last_seen_at": ts,
                }
                inserted += 1
                continue
            record["payload"] = self._serialize_item(item)
            record["last_seen_at"] = ts
            updated += 1
        self._enforce_pending_limits(
            max_per_source=max_per_source,
            max_total=max_total,
        )
        self._save()
        logger.info(
            "[proactive.state] pending upsert inserted=%d updated=%d total=%d",
            inserted,
            updated,
            sum(len(v) for v in self._state["pending_items"].values()),
        )

    def list_pending_candidates(
        self,
        limit: int,
        now: datetime | None = None,
    ) -> list[FeedItem]:
        now = now or _utcnow()
        rows: list[tuple[datetime, datetime, FeedItem]] = []
        for source_map in self._state["pending_items"].values():
            for record in source_map.values():
                payload = record.get("payload") or {}
                item = self._deserialize_item(payload)
                if item is None:
                    continue
                published_at = item.published_at or datetime.min.replace(
                    tzinfo=timezone.utc
                )
                first_seen_at = _parse_iso(str(record.get("first_seen_at", ""))) or now
                rows.append((published_at, first_seen_at, item))
        rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
        picked = rows[:limit] if limit > 0 else rows
        return [item for _, _, item in picked]

    def remove_pending_items(
        self,
        entries: list[tuple[str, str]],
    ) -> None:
        if not entries:
            logger.info("[proactive.state] remove_pending_items: entries 为空，跳过")
            return
        removed = 0
        for source_key, item_id in entries:
            source_map = self._state["pending_items"].get(source_key)
            if source_map is None or item_id not in source_map:
                continue
            del source_map[item_id]
            removed += 1
            if not source_map:
                del self._state["pending_items"][source_key]
        if removed <= 0:
            logger.info("[proactive.state] remove_pending_items: 无匹配项")
            return
        self._save()
        logger.info("[proactive.state] pending 已移除 count=%d", removed)

    def pending_stats(self) -> dict[str, int]:
        return {
            source_key: len(source_map)
            for source_key, source_map in self._state["pending_items"].items()
        }

    # ── rejection_cooldown ────────────────────────────────────────

    def is_rejection_cooled(
        self,
        source_key: str,
        item_id: str,
        ttl_hours: int,
        now: datetime | None = None,
    ) -> bool:
        """LLM 拒绝冷却中返回 True；ttl_hours≤0 表示禁用，始终返回 False。"""
        if ttl_hours <= 0:
            return False
        now = now or _utcnow()
        source_map = self._state["rejection_cooldown"].get(source_key, {})
        ts = _parse_iso(source_map.get(item_id))
        if ts is None:
            return False
        if ts < now - timedelta(hours=ttl_hours):
            return False
        return True

    def mark_rejection_cooldown(
        self,
        entries: list[tuple[str, str]],
        hours: int,
        now: datetime | None = None,
    ) -> None:
        """记录 LLM 拒绝冷却；hours≤0 或 entries 为空时跳过。"""
        if hours <= 0 or not entries:
            return
        now = now or _utcnow()
        ts = now.isoformat()
        added = 0
        for source_key, item_id in entries:
            source_map = self._state["rejection_cooldown"].setdefault(source_key, {})
            if item_id not in source_map:
                added += 1
            source_map[item_id] = ts
        self._save()
        logger.info(
            "[proactive.state] 拒绝冷却已记录 count=%d newly_added=%d ttl_hours=%d",
            len(entries),
            added,
            hours,
        )

    def cleanup(
        self,
        seen_ttl_hours: int,
        delivery_ttl_hours: int,
        semantic_ttl_hours: int,
        rejection_cooldown_ttl_hours: int = 0,
        pending_ttl_hours: int = 24,
    ) -> None:
        now = _utcnow()
        seen_cutoff = now - timedelta(hours=max(seen_ttl_hours, 1))
        delivery_cutoff = now - timedelta(hours=max(delivery_ttl_hours, 1))
        semantic_cutoff = now - timedelta(hours=max(semantic_ttl_hours, 1))
        pending_cutoff = now - timedelta(hours=max(pending_ttl_hours, 1))

        removed_seen = 0
        for source_key in list(self._state["seen_items"].keys()):
            source_map = self._state["seen_items"][source_key]
            for item_id in list(source_map.keys()):
                ts = _parse_iso(source_map[item_id])
                if ts is None or ts < seen_cutoff:
                    del source_map[item_id]
                    removed_seen += 1
            if not source_map:
                del self._state["seen_items"][source_key]

        removed_delivery = 0
        for session_key in list(self._state["deliveries"].keys()):
            sess = self._state["deliveries"][session_key]
            for delivery_key in list(sess.keys()):
                ts = _parse_iso(sess[delivery_key])
                if ts is None or ts < delivery_cutoff:
                    del sess[delivery_key]
                    removed_delivery += 1
            if not sess:
                del self._state["deliveries"][session_key]

        before_semantic = len(self._state["semantic_items"])
        self._state["semantic_items"] = [
            x
            for x in self._state["semantic_items"]
            if (
                _parse_iso(str(x.get("ts", "")))
                or datetime.min.replace(tzinfo=timezone.utc)
            )
            >= semantic_cutoff
            and str(x.get("text", "")).strip()
        ]
        removed_semantic = before_semantic - len(self._state["semantic_items"])

        removed_cooldown = 0
        if rejection_cooldown_ttl_hours > 0:
            cooldown_cutoff = now - timedelta(hours=rejection_cooldown_ttl_hours)
            for source_key in list(self._state["rejection_cooldown"].keys()):
                source_map = self._state["rejection_cooldown"][source_key]
                for item_id in list(source_map.keys()):
                    ts = _parse_iso(source_map[item_id])
                    if ts is None or ts < cooldown_cutoff:
                        del source_map[item_id]
                        removed_cooldown += 1
                if not source_map:
                    del self._state["rejection_cooldown"][source_key]

        removed_pending = 0
        for source_key in list(self._state["pending_items"].keys()):
            source_map = self._state["pending_items"][source_key]
            for item_id in list(source_map.keys()):
                record = source_map[item_id] or {}
                payload = record.get("payload") or {}
                item_ts = _parse_iso(payload.get("published_at")) or _parse_iso(
                    str(record.get("first_seen_at", ""))
                )
                if item_ts is None or item_ts < pending_cutoff:
                    del source_map[item_id]
                    removed_pending += 1
            if not source_map:
                del self._state["pending_items"][source_key]

        self._save()
        logger.debug(
            "[proactive.state] cleanup 完成 removed_seen=%d removed_delivery=%d removed_semantic=%d removed_cooldown=%d removed_pending=%d",
            removed_seen,
            removed_delivery,
            removed_semantic,
            removed_cooldown,
            removed_pending,
        )

    def get_bg_context_last_main_at(self) -> datetime | None:
        """返回上次以 background_context 为主 topic 发送的时间，若无则 None。"""
        raw = self._state.get("bg_context_last_main_at")
        return _parse_iso(raw) if raw else None

    def mark_bg_context_main_send(self, now: datetime | None = None) -> None:
        """记录本次以 background_context 为主 topic 发送的时间。"""
        now = now or _utcnow()
        self._state["bg_context_last_main_at"] = now.isoformat()
        self._save()
        logger.info("[proactive.state] bg_context 主 topic 发送已记录 ts=%s", now.isoformat())

    def _load(self) -> dict[str, Any]:
        # 1. 从磁盘读取原始数据
        raw = load_json(self.path, default=None, domain="proactive.state")
        if raw is None:
            return {
                "version": 4,
                "seen_items": {},
                "deliveries": {},
                "semantic_items": [],
                "rejection_cooldown": {},
                "pending_items": {},
                "bg_context_last_main_at": None,
            }

        # 2. 规范化字段（向后兼容）
        state: dict[str, Any] = {
            "version": int(raw.get("version", 3)),
            "seen_items": dict(raw.get("seen_items", {})),
            "deliveries": dict(raw.get("deliveries", {})),
            "semantic_items": list(raw.get("semantic_items", [])),
            "rejection_cooldown": dict(raw.get("rejection_cooldown", {})),
            "pending_items": self._normalize_pending_items(
                raw.get("pending_items", {})
            ),
            "bg_context_last_main_at": raw.get("bg_context_last_main_at"),
        }
        state["version"] = 4
        logger.info("[proactive.state] 从磁盘加载状态成功 path=%s", self.path)
        return state

    def _save(self) -> None:
        save_json(self.path, self._state, domain="proactive.state")
        logger.debug("[proactive.state] 状态已保存 path=%s", self.path)

    def _serialize_item(self, item: FeedItem) -> dict[str, Any]:
        return {
            "source_name": item.source_name,
            "source_type": item.source_type,
            "title": item.title,
            "content": item.content,
            "url": item.url,
            "author": item.author,
            "published_at": (
                item.published_at.isoformat() if item.published_at else None
            ),
        }

    def _deserialize_item(self, payload: dict[str, Any]) -> FeedItem | None:
        try:
            published_at = _parse_iso(payload.get("published_at"))
            return FeedItem(
                source_name=str(payload.get("source_name", "")),
                source_type=str(payload.get("source_type", "")),
                title=payload.get("title"),
                content=str(payload.get("content", "")),
                url=payload.get("url"),
                author=payload.get("author"),
                published_at=published_at,
            )
        except Exception:
            return None

    def _normalize_pending_items(
        self, raw_pending: Any
    ) -> dict[str, dict[str, dict[str, Any]]]:
        if not isinstance(raw_pending, dict):
            return {}
        normalized: dict[str, dict[str, dict[str, Any]]] = {}
        for source_key, source_map in raw_pending.items():
            if not isinstance(source_map, dict):
                continue
            source_key_str = str(source_key)
            norm_source: dict[str, dict[str, Any]] = {}
            for item_id, record in source_map.items():
                if not isinstance(record, dict):
                    continue
                payload = record.get("payload")
                if not isinstance(payload, dict):
                    continue
                norm_source[str(item_id)] = {
                    "payload": payload,
                    "first_seen_at": str(record.get("first_seen_at", "")),
                    "last_seen_at": str(record.get("last_seen_at", "")),
                }
            if norm_source:
                normalized[source_key_str] = norm_source
        return normalized

    def _enforce_pending_limits(
        self,
        *,
        max_per_source: int,
        max_total: int,
    ) -> None:
        max_per_source = max(max_per_source, 1)
        max_total = max(max_total, 1)

        # 先按 source 限流，优先保留发布时间/首次发现时间较新的条目。
        for source_key in list(self._state["pending_items"].keys()):
            source_map = self._state["pending_items"][source_key]
            rows: list[tuple[datetime, datetime, str]] = []
            for item_id, record in source_map.items():
                item = self._deserialize_item(record.get("payload") or {})
                if item is None:
                    continue
                published_at = item.published_at or datetime.min.replace(
                    tzinfo=timezone.utc
                )
                first_seen_at = _parse_iso(str(record.get("first_seen_at", ""))) or (
                    datetime.min.replace(tzinfo=timezone.utc)
                )
                rows.append((published_at, first_seen_at, item_id))
            rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
            for _, _, item_id in rows[max_per_source:]:
                del source_map[item_id]
            if not source_map:
                del self._state["pending_items"][source_key]

        rows_all: list[tuple[datetime, datetime, str, str]] = []
        for source_key, source_map in self._state["pending_items"].items():
            for item_id, record in source_map.items():
                item = self._deserialize_item(record.get("payload") or {})
                if item is None:
                    continue
                published_at = item.published_at or datetime.min.replace(
                    tzinfo=timezone.utc
                )
                first_seen_at = _parse_iso(str(record.get("first_seen_at", ""))) or (
                    datetime.min.replace(tzinfo=timezone.utc)
                )
                rows_all.append((published_at, first_seen_at, source_key, item_id))
        rows_all.sort(key=lambda row: (row[0], row[1]), reverse=True)
        for _, _, source_key, item_id in rows_all[max_total:]:
            source_map = self._state["pending_items"].get(source_key)
            if source_map is None:
                continue
            source_map.pop(item_id, None)
            if not source_map:
                del self._state["pending_items"][source_key]
