"""
ProactiveStateStore — 主动消息流的最小持久化状态。

状态文件只保留四类主链路真实使用的数据：
1) seen_items: 每个 source 下已处理过的 item_id（长 TTL）
2) deliveries: 每个 session 下已发送过的 delivery_key
3) semantic_items: 发送前后用于语义去重的历史文本
4) rejection_cooldown: LLM 拒绝后的短期冷却
5) bg_context_last_main_at: 纯背景感知触达的主 topic 冷却时间
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.common.timekit import parse_iso as _parse_iso, utcnow as _utcnow
from infra.persistence.json_store import load_json, save_json

logger = logging.getLogger(__name__)


def _dedupe_source_key(source_key: str) -> str:
    """归一化用于状态去重的 source_key。

    MCP 内容流会把 ack 用的 event_id 编进 source_key（mcp:<server>:<event_id>）。
    去重状态只应按稳定来源维度记账，否则同一内容换一个 event_id 会绕过去重。
    """
    raw = str(source_key or "").strip()
    if not raw.startswith("mcp:"):
        return raw
    parts = raw.split(":", 2)
    if len(parts) < 2:
        return raw
    return ":".join(parts[:2])


class ProactiveStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load()
        logger.info(
            "[proactive.state] 初始化完成 path=%s seen_sources=%d delivery_sessions=%d semantic_items=%d reject_cool=%d",
            self.path,
            len(self._state["seen_items"]),
            len(self._state["deliveries"]),
            len(self._state["semantic_items"]),
            sum(len(v) for v in self._state["rejection_cooldown"].values()),
        )

    def is_item_seen(
        self,
        source_key: str,
        item_id: str,
        ttl_hours: int,
        now: datetime | None = None,
    ) -> bool:
        now = now or _utcnow()
        dedupe_key = _dedupe_source_key(source_key)
        source_map = self._state["seen_items"].get(dedupe_key, {})
        ts = _parse_iso(source_map.get(item_id))
        if ts is None:
            return False
        if ts < now - timedelta(hours=max(ttl_hours, 1)):
            logger.info(
                "[proactive.state] item 过期，视为未见 source=%s item_id=%s ts=%s ttl_hours=%d",
                dedupe_key,
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
            return
        now = now or _utcnow()
        ts = now.isoformat()
        added = 0
        for source_key, item_id in entries:
            dedupe_key = _dedupe_source_key(source_key)
            source_map = self._state["seen_items"].setdefault(dedupe_key, {})
            if item_id not in source_map:
                added += 1
            source_map[item_id] = ts
        self._save()
        logger.info(
            "[proactive.state] 已标记已见条目 count=%d newly_added=%d entries=%s",
            len(entries),
            added,
            [(sk, iid[:16]) for sk, iid in entries[:3]],
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
        if ts < now - timedelta(hours=max(window_hours, 1)):
            return False
        logger.info(
            "[proactive.state] 命中发送去重 session=%s delivery_key=%s ts=%s window_hours=%d",
            session_key,
            delivery_key[:16],
            sess.get(delivery_key),
            window_hours,
        )
        return True

    def mark_delivery(
        self,
        session_key: str,
        delivery_key: str,
        now: datetime | None = None,
    ) -> None:
        now = now or _utcnow()
        ts = now.isoformat()
        self._state["deliveries"].setdefault(session_key, {})[delivery_key] = ts
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
        now = now or _utcnow()
        cutoff = now - timedelta(hours=window_hours)
        count = 0
        for raw_ts in self._state["deliveries"].get(session_key, {}).values():
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
        cutoff = now - timedelta(hours=window_hours)
        items: list[dict[str, str]] = []
        for raw in self._state["semantic_items"]:
            ts = _parse_iso(str(raw.get("ts", "")))
            text = str(raw.get("text", "")).strip()
            if ts is None or ts < cutoff or not text:
                continue
            items.append(
                {
                    "source_key": str(raw.get("source_key", "")),
                    "item_id": str(raw.get("item_id", "")),
                    "text": text,
                    "ts": ts.isoformat(),
                }
            )
        items.sort(key=lambda item: item["ts"], reverse=True)
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
            return
        now = now or _utcnow()
        ts = now.isoformat()
        added = 0
        for entry in entries:
            text = str(entry.get("text", "")).strip()
            if not text:
                continue
            self._state["semantic_items"].append(
                {
                    "source_key": str(entry.get("source_key", "")),
                    "item_id": str(entry.get("item_id", "")),
                    "text": text,
                    "ts": ts,
                }
            )
            added += 1
        if added <= 0:
            return
        self._save()
        logger.info("[proactive.state] 已记录语义条目 count=%d ts=%s", added, ts)

    def is_rejection_cooled(
        self,
        source_key: str,
        item_id: str,
        ttl_hours: int,
        now: datetime | None = None,
    ) -> bool:
        if ttl_hours <= 0:
            return False
        now = now or _utcnow()
        dedupe_key = _dedupe_source_key(source_key)
        source_map = self._state["rejection_cooldown"].get(dedupe_key, {})
        ts = _parse_iso(source_map.get(item_id))
        if ts is None:
            return False
        return ts >= now - timedelta(hours=ttl_hours)

    def mark_rejection_cooldown(
        self,
        entries: list[tuple[str, str]],
        hours: int,
        now: datetime | None = None,
    ) -> None:
        if hours <= 0 or not entries:
            return
        now = now or _utcnow()
        ts = now.isoformat()
        added = 0
        for source_key, item_id in entries:
            dedupe_key = _dedupe_source_key(source_key)
            source_map = self._state["rejection_cooldown"].setdefault(dedupe_key, {})
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
    ) -> None:
        now = _utcnow()
        removed_seen = self._cleanup_nested_map(
            "seen_items",
            now - timedelta(hours=max(seen_ttl_hours, 1)),
        )
        removed_delivery = self._cleanup_nested_map(
            "deliveries",
            now - timedelta(hours=max(delivery_ttl_hours, 1)),
        )
        removed_semantic = self._cleanup_semantic_items(
            now - timedelta(hours=max(semantic_ttl_hours, 1))
        )
        removed_cooldown = 0
        if rejection_cooldown_ttl_hours > 0:
            removed_cooldown = self._cleanup_nested_map(
                "rejection_cooldown",
                now - timedelta(hours=rejection_cooldown_ttl_hours),
            )
        # 清理 context-only 时间戳（24h 窗口）
        removed_context_only = self._cleanup_context_only_timestamps(24, now)
        self._save()
        logger.debug(
            "[proactive.state] cleanup 完成 removed_seen=%d removed_delivery=%d removed_semantic=%d removed_cooldown=%d removed_context_only=%d",
            removed_seen,
            removed_delivery,
            removed_semantic,
            removed_cooldown,
            removed_context_only,
        )

    def _cleanup_context_only_timestamps(
        self, window_hours: int, now: datetime
    ) -> int:
        """清理过期的 context-only 时间戳。"""
        cutoff = now - timedelta(hours=window_hours)
        removed = 0
        for session_key in list(
            self._state.get("context_only_sent_timestamps", {}).keys()
        ):
            timestamps = self._state["context_only_sent_timestamps"][session_key]
            before = len(timestamps)
            self._state["context_only_sent_timestamps"][session_key] = [
                ts
                for ts in timestamps
                if (_parse_iso(ts) or datetime.min.replace(tzinfo=timezone.utc))
                >= cutoff
            ]
            after = len(self._state["context_only_sent_timestamps"][session_key])
            removed += before - after
            if not self._state["context_only_sent_timestamps"][session_key]:
                del self._state["context_only_sent_timestamps"][session_key]
        return removed

    def get_bg_context_last_main_at(self) -> datetime | None:
        raw = self._state.get("bg_context_last_main_at")
        return _parse_iso(raw) if raw else None

    def mark_bg_context_main_send(self, now: datetime | None = None) -> None:
        now = now or _utcnow()
        self._state["bg_context_last_main_at"] = now.isoformat()
        self._save()
        logger.info(
            "[proactive.state] bg_context 主 topic 发送已记录 ts=%s",
            now.isoformat(),
        )

    # ── context-only 配额管理 ──────────────────────────────────────────

    def get_last_context_only_at(self, session_key: str) -> datetime | None:
        """获取指定 session 最后一次 context-only 发送时间。"""
        raw = self._state.get("context_only_last_at", {}).get(session_key)
        return _parse_iso(raw) if raw else None

    def mark_context_only_send(
        self, session_key: str, now: datetime | None = None
    ) -> None:
        """记录 context-only 发送时间。"""
        now = now or _utcnow()
        ts = now.isoformat()
        self._state.setdefault("context_only_last_at", {})[session_key] = ts
        self._state.setdefault("context_only_sent_timestamps", {}).setdefault(
            session_key, []
        ).append(ts)
        self._save()
        logger.info(
            "[proactive.state] context-only 发送已记录 session=%s ts=%s",
            session_key,
            ts,
        )

    def count_context_only_in_window(
        self, session_key: str, window_hours: int, now: datetime | None = None
    ) -> int:
        """统计指定 session 在窗口内的 context-only 发送次数。"""
        now = now or _utcnow()
        cutoff = now - timedelta(hours=window_hours)
        count = 0
        for raw_ts in self._state.get("context_only_sent_timestamps", {}).get(
            session_key, []
        ):
            ts = _parse_iso(raw_ts)
            if ts and ts >= cutoff:
                count += 1
        return count

    def _cleanup_nested_map(self, key: str, cutoff: datetime) -> int:
        removed = 0
        for outer_key in list(self._state[key].keys()):
            inner_map = self._state[key][outer_key]
            for inner_key in list(inner_map.keys()):
                ts = _parse_iso(inner_map[inner_key])
                if ts is None or ts < cutoff:
                    del inner_map[inner_key]
                    removed += 1
            if not inner_map:
                del self._state[key][outer_key]
        return removed

    def _cleanup_semantic_items(self, cutoff: datetime) -> int:
        before = len(self._state["semantic_items"])
        self._state["semantic_items"] = [
            row
            for row in self._state["semantic_items"]
            if (_parse_iso(str(row.get("ts", ""))) or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff
            and str(row.get("text", "")).strip()
        ]
        return before - len(self._state["semantic_items"])

    def _load(self) -> dict[str, Any]:
        raw = load_json(self.path, default=None, domain="proactive.state")
        if raw is None:
            return self._empty_state()
        state = self._empty_state()
        state["version"] = int(raw.get("version", 5))
        state["seen_items"] = self._normalize_nested_map(raw.get("seen_items"))
        state["deliveries"] = self._normalize_nested_map(raw.get("deliveries"))
        state["rejection_cooldown"] = self._normalize_nested_map(
            raw.get("rejection_cooldown")
        )
        state["semantic_items"] = self._normalize_semantic_items(
            raw.get("semantic_items", [])
        )
        state["bg_context_last_main_at"] = raw.get("bg_context_last_main_at")
        # context_only_last_at 是 dict[session_key, iso_ts]，不是嵌套 map
        raw_last_at = raw.get("context_only_last_at", {})
        if isinstance(raw_last_at, dict):
            state["context_only_last_at"] = {str(k): str(v) for k, v in raw_last_at.items() if v}
        else:
            state["context_only_last_at"] = {}
        state["context_only_sent_timestamps"] = self._normalize_context_only_timestamps(
            raw.get("context_only_sent_timestamps", {})
        )
        logger.info("[proactive.state] 从磁盘加载状态成功 path=%s", self.path)
        return state

    def _save(self) -> None:
        save_json(self.path, self._state, domain="proactive.state")
        logger.debug("[proactive.state] 状态已保存 path=%s", self.path)

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "version": 5,
            "seen_items": {},
            "deliveries": {},
            "semantic_items": [],
            "rejection_cooldown": {},
            "bg_context_last_main_at": None,
            "context_only_last_at": {},
            "context_only_sent_timestamps": {},
        }

    @staticmethod
    def _normalize_nested_map(raw: Any) -> dict[str, dict[str, str]]:
        if not isinstance(raw, dict):
            return {}
        normalized: dict[str, dict[str, str]] = {}
        for outer_key, inner_raw in raw.items():
            if not isinstance(inner_raw, dict):
                continue
            inner: dict[str, str] = {}
            for inner_key, value in inner_raw.items():
                text = str(value or "").strip()
                if text:
                    inner[str(inner_key)] = text
            if inner:
                normalized[str(outer_key)] = inner
        return normalized

    @staticmethod
    def _normalize_semantic_items(raw: Any) -> list[dict[str, str]]:
        if not isinstance(raw, list):
            return []
        normalized: list[dict[str, str]] = []
        for row in raw:
            if not isinstance(row, dict):
                continue
            normalized.append(
                {
                    "source_key": str(row.get("source_key", "")),
                    "item_id": str(row.get("item_id", "")),
                    "text": str(row.get("text", "")),
                    "ts": str(row.get("ts", "")),
                }
            )
        return normalized

    @staticmethod
    def _normalize_context_only_timestamps(raw: Any) -> dict[str, list[str]]:
        """规范化 context_only_sent_timestamps 结构。"""
        if not isinstance(raw, dict):
            return {}
        normalized: dict[str, list[str]] = {}
        for session_key, timestamps in raw.items():
            if not isinstance(timestamps, list):
                continue
            normalized[str(session_key)] = [str(ts) for ts in timestamps if ts]
        return normalized
