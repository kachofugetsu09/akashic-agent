from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agent.prompting import is_context_frame
from proactive_v2.presence import PresenceStore
from session.manager import SessionManager


@dataclass
class RecentProactiveMessage:
    content: str
    timestamp: datetime | None = None
    state_summary_tag: str = "none"
    source_refs: list[Any] = field(default_factory=list)


class Sensor:
    def __init__(
        self,
        *,
        cfg: Any,
        sessions: SessionManager,
        presence: PresenceStore | None,
    ) -> None:
        self._cfg = cfg
        self._sessions = sessions
        self._presence = presence

    def target_session_key(self) -> str:
        channel = (self._cfg.default_channel or "").strip()
        chat_id = self._cfg.default_chat_id.strip()
        return f"{channel}:{chat_id}" if channel and chat_id else ""

    def last_user_at(self) -> datetime | None:
        if not self._presence:
            return None
        return self._presence.get_last_user_at(self.target_session_key())

    def collect_recent(self) -> list[dict]:
        session_key = self.target_session_key()
        if not session_key:
            return []
        try:
            session = self._sessions.get_or_create(session_key)
        except Exception:
            return []
        messages = session.messages[-self._cfg.recent_chat_messages :]
        results: list[dict] = []
        for message in messages:
            if message.get("role") not in ("user", "assistant"):
                continue
            if not message.get("content"):
                continue
            content = str(message.get("content", ""))
            if is_context_frame(content):
                continue
            results.append(
                {
                    "role": message["role"],
                    "content": content[:200],
                    "timestamp": str(message.get("timestamp", "")),
                }
            )
        return results

    def collect_recent_proactive(self, n: int = 5) -> list[RecentProactiveMessage]:
        session_key = self.target_session_key()
        if not session_key:
            return []
        try:
            session = self._sessions.get_or_create(session_key)
        except Exception:
            return []
        results: list[RecentProactiveMessage] = []
        for message in reversed(session.messages):
            if message.get("role") != "assistant":
                continue
            if not message.get("proactive") or not message.get("content"):
                continue
            results.append(
                RecentProactiveMessage(
                    content=str(message["content"]),
                    timestamp=self._parse_timestamp(message.get("timestamp")),
                    state_summary_tag=str(
                        message.get("state_summary_tag", "none") or "none"
                    ),
                    source_refs=list(message.get("source_refs") or []),
                )
            )
            if len(results) >= n:
                break
        return list(reversed(results))

    @staticmethod
    def _parse_timestamp(raw: Any) -> datetime | None:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            ts = datetime.fromisoformat(text)
        except Exception:
            return None
        if ts.tzinfo is None:
            return ts.replace(tzinfo=datetime.now().astimezone().tzinfo)
        return ts
