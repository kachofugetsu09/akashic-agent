from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent.plugins import Plugin, tool
from agent.plugins.proactive_effects import (
    ProactiveEffect,
    ProactiveEffectContext,
    ProactiveEffectProvider,
)
from plugins.proactive_feedback.events import ProactiveFeedbackRecorded

from .db import apply_feedback, build_effect, get_state, open_db

logger = logging.getLogger("plugin.emotion")


class EmotionPlugin(Plugin):
    name = "emotion"

    async def initialize(self) -> None:
        workspace = self.context.workspace
        if workspace is None:
            logger.warning("emotion 插件缺少 workspace，跳过加载")
            return
        self._db_path = workspace / "emotion" / "emotion.db"
        conn = open_db(self._db_path)
        conn.close()
        self.context.event_bus.on(ProactiveFeedbackRecorded, self._on_feedback_recorded)

    async def terminate(self) -> None:
        return None

    def proactive_effect_providers(self) -> list[ProactiveEffectProvider]:
        return [self]

    def build_proactive_effect(
        self,
        ctx: ProactiveEffectContext,
    ) -> ProactiveEffect | None:
        db_path = getattr(self, "_db_path", None)
        if db_path is None:
            return None
        conn = open_db(Path(db_path))
        try:
            return build_effect(
                conn,
                tick_id=ctx.tick_id,
                session_key=ctx.session_key,
                now_utc=ctx.now_utc,
                last_user_at=ctx.last_user_at,
                base_threshold=ctx.base_judge_send_threshold,
            )
        finally:
            conn.close()

    def _on_feedback_recorded(self, event: ProactiveFeedbackRecorded) -> None:
        db_path = getattr(self, "_db_path", None)
        if db_path is None:
            return
        feedback = event.feedback
        payload: dict[str, Any] = {
            "feedback_event_id": event.event_id,
            "user_message_id": feedback.user_message_id,
            "assistant_message_id": feedback.assistant_message_id,
            "proactive_message_id": feedback.proactive_message_id,
            "feedback_type": feedback.feedback_type,
            "confidence": feedback.confidence,
            "pua_score": feedback.pua_score,
            "lag_seconds": feedback.lag_seconds,
            "matched_by": feedback.matched_by,
        }
        conn = open_db(Path(db_path))
        try:
            _ = apply_feedback(
                conn,
                source_event_id=f"proactive_feedback:{event.event_id}",
                session_key=feedback.session_key,
                feedback_type=feedback.feedback_type,
                confidence=feedback.confidence,
                payload=payload,
            )
        finally:
            conn.close()

    @tool(
        "get_emotion_state",
        risk="read-only",
        search_hint="查询 proactive VAD 情绪状态",
    )
    async def get_emotion_state(self, event: Any) -> dict[str, Any]:
        """查询 proactive VAD 情绪状态。"""
        _ = event
        db_path = getattr(self, "_db_path", None)
        if db_path is None:
            return {"available": False}
        conn = open_db(Path(db_path))
        try:
            state = get_state(conn)
        finally:
            conn.close()
        return {
            "available": True,
            "valence": state.valence,
            "arousal": state.arousal,
            "dominance": state.dominance,
            "updated_at": state.updated_at,
        }
