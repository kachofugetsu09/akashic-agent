from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent.plugins import Plugin, tool
from proactive_v2.frame import ProactiveFrame
from plugins.proactive_feedback.events import ProactiveFeedbackRecorded

from .db import apply_feedback, build_effect, get_state, open_db

logger = logging.getLogger("plugin.emotion")


class EmotionProactivePromptModule:
    slot = "proactive.prompt.emotion"
    phase = "proactive.prompt"

    def __init__(self, plugin: "EmotionPlugin") -> None:
        self._plugin = plugin

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        effect = self._plugin.build_proactive_prompt_effect(frame)
        if effect is None:
            return frame
        frame.slots["proactive:prompt:system_bottom:emotion"] = str(
            effect.get("prompt_section") or ""
        )
        frame.slots["proactive:effect:emotion"] = effect
        return frame


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

    def proactive_modules(self) -> list[object]:
        return [EmotionProactivePromptModule(self)]

    def build_proactive_prompt_effect(
        self,
        frame: ProactiveFrame,
    ) -> dict[str, Any] | None:
        db_path = getattr(self, "_db_path", None)
        if db_path is None:
            return None
        conn = open_db(Path(db_path))
        try:
            return build_effect(
                conn,
                tick_id=f"frame:{frame.input.started_at.isoformat()}",
                session_key=str(
                    frame.slots.get("proactive:session_key")
                    or frame.input.session_key
                ),
                now_utc=frame.input.started_at,
                last_user_at=frame.slots.get("proactive:last_user_at"),
                base_threshold=float(
                    frame.slots.get("proactive:base_judge_send_threshold") or 0.60
                ),
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
