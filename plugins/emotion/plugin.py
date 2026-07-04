from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent.plugins import EventTrigger, Plugin, PluginJobContext, PluginJobSpec, tool
from bus.events_lifecycle import DriftFinished
from proactive_v2.frame import ProactiveFrame
from plugins.proactive_feedback.events import ProactiveFeedbackRecorded

from .db import apply_feedback, build_effect, get_state, open_db

logger = logging.getLogger("plugin.emotion")

_FEEDBACK_CONTEXT_SKILL = "emotion:feedback-preference-context"
_PROACTIVE_CONTEXT_TEMPLATE = """# Proactive Context

在这里写会影响未来主动推送取舍的稳定偏好。

- 主 agent 负责维护这份文件。
- proactive agent 每轮都会读取它作为额外上下文。
- 优先写短规则和倾向,避免写流程文档。
- 这里不提供新闻事实,不提供候选内容。
- 写结论即可,不要写冗长过程。
"""

_MERGE_PROACTIVE_CONTEXT_SYSTEM = (
    "你是 proactive context editor。"
    "你的职责是把候选反馈保守蒸馏成短规则，帮助未来主动推送更合适。"
    "不要把一次反馈扩写成策略文档，也不要制造新的硬约束。"
)

_MERGE_PROACTIVE_CONTEXT_PROMPT = """\
你的任务是把「待合并推送偏好候选」融合进「当前 Proactive Context」。

## 合并原则
- 只保留会稳定影响未来主动推送决策的偏好。
- 多条反馈支持同一方向时，可以写成规则；单条或弱证据只能写成轻量倾向。
- 把 pending 聚类成少量稳定主题；37 条候选也最多沉淀成 8 条新增或修改规则。
- pending 中 effect=boost/block/verify/timing/tone 分别对应提高优先级、降低/屏蔽、推送前核验、时机、表达方式。
- 合并同类项，删除重复、过窄、证据不足、no_candidate、临时状态和一次性事件。
- 优先修改现有相关 bullet，不要为轻量倾向新建 section。
- 保留当前 Proactive Context 中仍然有效的规则，但可以压缩被 pending 触及的冗长段落。

## 禁止事项
- 不写 evidence id、feedback id、message id、chunk、计数、推理过程或审核状态。
- 不写“数据来源”“触发条件”“执行注意”“计算逻辑”“白名单每周更新”这类流程说明。
- 不因为一条反馈就写“仅推送”“禁止”“一律过滤”“必须查询”这类硬规则。
- 不扩写成用户画像、聊天总结、新闻事实或配置文档。
- 不新增带“新增”“其他倾向”“通用规则”这类兜底标题的小节。

## 输出格式
- 直接输出完整 `# Proactive Context` markdown。
- 新增或修改的规则尽量一行一个 bullet，必要时最多两行。
- 全文长度不要超过当前 Proactive Context 的 1.25 倍。
- 优先使用“优先/降权/避免/保持/可适度”这类运行时上下文表达。
- 不要代码块，不要解释。

---
当前 Proactive Context：
{current_context}

待合并推送偏好候选：
{pending}
"""


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

    def jobs(self) -> list[PluginJobSpec]:
        return [
            PluginJobSpec(
                id="merge_proactive_pending",
                triggers=[EventTrigger(DriftFinished)],
                handler=self.merge_proactive_pending,
            )
        ]

    async def merge_proactive_pending(self, ctx: PluginJobContext) -> None:
        event = ctx.event
        if not isinstance(event, DriftFinished):
            return
        if event.skill_name != _FEEDBACK_CONTEXT_SKILL or event.status != "completed":
            return
        workspace = self.context.workspace
        if workspace is None:
            return

        pending_path = workspace / "proactive_pending.md"
        context_path = workspace / "PROACTIVE_CONTEXT.md"
        pending = self._read_text(pending_path).strip()
        if not pending or "- [ ]" not in pending:
            return

        current_context = self._read_text(context_path).strip()
        if not current_context:
            current_context = _PROACTIVE_CONTEXT_TEMPLATE.strip()
        prompt = _MERGE_PROACTIVE_CONTEXT_PROMPT.format(
            current_context=current_context,
            pending=pending,
        )
        merged = await ctx.llm.generate_text(
            system=_MERGE_PROACTIVE_CONTEXT_SYSTEM,
            prompt=prompt,
            max_tokens=4096,
        )
        if not merged:
            return
        _ = context_path.write_text(merged.strip() + "\n", encoding="utf-8")
        _ = pending_path.write_text("", encoding="utf-8")
        logger.info("emotion proactive pending 已合并到 PROACTIVE_CONTEXT.md")

    @staticmethod
    def _read_text(path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

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
