from __future__ import annotations

from typing import cast

from agent.lifecycle.types import BeforeTurnCtx, TurnState
from agent.plugins import Plugin


class ChatIdCommandModule:
    requires = ("session:session",)
    produces = ("session:ctx",)

    async def run(self, frame: object) -> object:
        if "session:ctx" in frame.slots:  # type: ignore[attr-defined]
            return frame
        state: TurnState = frame.input  # type: ignore[attr-defined]
        if _normalize_command(state.msg.content) not in {"/chatid", "/myid"}:
            return frame
        chat_id = state.msg.chat_id or "（未知）"
        reply = _format_reply(chat_id)
        frame.slots["session:ctx"] = _abort_ctx(state, reply)  # type: ignore[attr-defined]
        return frame


class SetupHelper(Plugin):
    name = "setup_helper"
    desc = "快速查询当前会话 chat_id，用于配置 proactive"

    def telegram_bot_commands(self) -> list[tuple[str, str]]:
        return [("chatid", "查看我的 chat_id（配置 proactive 用）")]

    def before_turn_modules_early(self) -> list[object]:
        return cast("list[object]", [ChatIdCommandModule()])


def _normalize_command(content: str) -> str:
    parts = (content or "").strip().split(maxsplit=1)
    if not parts:
        return ""
    head = parts[0].lower()
    if "@" in head:
        head = head.split("@", 1)[0]
    return head


def _format_reply(chat_id: str) -> str:
    return "\n".join([
        f"你的 chat_id 是：`{chat_id}`",
        "",
        "将它填入 config.toml 即可开启主动推送：",
        "",
        "```toml",
        "[proactive]",
        "enabled = true",
        "",
        "[proactive.target]",
        "channel = \"telegram\"",
        f"chat_id = \"{chat_id}\"",
        "```",
    ])


def _abort_ctx(state: TurnState, reply: str) -> BeforeTurnCtx:
    return BeforeTurnCtx(
        session_key=state.session_key,
        channel=state.msg.channel,
        chat_id=state.msg.chat_id,
        content=state.msg.content,
        timestamp=state.msg.timestamp,
        skill_names=[],
        retrieved_memory_block="",
        retrieval_trace_raw=None,
        history_messages=(),
        abort=True,
        abort_reply=reply,
    )
