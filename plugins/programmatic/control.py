from __future__ import annotations

from typing import cast

from pydantic import Field

from agent.control.protocol.models import StrictModel, SessionIdParams
from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.messages import MESSAGE_CATALOG, SESSION_ADMISSION
from plugins.turn_projection.plugin import TURN_PROJECTION
from session.log import SessionAttributes
from session.message import ContentPart, Input

from .result import read_result


class AdmitParams(SessionIdParams):
    persist_memory: bool = False


class SendParams(SessionIdParams):
    message_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=1_048_576)


class PauseParams(SessionIdParams):
    message_id: str = Field(min_length=1, max_length=256)


class ResumeParams(PauseParams):
    input_id: str = Field(min_length=1, max_length=256)


class ResultParams(SessionIdParams):
    input_id: str = Field(min_length=1, max_length=256)


PARAMS: dict[str, type[StrictModel]] = {
    "programmatic/session/admit": AdmitParams,
    "programmatic/message/send": SendParams,
    "programmatic/message/pause": PauseParams,
    "programmatic/message/resume": ResumeParams,
    "programmatic/message/result": ResultParams,
}


def check_session(session_id: str) -> None:
    if not session_id.startswith("programmatic:") or not session_id[13:]:
        raise ValueError("程序调用需要 programmatic Session")


class Programmatic:
    """程序来源拥有固定身份和创建属性；读取与回复各用既有能力。"""

    def __init__(self, ctx: Context):
        self.ctx = ctx

    async def call(self, method: str, params: StrictModel) -> dict[str, object]:
        """仅接受声明的 typed 方法；每次调用已由入口绑定一个实际 Root。"""
        from .plugin import open_source

        session_id = cast(SessionIdParams, params).session_id
        check_session(session_id)
        ctx = self.ctx
        # 1. 创建时提交不可变资格；ACK 丢失可用调用方原身份幂等重试。
        if method == "programmatic/session/admit":
            create = cast(AdmitParams, params)
            attributes = ctx.require(SESSION_ADMISSION).ensure(ctx, session_id, SessionAttributes(
                visibility="internal", learning="eligible" if create.persist_memory else "excluded",
            ))
            return {"version": 2, "session_id": session_id, "visibility": attributes.visibility,
                    "learning": attributes.learning}

        # 2. 来源只读取已创建属性，后续输入无改变学习资格的字段。
        if method == "programmatic/message/result":
            reader = ctx.require(MESSAGE_CATALOG).reader(session_id)
            if reader.attributes.visibility != "internal":
                raise ValueError("程序调用 Session 尚未通过内部来源准入")
            return read_result(reader,
                cast(ResultParams, params).input_id, ctx.require(TURN_PROJECTION))
        source = open_source(ctx, session_id)
        if method == "programmatic/message/send":
            send = cast(SendParams, params)
            if not send.text.strip():
                raise ValueError("程序输入不能为空白")
            message = await source.accept(send.message_id, Input((
                ContentPart("text", send.text),
                ContentPart("channel.origin", {"channel": "programmatic", "chat_id": session_id[13:],
                                                "sender": "control"}),
            )))
        elif method == "programmatic/message/pause":
            message = await source.pause(cast(PauseParams, params).message_id)
        elif method == "programmatic/message/resume":
            resume = cast(ResumeParams, params)
            message = await source.resume(resume.message_id, resume.input_id)
        else:
            raise AssertionError("未声明的程序调用方法: " + method)
        return {"version": 2, "session_id": message.session_id,
                "message_id": message.message_id, "seq": message.seq}


PROGRAMMATIC = ServiceKey[Programmatic]("programmatic.v1")
