from __future__ import annotations

from typing import cast

from pydantic import Field

from agent.control.protocol.models import StrictModel, SessionIdParams
from agent.plugin_composition import Context, ServiceKey
from agent.control.protocol.method import OutputReservation, RequestTransport
from agent.plugin_composition.messages import MESSAGE_CATALOG, SESSION_ADMISSION
from plugins.turn_projection.plugin import TURN_PROJECTION, Turn
from session.log import MessageReader, SessionAttributes
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
        self._reservations: dict[tuple[str, str], tuple[str, OutputReservation]] = {}

    def reserve_input(self, session_id: str, input_id: str, transport: RequestTransport) -> OutputReservation:
        """保存 Input 所属连接，最终 Output 只能从同一连接确认。"""
        key = (session_id, input_id)
        existing = self._reservations.get(key)
        if existing is not None:
            return existing[1]
        reservation = transport.reserve_input(session_id, input_id)
        self._reservations[key] = (transport.connection_id, reservation)
        return reservation

    def _reserve_before_accept(
        self, session_id: str, input_id: str, transport: RequestTransport | None,
    ) -> bool:
        """在触发来源 watcher 前登记 reservation；返回是否由本次调用新建。"""
        if transport is None or (session_id, input_id) in self._reservations:
            return False
        self.reserve_input(session_id, input_id, transport)
        return True

    def _drop_reservation(
        self, session_id: str, input_id: str, transport: RequestTransport | None,
    ) -> None:
        """只回收本次接纳创建且仍由同一连接拥有的 reservation。"""
        if transport is None:
            return
        key = (session_id, input_id)
        owner = self._reservations.get(key)
        if owner is not None and owner[0] == transport.connection_id:
            del self._reservations[key]

    async def wait(self, reader: MessageReader, turn: Turn) -> None:
        """等待同连接完整最终 Output frame 的 writer flush。"""
        ending = turn.ending_message_id
        if ending is None:
            raise ValueError("programmatic delivery 缺少 Session 或最终 Output")
        input_id: str | None = None
        for identity in reversed(turn.message_ids):
            message = reader.get(identity)
            if message is not None and isinstance(message.body, Input):
                input_id = identity
                break
        reservation = self._reservations.get((reader.session_id, input_id or ""))
        if reservation is None:
            raise ValueError("程序最终 Output 没有同连接 Input reservation")
        await reservation[1].wait_output(ending)

    async def call(
        self, method: str, params: StrictModel,
        transport: RequestTransport | None = None,
    ) -> dict[str, object]:
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
            created = self._reserve_before_accept(session_id, send.message_id, transport)
            try:
                message = await source.accept(send.message_id, Input((
                    ContentPart("text", send.text),
                    ContentPart("channel.origin", {"channel": "programmatic", "chat_id": session_id[13:],
                                                    "sender": "control"}),
                )))
            except BaseException:
                if created:
                    self._drop_reservation(session_id, send.message_id, transport)
                raise
        elif method == "programmatic/message/pause":
            message = await source.pause(cast(PauseParams, params).message_id)
        elif method == "programmatic/message/resume":
            resume = cast(ResumeParams, params)
            created = self._reserve_before_accept(session_id, resume.input_id, transport)
            try:
                message = await source.resume(resume.message_id, resume.input_id)
            except BaseException:
                if created:
                    self._drop_reservation(session_id, resume.input_id, transport)
                raise
        else:
            raise AssertionError("未声明的程序调用方法: " + method)
        return {"version": 2, "session_id": message.session_id,
                "message_id": message.message_id, "seq": message.seq}


PROGRAMMATIC = ServiceKey[Programmatic]("programmatic.v1")
