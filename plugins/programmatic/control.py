from __future__ import annotations

from typing import Protocol, cast

from pydantic import Field

from agent.control.protocol.models import StrictModel, SessionIdParams
from agent.plugin_composition import Context, ServiceKey
from agent.control.frame_book import CONTROL_FRAMES, FrameRouteStage, FrameResolver
from agent.plugin_composition.rpc import RequestTransport, RpcMethod
from agent.plugin_composition.messages import MESSAGE_CATALOG, SESSION_ADMISSION
from plugins.turn_projection.plugin import TURN_PROJECTION, TurnProjection
from agent.plugin_composition.messages import MessageReader, SessionAttributes
from agent.plugin_contracts import ContentPart, Input

from .result import read_result, read_result_snapshot


class FinalOutputTurn(Protocol):
    @property
    def ending_message_id(self) -> str | None: ...
    @property
    def message_ids(self) -> tuple[str, ...]: ...


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


def resolve_completed_output(
    reader: MessageReader, projection: TurnProjection, input_id: str,
) -> str | None:
    """Resolve one completed Output from a read-only Session prefix."""
    messages = reader.snapshot()
    target = next((message for message in messages if message.message_id == input_id), None)
    if target is None or target.source != "programmatic" or not isinstance(target.body, Input):
        return None
    turn = next(
        (turn for turn in projection.project(messages, target.source) if input_id in turn.message_ids),
        None,
    )
    if turn is None or turn.status != "complete":
        return None
    return turn.ending_message_id


class Programmatic:
    """程序来源拥有固定身份和创建属性；读取与回复各用既有能力。"""

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self._frames = ctx.require(CONTROL_FRAMES)

    def _resolver(self, session_id: str, input_id: str) -> FrameResolver:
        """Capture only the reader, pure projection and Input identity."""
        reader = self.ctx.require(MESSAGE_CATALOG).reader(session_id)
        projection = self.ctx.require(TURN_PROJECTION)
        return lambda: resolve_completed_output(reader, projection, input_id)

    def settle_changed(self, reader: MessageReader, source: str) -> None:
        """随来源终态回收无 claim route，避免长连接积累已结束输入。"""
        if source != "programmatic":
            return
        input_ids = self._frames.active_input_ids(reader.session_id)
        if not input_ids:
            return
        messages = reader.snapshot()
        projection = self.ctx.require(TURN_PROJECTION)
        turns = projection.project(messages, source)
        for input_id in input_ids:
            result = read_result_snapshot(reader, input_id, projection, messages, turns)
            if result["status"] == "open":
                continue
            status = result["status"]
            if not isinstance(status, str):
                raise TypeError("programmatic result status 必须是字符串")
            error = None if status == "complete" else RuntimeError(
                f"programmatic input 已结束: {status}",
            )
            self._frames.settle_input(reader.session_id, input_id, error)

    def _reserve_before_accept(
        self, session_id: str, input_id: str, transport: RequestTransport | None,
    ) -> bool:
        """在触发来源 watcher 前登记当前请求连接的 reservation。"""
        if transport is None:
            return False
        resolver = self._resolver(session_id, input_id)
        _reservation, created = self._frames.route_input_with_owner(
            session_id, input_id, transport.connection_id, resolver,
        )
        return created

    def _stage_before_resume(
        self, session_id: str, input_id: str, transport: RequestTransport | None,
    ) -> FrameRouteStage | None:
        if transport is None:
            return None
        return self._frames.stage_input(
            session_id, input_id, transport.connection_id,
            self._resolver(session_id, input_id),
        )

    async def wait(self, reader: MessageReader, turn: FinalOutputTurn) -> None:
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
        if input_id is None:
            raise ValueError("程序最终 Output 没有同连接 Input reservation")
        await self._frames.wait_input(reader.session_id, input_id, ending)

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
            result = read_result(reader,
                cast(ResultParams, params).input_id, ctx.require(TURN_PROJECTION))
            if result["status"] != "open":
                input_id = cast(ResultParams, params).input_id
                self._frames.release_input(session_id, input_id)
            return result
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
                    self._frames.release_input(session_id, send.message_id)
                raise
        elif method == "programmatic/message/pause":
            message = await source.pause(cast(PauseParams, params).message_id)
        elif method == "programmatic/message/resume":
            resume = cast(ResumeParams, params)
            stage = self._stage_before_resume(session_id, resume.input_id, transport)
            try:
                message = await source.resume(resume.message_id, resume.input_id)
            except BaseException:
                if stage is not None:
                    stage.abort()
                raise
            if stage is not None:
                _ = stage.commit()
        else:
            raise AssertionError("未声明的程序调用方法: " + method)
        return {"version": 2, "session_id": message.session_id,
                "message_id": message.message_id, "seq": message.seq}


PROGRAMMATIC = ServiceKey[Programmatic]("programmatic.v1")


def rpc_methods(programmatic: Programmatic) -> dict[str, RpcMethod]:
    """来源自己声明协议参数；Core 不持有程序调用方法目录。"""
    def build(name: str, params: type[StrictModel]) -> RpcMethod:
        async def call(value: StrictModel) -> object:
            return await programmatic.call(name, value)

        async def call_with_transport(value: StrictModel, transport: RequestTransport) -> object:
            return await programmatic.call(name, value, transport)

        return RpcMethod(params, call, call_with_transport)

    return {name: build(name, params) for name, params in PARAMS.items()}
