from __future__ import annotations

from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition import Context, ServiceKey
from plugins.gateway.contract import (
    CONTROL_FRAMES,
    FrameResolver,
    FrameRouteStage,
)
from agent.plugin_composition.messages import (
    MESSAGE_CATALOG,
    SESSION_ADMISSION,
    MessageReader,
    SessionAttributes,
)
from plugins.gateway.contract import RequestTransport, RpcMethod
from agent.plugin_contracts import ContentPart, Input
from plugins.delivery.contract import (
    FinalOutputTurn as FinalOutputTurn,
)

from .result import TURN_PROJECTION, TurnProjection, read_result, read_result_snapshot


class SessionIdParams(BaseModel):
    """程序调用在自己的 RPC 输入边界校验参数。"""
    model_config = ConfigDict(extra="forbid", strict=True)
    session_id: str = Field(min_length=1, max_length=512)


class AdmitParams(SessionIdParams):
    persist_memory: bool = False


class SendParams(SessionIdParams):
    message_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=1_048_576)
    model_id: str | None = Field(default=None, min_length=1, max_length=512)
    reasoning_effort: str | None = Field(default=None, min_length=1, max_length=64)


class PauseParams(SessionIdParams):
    message_id: str = Field(min_length=1, max_length=256)


class ResumeParams(PauseParams):
    input_id: str = Field(min_length=1, max_length=256)


class ResultParams(SessionIdParams):
    input_id: str = Field(min_length=1, max_length=256)


PARAMS: dict[str, type[BaseModel]] = {
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
        self.call = ctx.entrypoint(self.call)
        self._frames = ctx.require(CONTROL_FRAMES)
        # Session -> (已检查到的来源 head, 当时的 active input 集合)
        self._settled: dict[str, tuple[int, frozenset[str]]] = {}

    def _resolver(self, session_id: str, input_id: str) -> FrameResolver:
        """Capture only the reader, pure projection and Input identity."""
        reader = self.ctx.require(MESSAGE_CATALOG).reader(session_id)
        projection = self.ctx.require(TURN_PROJECTION)
        return lambda: resolve_completed_output(reader, projection, input_id)

    async def settle_changed(self, reader: MessageReader, source: str) -> None:
        """随来源终态回收无 claim route，避免长连接积累已结束输入。"""
        if source != "programmatic":
            return
        input_ids = self._frames.active_input_ids(reader.session_id)
        if not input_ids:
            _ = self._settled.pop(reader.session_id, None)
            return
        if not self._may_end(reader, source, frozenset(input_ids)):
            return
        projection = self.ctx.require(TURN_PROJECTION)
        # 1. 纯只读快照只解释已经提交的 Input，不消费连接状态。
        def read(reader: MessageReader) -> tuple[int, tuple[tuple[str, str], ...]]:
            messages = reader.snapshot()
            turns = projection.project(messages, source)
            # 连接先登记 reservation，再提交 Input；这段等待不代表消息损坏。
            committed_inputs = {message.message_id for message in messages
                                if message.source == source and isinstance(message.body, Input)}
            ended: list[tuple[str, str]] = []
            for input_id in input_ids:
                if input_id not in committed_inputs:
                    continue
                result = read_result_snapshot(reader, input_id, projection, messages, turns)
                status = result["status"]
                if not isinstance(status, str):
                    raise TypeError("programmatic result status 必须是字符串")
                if status != "open":
                    ended.append((input_id, status))
            return reader.head(source=source), tuple(ended)
        # 2. 读取期间可能恢复同一 Input；旧前缀不能结束后来建立的回传通道。
        head, ended = await reader.read_async(read)
        if reader.head(source=source) != head:
            return
        self._settled[reader.session_id] = (head, frozenset(input_ids))
        for input_id, status in ended:
            error = None if status == "complete" else RuntimeError(f"programmatic input 已结束: {status}")
            self._frames.settle_input(reader.session_id, input_id, error)

    # Turn 只会因终态 Output 或 Control 结束；水位之后没有这两类消息时无需重读整个 Session。
    def _may_end(self, reader: MessageReader, source: str, input_ids: frozenset[str]) -> bool:
        head = reader.head(source=source)
        previous = self._settled.get(reader.session_id)
        if previous is None or previous[1] != input_ids:
            return True
        checked = previous[0]
        if head <= checked:
            return False
        if reader.latest_finished_output_seq(source, after_seq=checked, through_seq=head) is not None:
            return True
        control = reader.latest_control(source, through_seq=head)
        if control is not None and control.seq > checked:
            return True
        self._settled[reader.session_id] = (head, input_ids)
        return False

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
        def read(snapshot: MessageReader) -> str:
            for identity in reversed(turn.message_ids):
                message = snapshot.get(identity)
                if message is not None and isinstance(message.body, Input):
                    return identity
            raise ValueError("程序最终 Output 没有同连接 Input reservation")
        input_id = await reader.read_async(read)
        try:
            await self._frames.wait_input(reader.session_id, input_id, ending)
        except LookupError as error:
            raise ConnectionError("程序调用的最终 Output 连接已结束") from error

    async def call(
        self, method: str, params: BaseModel,
        transport: RequestTransport | None = None,
    ) -> dict[str, object]:
        """为每次公开调用借本 Programmatic Fiber 的短作用域。"""
        session_id = cast(SessionIdParams, params).session_id
        check_session(session_id)
        return await self._call(method, params, transport, session_id)

    async def _call(
        self, method: str, params: BaseModel,
        transport: RequestTransport | None, session_id: str,
    ) -> dict[str, object]:
        """在同一次 owner 许可内完成接纳、消息与结果操作。"""
        from .plugin import open_source

        ctx = self.ctx
        # 1. 创建时提交不可变资格；ACK 丢失可用调用方原身份幂等重试。
        if method == "programmatic/session/admit":
            create = cast(AdmitParams, params)
            attributes = await ctx.require(SESSION_ADMISSION).ensure_async(ctx, session_id, SessionAttributes(
                visibility="internal", learning="eligible" if create.persist_memory else "excluded",
            ))
            return {"version": 2, "session_id": session_id, "visibility": attributes.visibility,
                    "learning": attributes.learning}

        # 2. 来源只读取已创建属性，后续输入无改变学习资格的字段。
        if method == "programmatic/message/result":
            reader = ctx.require(MESSAGE_CATALOG).reader(session_id)
            if reader.attributes.visibility != "internal":
                raise ValueError("程序调用 Session 尚未通过内部来源准入")
            input_id = cast(ResultParams, params).input_id
            projection = ctx.require(TURN_PROJECTION)
            return await reader.read_async(lambda snapshot: read_result(snapshot, input_id, projection))
        source = open_source(ctx, session_id)
        if method == "programmatic/message/send":
            send = cast(SendParams, params)
            if not send.text.strip():
                raise ValueError("程序输入不能为空白")
            parts = (
                ContentPart("text", send.text),
                ContentPart("channel.origin", {"channel": "programmatic", "chat_id": session_id[13:],
                                                "sender": "control"}),
            )
            if send.model_fields_set & {"model_id", "reasoning_effort"}:
                parts += (ContentPart("model.selection", {
                    "model_id": send.model_id, "reasoning_effort": send.reasoning_effort,
                }),)
            created = self._reserve_before_accept(session_id, send.message_id, transport)
            try:
                message = await source.accept(send.message_id, Input(parts))
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
    def build(name: str, params: type[BaseModel]) -> RpcMethod:
        async def call(value: BaseModel) -> object:
            return await programmatic.call(name, value)

        async def call_with_transport(value: BaseModel, transport: RequestTransport) -> object:
            return await programmatic.call(name, value, transport)

        return RpcMethod(params, call, call_with_transport)

    return {name: build(name, params) for name, params in PARAMS.items()}
