from collections.abc import Awaitable, Callable, Mapping
from typing import cast

from agent.plugin_composition import Context
from agent.plugin_composition.artifacts import ARTIFACT_READ
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.channels import ChannelInboundMessage
from agent.plugin_composition.commands import COMMANDS
from agent.plugin_composition.messages import (
    MESSAGE_CATALOG,
    MESSAGE_WRITERS,
    OWNER_STATE,
    SESSION_ADMISSION,
    MessageConflict,
    MessageReader,
    SessionAttributes,
)
from agent.plugin_composition.models import MODEL_CATALOG, ChatModelSelection
from agent.plugin_composition.tasks import (
    RESTART_GATE,
    TASKS,
    Task,
)
from agent.plugin_contracts import (
    Body,
    ContentPart,
    ContentReferences,
    Control,
    Input,
    Message,
    Output,
)
from agent.plugin_contracts.models import (
    MODEL_SELECTION as MODEL_SELECTION,
    ModelSelection as ModelSelection,
)
from agent.plugin_contracts.sources import (
    CHECK_ORIGIN as CHECK_ORIGIN,
    CONVERSATION_COMPLETE as CONVERSATION_COMPLETE,
    SOURCE_CHANGED,
    SOURCE_SESSION as SOURCE_SESSION,
    SOURCES as SOURCES,
    ConversationComplete as ConversationComplete,
    SessionFactory as SessionFactory,
    SourceChanged,
    SourceSession as SourceSession,
)

from .commands import CONTENT, CONVERSATION_COMMANDS, SOURCE_CHECK, run_commands
from .source import update_selection

api_version = 3
name = "conversation"
version = "1.0.0"
desc = "接纳和控制同一来源的消息，程序由调用者另行选择"
inject = (
    BINDINGS, OWNER_STATE, TASKS, ARTIFACT_READ, COMMANDS, CONTENT, SOURCE_CHECK,
    MESSAGE_WRITERS, SESSION_ADMISSION, SOURCES, SOURCE_SESSION, RESTART_GATE,
    MODEL_SELECTION, MESSAGE_CATALOG,
)


async def apply(ctx: Context) -> None:
    """来源只使用模型选择校验与持久化能力，不持有模型执行权。"""
    model_selection = ctx.require(MODEL_SELECTION)

    def update_metadata(body: Body) -> Mapping[str, object | None]:
        return update_selection(body, write_saved=model_selection.write_saved)

    _ = await ctx.require(MESSAGE_WRITERS).register_metadata(
        ctx, keys=frozenset({"model_selection", "model_runtime_override"}), update=update_metadata,
    )
    def changed(reader: MessageReader, source: str) -> None:
        ctx.emit(SOURCE_CHANGED, SourceChanged(reader, source))

    def open(session_id: str) -> SourceSession:
        def check_model(part: ContentPart) -> ContentReferences:
            references = ctx.require(MODEL_SELECTION).check(part)
            value = cast(Mapping[str, str | None], part.value)
            with ctx.borrow(MODEL_CATALOG) as catalog:
                if catalog is None:
                    raise ValueError("当前组合不提供模型选择目录")
                _ = catalog.validate_chat_selection(
                    ChatModelSelection(value["model_id"], value["reasoning_effort"]),
                )
            return references

        def check_reply_target(part: ContentPart) -> ContentReferences:
            references = check_reply(part)
            target = ctx.require(MESSAGE_CATALOG).reader(session_id).get(cast(str, part.value))
            if target is None or not isinstance(target.body, (Input, Output)):
                raise MessageConflict("引用目标不是当前 Session 的 Input 或 Output")
            return references

        writers = ctx.require(MESSAGE_WRITERS)
        inputs = writers.bind(
            ctx, author="user", source="conversation", body_types=(Input,),
            content={"text": ctx.require(CONTENT).check_text, "artifact_ref": ctx.require(CONTENT).check_artifact, "channel.origin": check_origin,
                     "reply_ref": check_reply_target, "model.selection": check_model},
            update_metadata=update_metadata,
        )
        controls = writers.bind(
            ctx, author="app", source="conversation", body_types=(Control,), content={},
        )
        return ctx.require(SOURCE_SESSION)(
            reader=ctx.require(MESSAGE_CATALOG).reader(session_id),
            inputs=inputs(session_id), controls=controls(session_id),
            tasks=ctx.require(TASKS).open(ctx), changed=changed,
            restart_gate=ctx.require(RESTART_GATE),
        )

    async def accept(session_id: str, message_id: str, message: ChannelInboundMessage) -> Message:
        command = message.content.strip().split(maxsplit=1)
        if command and command[0].split("@", 1)[0].lower() == "/stop":
            return await open(session_id).pause(message_id)
        retry = message.metadata.get("retry_of_client_message_id")
        if retry is not None:
            if not isinstance(retry, str) or not retry:
                raise ValueError("重试必须引用已有 Input")
            return await open(session_id).resume(message_id, retry)
        # 1. 带宽键的首条输入先接纳 Session；之后同一 scope 重复声明是幂等核对。
        dimensions = session_dimensions(message.metadata)
        if dimensions:
            _ = ctx.require(SESSION_ADMISSION).ensure(ctx, session_id, SessionAttributes.scoped(dimensions))
        # 2. 来源只核验已发布附件，不接管传输 lease 或派生另一份元数据。
        if message.attachments:
            artifacts = ctx.require(ARTIFACT_READ)
            for ref in message.attachments:
                lease = await artifacts.acquire(ref)
                await lease.aclose()
        parts = (
            ContentPart("channel.origin", {
                "channel": message.channel, "chat_id": message.chat_id,
                "sender": message.sender,
            }),
            ContentPart("text", message.content),
            *(ContentPart("artifact_ref", ref.artifact_id) for ref in message.attachments),
        )
        reply = message.metadata.get("reply_to_message_id")
        if reply is not None:
            parts += (ContentPart("reply_ref", reply),)
        if "model_runtime_id" in message.metadata:
            model_id = message.metadata["model_runtime_id"]
            effort = message.metadata.get("model_reasoning_effort", "")
            if not isinstance(model_id, str) or not isinstance(effort, str):
                raise TypeError("模型选择必须是字符串")
            parts += (ContentPart("model.selection", {
                "model_id": model_id.strip() or None, "reasoning_effort": effort.strip() or None,
            }),)
        # 3. Input 与全部引用原子提交；传输时间、handoff 和重复 ID 不进入正文。
        return await open(session_id).accept(message_id, Input(parts))

    @ctx.entrypoint
    async def command(task: Task, reader: MessageReader, source: str) -> Message | None:
        return await run_commands(ctx, task, reader, source)

    @ctx.entrypoint
    async def complete(
        session_id: str, program: Callable[[Task, MessageReader], Awaitable[Message]],
    ) -> Message:
        return await open(session_id).complete(program)

    _ = await ctx.provide(CHECK_ORIGIN, check_origin)
    _ = await ctx.provide(CONVERSATION_COMMANDS, command)
    _ = await ctx.provide(CONVERSATION_COMPLETE, complete)
    _ = await ctx.require(SOURCES).register(ctx, name="conversation", open=open, accept=accept, channels=None,
        needs_reply=lambda reader: ctx.require(SOURCE_SESSION).needs_reply(reader, "conversation"))


def check_origin(part: ContentPart) -> ContentReferences:
    """来源保存原始传输事实；metadata 不获得 source、角色或路由覆盖权。"""
    raw_value = part.value
    if not isinstance(raw_value, Mapping):
        raise ValueError("channel.origin 必须是对象")
    value = cast(Mapping[str, object], raw_value)
    if set(value) != {"channel", "chat_id", "sender"} or any(
        not isinstance(item, str) or not item for item in value.values()
    ):
        raise ValueError("channel.origin 身份无效")
    return ContentReferences()


def session_dimensions(metadata: Mapping[str, object]) -> dict[str, str]:
    """传输只携带维度声明；取值合法性由各维度 owner 在接纳时校验。"""
    raw = metadata.get("session_dimensions")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("session_dimensions 必须是对象")
    dimensions = cast(Mapping[object, object], raw)
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in dimensions.items()):
        raise ValueError("session_dimensions 只能包含字符串")
    return {cast(str, key): cast(str, value) for key, value in dimensions.items()}


def check_reply(part: ContentPart) -> ContentReferences:
    if not isinstance(part.value, str) or not part.value:
        raise ValueError("reply_ref 必须是目标 message_id")
    return ContentReferences()
