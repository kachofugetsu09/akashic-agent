from collections.abc import Callable, Mapping
from typing import cast

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.artifacts import ARTIFACT_READ
from agent.plugin_composition.channels import CHANNEL_INPUT, ChannelInboundMessage
from agent.plugin_composition.messages import MESSAGE_CATALOG, MESSAGE_WRITERS
from agent.plugin_composition.tasks import TASKS, Task
from session.log import MessageReader
from plugins.content.plugin import check_text
from plugins.content.api import check_artifact
from plugins.models.selection import check_selection
from session.message import ContentPart, ContentReferences, Control, Input, Message, Output

from .source import Conversation
from .commands import CONVERSATION_COMMANDS, run_commands

api_version = 3
name = "conversation"
version = "1.0.0"
desc = "接纳和控制同一来源的消息，程序由调用者另行选择"
inject = ()

CONVERSATION = ServiceKey[Callable[[str], Conversation]]("conversation.v1")


async def apply(ctx: Context, config: object) -> None:
    """来源能力不依赖模型或自动回复，正式调用时才取得宿主读写权。"""
    def open(session_id: str) -> Conversation:
        writers = ctx.require(MESSAGE_WRITERS)
        inputs = writers.bind(
            ctx, author="user", source="conversation", body_types=(Input,),
            content={"text": check_text, "artifact_ref": check_artifact, "channel.origin": check_origin,
                     "reply_ref": check_reply, "model.selection": check_selection},
        )
        controls = writers.bind(
            ctx, author="app", source="conversation", body_types=(Control,), content={},
        )
        return Conversation(
            reader=ctx.require(MESSAGE_CATALOG).reader(session_id),
            inputs=inputs(session_id), controls=controls(session_id),
            tasks=ctx.require(TASKS).open(ctx),
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
        # 1. 来源只核验已发布附件，不接管传输 lease 或派生另一份元数据。
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
        reader = ctx.require(MESSAGE_CATALOG).reader(session_id)
        reply = message.metadata.get("reply_to_message_id")
        if reply is not None:
            part = ContentPart("reply_ref", reply)
            _ = check_reply(part)
            target = reader.get(cast(str, reply))
            if target is None or not isinstance(target.body, (Input, Output)):
                raise ValueError("引用目标不是当前 Session 的可展示消息")
            parts += (part,)
        if "model_runtime_id" in message.metadata:
            model_id = message.metadata["model_runtime_id"]
            effort = message.metadata.get("model_reasoning_effort", "")
            if not isinstance(model_id, str) or not isinstance(effort, str):
                raise TypeError("模型选择必须是字符串")
            parts += (ContentPart("model.selection", {
                "model_id": model_id.strip() or None, "reasoning_effort": effort.strip() or None,
            }),)
        # 2. Input 与全部引用原子提交；传输时间、handoff 和重复 ID 不进入正文。
        return await open(session_id).accept(message_id, Input(parts))

    async def command(task: Task, reader: MessageReader, source: str) -> Message | None:
        return await run_commands(ctx, task, reader, source)

    _ = await ctx.provide(CONVERSATION_COMMANDS, command)
    _ = await ctx.provide(CONVERSATION, open)
    _ = await ctx.provide(CHANNEL_INPUT, accept)


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


def check_reply(part: ContentPart) -> ContentReferences:
    if not isinstance(part.value, str) or not part.value:
        raise ValueError("reply_ref 必须是目标 message_id")
    return ContentReferences()
