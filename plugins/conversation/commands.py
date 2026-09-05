"""来源的命令组合：先固定 handler，结果只保存在 Message。"""
from __future__ import annotations

import json
from typing import Annotated, Literal, cast
from collections.abc import Awaitable, Callable, Mapping

from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.commands import COMMANDS, CommandExecution, CommandRegistry
from agent.plugin_composition.messages import MESSAGE_WRITERS, OWNER_STATE
from agent.plugin_composition.tasks import Task
from plugins.content.plugin import check_text
from session.log import MessageReader
from session.message import ContentPart, ContentReferences, Control, Input, Message, Output
from session.message_codec import json_value

from .program import check_source

Text = Annotated[str, Field(min_length=1)]


class CommandIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    session_id: Text
    source: Text
    input_id: Text
    binding_id: Text

    @property
    def output_id(self) -> str:
        return "command:" + self.binding_id + ":" + self.input_id


class CommandFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    input_id: Text
    binding_id: Text
    name: Text
    kind: Literal["success", "error"]


def check_result(part: ContentPart) -> ContentReferences:
    fact = CommandFact.model_validate(json_value(part.value))
    return ContentReferences(binding_ids=(fact.binding_id,))


def _input(reader: MessageReader, input_id: str, source: str) -> tuple[str, Mapping[str, str]]:
    """只使用已接纳 Input 中的来源事实，不能从 Session 名称反推渠道。"""
    message = reader.get(input_id)
    if message is None or message.source != source or not isinstance(message.body, Input):
        raise ValueError("命令引用缺少同来源 Input")
    texts = [cast(str, part.value) for part in message.body.parts if part.kind == "text"]
    origins = [part.value for part in message.body.parts if part.kind == "channel.origin"]
    if len(texts) != 1 or len(origins) != 1:
        raise ValueError("命令 Input 必须有唯一正文和已验证渠道身份")
    return texts[0], cast(Mapping[str, str], origins[0])


async def run_commands(ctx: Context, task: Task, reader: MessageReader, source: str) -> Message | None:
    """无命令时交给默认程序；未提交结果按固定 handler 的领域回执恢复。"""
    # 1. 先恢复已有 intent，再决定最新 Input；结果正文没有第二份 owner 副本。
    bindings = ctx.require(BINDINGS)
    state = ctx.require(OWNER_STATE).open(ctx)
    registry = ctx.require(COMMANDS).freeze()
    snapshot = reader.snapshot()
    inputs = [m for m in snapshot if m.source == source and isinstance(m.body, Input)]
    abandoned_through = max((m.body.through_seq for m in snapshot
                             if m.source == source and isinstance(m.body, Control)
                             and m.body.action == "abandon"), default=-1)
    input_positions = {message.message_id: message.seq for message in inputs}
    latest = inputs[-1]
    intents: list[tuple[CommandIntent, bool]] = []
    selected: CommandIntent | None = None
    for key, row in state.list():
        if not key.startswith("command:"):
            continue
        intent = CommandIntent.model_validate_json(json.dumps(json_value(row.value)))
        if row.version != 0:
            raise ValueError("命令 intent 不允许原位改写")
        if (intent.session_id, intent.source) != (reader.session_id, source):
            continue
        # 放弃停止后续命令调用；原 intent 与领域 receipt 保留，不能伪称效果已回滚。
        if input_positions[intent.input_id] <= abandoned_through:
            continue
        if intent.input_id == latest.message_id:
            selected = intent
        if reader.get(intent.output_id) is None:
            intents.append((intent, True))
    if selected is None:
        assert isinstance(latest.body, Input)
        texts = [cast(str, part.value) for part in latest.body.parts if part.kind == "text"]
        identity = registry.bind(bindings, texts[0]) if len(texts) == 1 else None
        if identity is not None:
            selected = CommandIntent(session_id=reader.session_id, source=source,
                                     input_id=latest.message_id, binding_id=identity)
            value = cast(Mapping[str, object], selected.model_dump())
            _ = _input(reader, latest.message_id, source)
            check_source(task, reader, source, reader.head(source=source))
            intent_key = selected.output_id
            _ = state.transact(lambda tx: tx.save(intent_key, value, expected_version=None))
            intents.append((selected, False))

    if not intents:
        return None if selected is None else reader.get(selected.output_id)
    positions = {message.message_id: message.seq for message in reader.snapshot()}
    intents.sort(key=lambda item: positions[item[0].input_id])

    # 2. handler 执行前后核对来源权限；中断后的副作用只能由领域回执确认。
    writer = ctx.require(MESSAGE_WRITERS).bind(
        ctx, author="app", source=source, body_types=(Output,),
        content={"text": check_text, "command.result": check_result},
    )(reader.session_id)
    task.on_close(writer.expire)
    state.check_access(reader, writer)
    try:
        for intent, recovering in intents:
            head = reader.head(source=source)
            check_source(task, reader, source, head)
            line, origin = _input(reader, intent.input_id, source)
            async def execute(commands: CommandRegistry) -> CommandExecution:
                value = await commands.execute(
                    line, session_key=reader.session_id, channel=origin["channel"],
                    chat_id=origin["chat_id"], sender=origin["sender"],
                    message_id=intent.input_id, recover=recovering,
                )
                if value is None or value.name != bindings.describe(intent.binding_id, COMMANDS)["name"]:
                    raise ValueError("固定的命令 handler 与原 Input 不匹配")
                return value
            if recovering:
                async with bindings.open(intent.binding_id, COMMANDS) as (archived, _metadata):
                    result = await execute(archived.freeze())
            else:
                result = await execute(registry)
            check_source(task, reader, source, head)
            fact = CommandFact(input_id=intent.input_id, binding_id=intent.binding_id,
                               name=result.name, kind=result.result.kind)
            parts = (ContentPart("command.result", fact.model_dump()),)
            if result.result.text:
                parts += (ContentPart("text", result.result.text),)
            finish = ("complete" if result.result.text else "quiet") if intent == selected else "continue"
            _ = writer.append(intent.output_id, Output(parts, finish), expected_source_head=head)
        return None if selected is None else reader.get(selected.output_id)
    finally:
        writer.expire()


CONVERSATION_COMMANDS = ServiceKey[Callable[[Task, MessageReader, str], Awaitable[Message | None]]]("conversation.commands.v1")
