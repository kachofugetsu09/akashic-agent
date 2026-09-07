from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any
from collections.abc import Mapping

from agent.plugin_composition import Context
from agent.plugin_composition.artifacts import ARTIFACT_READ
from agent.plugin_composition.messages import MESSAGE_WRITERS
from agent.plugin_composition.models import BoundChatModel, ChatModels, ModelRequest, ModelRole
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.tasks import Task
from plugins.context.api import ContextModel, Materials, Summary, check_summary, summary_range
from plugins.models.selection import selection
from plugins.models.content import load_artifacts, render_content as render_model_content
from plugins.models.projection import CallReader, ContentRenderer, MessageProjection, check_facts
from plugins.tools.api import Authorize, MessageReply
from plugins.tools.menu import ToolMenu
from session.log import MessageReader
from session.message import CallRef, ContentPart, Input, Message, Output, ToolResult

if TYPE_CHECKING:
    from plugins.content.plugin import Content
    from plugins.context.plugin import ContextBuilder
    from plugins.context.materials import ContextMaterials
    from plugins.tools.plugin import ToolCatalog
    from plugins.turn_projection.plugin import TurnProjection


def check_source(task: Task, reader: MessageReader, source: str, through_seq: int) -> None:
    """新输入或控制已接纳时禁止新效果，不依赖后台取消信号及时送达。"""
    from session.message import Control, Input

    if not task.active or any(
        message.source == source and isinstance(message.body, (Input, Control))
        for message in reader.snapshot()
        if message.seq > through_seq
    ):
        raise asyncio.CancelledError


async def run_reply(
    ctx: Context, task: Task, reader: MessageReader, source: str, *,
    models: ChatModels, content: Content, context: ContextBuilder, tools: ToolCatalog,
    react: Callable[..., Awaitable[Message]],
    materials: ContextMaterials,
    turn_projection: TurnProjection,
    render_content: ContentRenderer | None = None, read_call: CallReader, authorize: Authorize,
    tool_names: Sequence[str], max_output_tokens: int, max_steps: int,
) -> Message:
    """普通组合拥有本次程序资源，Source 不必同步签发模型或内容 writer。"""
    # 1. 内容检查器与模型绑定覆盖整个程序，取消时先排空已开始的工具。
    chosen = selection(reader.snapshot())
    async with (
        content.bind() as view,
        models.execution(model_id=chosen.model_id, reasoning_effort=chosen.reasoning_effort) as execution,
        materials.bind() as material_view,
    ):
        bindings = ctx.require(BINDINGS)
        writers = ctx.require(MESSAGE_WRITERS)
        source_head = reader.head(source=source)
        snapshot = reader.snapshot()
        turns = turn_projection.project(snapshot, source)
        open_ids: set[str] = set(turns[-1].message_ids) if turns and turns[-1].status == "open" else set()
        keep_input_ids = tuple(
            item.message_id for item in snapshot
            if item.message_id in open_ids and isinstance(item.body, Input)
        )
        def reply(ref: CallRef) -> MessageReply:
            return MessageReply(
                f"tool-result:{ref.message_id}:{ref.part_index}", ref, reader,
                writers.bind(
                    ctx, author="tool", source=source, body_types=(ToolResult,),
                    content={**view.checks, "tool.selection": lambda part: menu.check_selection(ref, part)},
                )(reader.session_id, call_ref=ref),
                lambda: check_source(task, reader, source, source_head),
            )

        menu = ToolMenu(tools, bindings, tools.execution(authorize), reply,
                        names=tool_names, reader=reader, source=source)
        output = writers.bind(
            ctx, author="assistant", source=source, body_types=(Output,),
            content={**view.checks, "model.facts": check_facts, "context.summary": check_summary}, check_call=menu.check_call,
        )(reader.session_id)
        task.on_close(output.expire)
        model = execution.chat(ModelRole.AGENT)
        artifacts: Mapping[str, tuple[Mapping[str, Any], ...]] = {}
        def render(part: ContentPart):
            return render_model_content(part, artifacts=artifacts, read_message=reader.get)
        projection = MessageProjection(
            model, source=source, render_content=render if render_content is None else render_content,
            tool_name=menu.name, read_call=read_call, keep_input_ids=keep_input_ids,
        )

        # 2. 内容协议提示与解码来自同一 view；Context 仍只接收已取得的材料。
        async def build_materials(messages: tuple[Message, ...]) -> Materials:
            nonlocal artifacts
            result = await material_view.prepare(messages, source)
            if render_content is None:
                start = 0 if result.summary is None else summary_range(messages, result.summary.source_message_ids).stop
                refs = tuple(
                    ref for index, message in enumerate(messages)
                    if index >= start or message.message_id in keep_input_ids
                    for ref in reader.attachments(message.message_id)
                )
                if refs:
                    artifacts = await load_artifacts(
                        ctx.require(ARTIFACT_READ), refs,
                        accepts_images="image" in model.descriptor.capabilities.input_modalities,
                    )
            check_source(task, reader, source, source_head)
            return replace(result, system_prompt="\n\n".join(
                part for part in (result.system_prompt, *view.prompts) if part
            ))

        async def reduce(
            snapshot: tuple[Message, ...], prepared: Materials, request: ModelRequest,
            model: BoundChatModel, projection: ContextModel, *, source: str, force: bool,
        ) -> Summary | None:
            result = await material_view.reduce(snapshot, prepared, request, model, projection,
                                                source=source, force=force)
            check_source(task, reader, source, source_head)
            return result

        try:
            return await react(
                reader, output, model=model, context=context, projection=projection,
                materials=build_materials, content=view, tools=menu,
                max_output_tokens=max_output_tokens, max_steps=max_steps,
                reduce=reduce,
            )
        finally:
            output.expire()
