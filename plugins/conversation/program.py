from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any
from collections.abc import Mapping

from agent.plugin_composition import Context
from agent.model_runtime.session_selection import read_session_model_selection
from agent.plugin_composition.artifacts import ARTIFACT_READ
from agent.plugin_composition.messages import MESSAGE_WRITERS
from agent.plugin_composition.models import BoundChatModel, ChatModels, ChatModelSelection, ModelRequest, ModelRole
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.tasks import Task
from agent.plugin_contracts.context import ContextModel, Materials, Reminder, Summary, check_summary, summary_range
from plugins.models.selection import selection
from plugins.models.content import load_artifacts, render_content as render_model_content
from plugins.models.projection import CallReader, ContentRenderer, MessageProjection, check_facts, check_tool_rejection
from agent.plugin_contracts.tool_api import Authorize, result_message_id
from plugins.tools.api import MessageReply
from plugins.tools.menu import ToolMenu, ToolPresentation
from agent.plugin_contracts.tools import ToolView
from plugins.standard_tools.shell import shell_cleanup
from session.log import MessageReader
from agent.plugin_contracts import CallRef, ContentPart, Input, Message, Output, ToolResult

if TYPE_CHECKING:
    from agent.plugin_contracts.content import ContentView as Content
    from plugins.context.plugin import ContextBuilder
    from plugins.context.materials import ContextMaterials
    from agent.plugin_contracts.tools import ToolCatalogPort as ToolCatalog
    from agent.plugin_contracts.turn_projection import TurnProjectionPort as TurnProjection
    from plugins.react.plugin import Preview


def check_source(task: Task, reader: MessageReader, source: str, through_seq: int) -> None:
    """新输入或控制已接纳时禁止新效果，不依赖后台取消信号及时送达。"""
    from agent.plugin_contracts import Control, Input

    if not task.active or any(
        message.source == source and isinstance(message.body, (Input, Control))
        for message in reader.snapshot(after_seq=through_seq)
    ):
        raise asyncio.CancelledError


async def run_reply(
    ctx: Context, task: Task, reader: MessageReader, source: str, *,
    models: ChatModels, content: Content, context: ContextBuilder, tools: ToolCatalog,
    react: Callable[..., Awaitable[Message]],
    materials: ContextMaterials,
    turn_projection: TurnProjection,
    render_content: ContentRenderer | None = None,
    read_call: CallReader,
    authorize: Authorize,
    max_output_tokens: int,
    max_steps: int,
    tool_view: ToolView | None = None,
    tool_names: Sequence[str] | None = None,
    exclude_materials: frozenset[str] = frozenset(),
    prompt_hints: Sequence[str] = (),
    fixed_bindings: Mapping[str, str] | None = None,
    preview: Preview | None = None,
    reminders: Sequence[Reminder] = (),
    terminal_tools: frozenset[str] = frozenset(),
    presentation: ToolPresentation | None = None,
) -> Message:
    """普通组合拥有本次程序资源，Source 不必同步签发模型或内容 writer。"""
    if tool_names is not None:
        if tool_view is not None or fixed_bindings is None:
            raise ValueError("旧工具名称只可核对原固定 binding")
        if len(set(tool_names)) != len(tool_names) or set(tool_names) != set(
            fixed_bindings
        ):
            raise ValueError("旧工具名称与原固定 binding 不一致")
    # 1. 内容检查器与模型绑定覆盖整个程序，取消时先排空已开始的工具。
    prompt_hints = tuple(prompt_hints)
    reader = reader.incremental()
    source_head = reader.head(source=source)
    snapshot = reader.snapshot()
    turns = turn_projection.project(snapshot, source)
    open_ids: set[str] = set(turns[-1].message_ids) if turns and turns[-1].status == "open" else set()
    chosen = selection(tuple(message for message in snapshot if message.message_id in open_ids))
    if chosen is None:
        metadata = reader.metadata()
        saved = read_session_model_selection(metadata if metadata is not None else {})
        chosen = ChatModelSelection(saved.model_ref or None, saved.reasoning_effort or None)
    from_seq = min((message.seq for message in snapshot if message.message_id in open_ids), default=source_head + 1)
    async with (
        shell_cleanup(ctx, reader, source, from_seq, task=task, drain=tools.drain_calls),
        content.bind() as view,
        models.execution(model_id=chosen.model_id, reasoning_effort=chosen.reasoning_effort) as execution,
        materials.bind(exclude=exclude_materials) as material_view,
    ):
        bindings = ctx.require(BINDINGS)
        model = execution.chat(ModelRole.AGENT)
        writers = ctx.require(MESSAGE_WRITERS)
        keep_input_ids = tuple(
            item.message_id for item in snapshot
            if item.message_id in open_ids and isinstance(item.body, Input)
        )
        def reply(ref: CallRef) -> MessageReply:
            return MessageReply(
                result_message_id(ref), ref, reader,
                writers.bind(
                    ctx, author="tool", source=source, body_types=(ToolResult,),
                    content=view.checks,
                )(reader.session_id, call_ref=ref),
                lambda: check_source(task, reader, source, source_head),
            )

        menu = ToolMenu(tools, bindings, tools.execution(
            authorize, child_permit=task.child_permit if task.has_external_permit else None,
        ), reply,
                        view=tool_view, limit=model.max_tool_schemas,
                        fixed_bindings=fixed_bindings, presentation=presentation)
        if terminal_tools - menu.names:
            raise ValueError("终结工具必须属于本次允许目录")
        output = writers.bind(
            ctx, author="assistant", source=source, body_types=(Output,),
            check_metadata=view.check_metadata,
            content={**view.checks, "model.facts": check_facts, "model.tool_rejection": check_tool_rejection, "context.summary": check_summary}, check_call=menu.check_call,
        )(reader.session_id)
        task.on_close(output.expire)
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
            result = await material_view.prepare(
                messages, source, caller=ctx, reminders=tuple(reminders),
            )
            if render_content is None:
                start = 0 if result.summary is None else summary_range(messages, result.summary.source_message_ids).stop
                refs = reader.attachments_for(tuple(
                    message.message_id for index, message in enumerate(messages)
                    if index >= start or message.message_id in keep_input_ids
                ))
                if refs:
                    artifacts = await load_artifacts(
                        ctx.require(ARTIFACT_READ), refs,
                        accepts_images="image" in model.descriptor.capabilities.input_modalities,
                    )
            check_source(task, reader, source, source_head)
            return replace(result, system_prompt="\n\n".join(
                part for part in (result.system_prompt, *view.prompts, *prompt_hints, menu.system_prompt) if part
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
                reduce=reduce, preview=preview, terminal_tools=terminal_tools,
            )
        finally:
            output.expire()
