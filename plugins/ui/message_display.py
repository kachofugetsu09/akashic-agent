"""UI owner 在本次实际回调的作用域内投影消息。"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import ExitStack

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.messages import MESSAGE_CATALOG, MessageReader
from plugins.ui.contract import MessageDisplayProviders, PartDisplayProvider, message_rows
from plugins.tools.contract import TOOL_DISPLAY_NAME
from plugins.ui.contract import ToolResultDisplayProvider, MessagePage
from agent.plugin_contracts import ContentPart, Control, Message, ToolCall, ToolResult


async def project_message_rows(
    ctx: Context, page: MessagePage, *, display_only: bool,
) -> list[dict[str, object]]:
    """只借用页面实际需要的展示服务，并保护它们直到投影完成。"""
    # 1. 固定页面用途；展示不会选择新消息或重跑工具。
    kinds: list[str] = []
    has_tool_call = False
    for message in page.messages:
        if isinstance(message.body, Control):
            continue
        for part in message.body.parts:
            if isinstance(part, ContentPart) and part.kind not in kinds:
                kinds.append(part.kind)
            elif isinstance(part, ToolCall):
                has_tool_call = True
    providers: dict[str, PartDisplayProvider] = {}
    result_providers: dict[str, ToolResultDisplayProvider] = {}

    # 2. 借用从唯一 provider 目录取得真实 scope，不持有或查询 Root。
    with ExitStack() as scopes:
        for kind in kinds:
            display = scopes.enter_context(ctx.borrow(ServiceKey[PartDisplayProvider](f"message.display:{kind}")))
            if display is not None:
                providers[kind] = display
            result_display = scopes.enter_context(ctx.borrow(ServiceKey[ToolResultDisplayProvider](f"message.result_display:{kind}")))
            if result_display is not None:
                result_providers[kind] = result_display
        tool_name = scopes.enter_context(ctx.borrow(TOOL_DISPLAY_NAME)) if has_tool_call else None
        result_values: dict[tuple[str, int], object] = {}
        if result_providers:
            catalog = scopes.enter_context(ctx.borrow(MESSAGE_CATALOG))
            if catalog is None:
                raise RuntimeError("消息展示缺少只读目录")
            for message in page.messages:
                if not isinstance(message.body, ToolResult):
                    continue
                read_message = _read_before(catalog.reader(message.session_id), message)
                for index, part in enumerate(message.body.parts):
                    provider = result_providers.get(part.kind)
                    if provider is not None:
                        result_values[message.message_id, index] = await provider(part, read_message)
        return message_rows(page, display_only=display_only, providers=MessageDisplayProviders(
            tool_name=tool_name, part_display=providers, result_values=result_values,
        ))


def _read_before(reader: MessageReader, message: Message) -> Callable[[str], Awaitable[Message | None]]:
    """把只读范围固定在同一 Session 的当前消息之前。"""
    async def read_message(message_id: str) -> Message | None:
        target = await reader.read_async(lambda current: current.get(message_id))
        if target is None or target.session_id != message.session_id or target.seq >= message.seq:
            return None
        return target

    return read_message
