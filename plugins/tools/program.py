"""工具 owner 为一次回复组装菜单和消息回执。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING

from agent.plugin_composition import Context
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import (
    MESSAGE_WRITERS,
    MessageReader,
    MessageWriter,
)
from agent.plugin_composition.tasks import ExternalRootPermit
from agent.plugin_contracts import CallRef, ContentPart, ContentReferences, ToolResult
from agent.plugin_contracts.tools import (
    TOOL_PROGRAM as TOOL_PROGRAM,
)

from .api import Authorize, MessageReply, result_message_id

if TYPE_CHECKING:
    from .menu import ToolMenu, ToolPresentation
    from .plugin import ToolCatalog, ToolView


ContentCheck = Callable[[ContentPart], ContentReferences]


class ToolProgramFactory:
    """返回真实 ToolMenu，并由 tools owner 构造每次调用的 MessageReply。"""

    def __init__(self, ctx: Context, catalog: ToolCatalog):
        self._ctx = ctx
        self._catalog = catalog

    def bind_reply(
        self,
        reader: MessageReader,
        source: str,
        *,
        content: Mapping[str, ContentCheck],
        check_start: Callable[[], None],
    ) -> Callable[[CallRef], Awaitable[MessageReply]]:
        """Open each tool writer under the Tools owner's actual call scope."""

        async def reply(call_ref: CallRef) -> MessageReply:
            async with self._ctx.runtime_scope():
                writers = self._ctx.require(MESSAGE_WRITERS)
                writer: Callable[..., MessageWriter] = writers.bind(
                    self._ctx,
                    author="tool",
                    source=source,
                    body_types=(ToolResult,),
                    content=content,
                )
                opened = writer(reader.session_id, call_ref=call_ref)
            return MessageReply(
                result_message_id(call_ref), call_ref, reader, opened, check_start,
            )

        return reply

    async def create_menu(
        self,
        reader: MessageReader,
        source: str,
        *,
        content: Mapping[str, ContentCheck],
        check_start: Callable[[], None],
        authorize: Authorize,
        view: ToolView | None = None,
        fixed_bindings: Mapping[str, str] | None = None,
        limit: int | None = None,
        presentation: ToolPresentation | None = None,
        child_permit: Callable[[], ExternalRootPermit] | None = None,
    ) -> ToolMenu:
        """按真实 reader、writer、授权和 view 组装一个可执行菜单。"""
        from .menu import NativePresentation, ToolMenu

        async with self._ctx.runtime_scope():
            bindings = self._ctx.require(BINDINGS)
            execution = self._catalog.execution(authorize, child_permit=child_permit)
            if view is not None:
                selected = presentation or NativePresentation({
                    ref.name: ref.description for ref in view.refs
                })
                fixed_bindings = {
                    ref.name: await self._catalog.bind_scoped(
                        ref, bindings, configuration=selected.configuration(ref.name),
                    )
                    for ref in view.refs
                }
                presentation = selected
                view = None
            if fixed_bindings is None:
                raise ValueError("工具菜单缺少 current view 或固定 binding")
            return ToolMenu(
                bindings,
                execution,
                self.bind_reply(reader, source, content=content, check_start=check_start),
                limit=limit,
                fixed_bindings=fixed_bindings,
                presentation=presentation,
            )
