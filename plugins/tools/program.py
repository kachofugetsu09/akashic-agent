"""工具 owner 为一次回复组装菜单和消息回执。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol, cast

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_WRITERS
from agent.plugin_contracts import CallRef, ContentPart, ContentReferences, ToolResult
from agent.restart import ExternalRootPermit
from session.log import MessageReader, MessageWriter

from .api import Authorize, MessageReply, result_message_id
from .execution import ToolExecution

if TYPE_CHECKING:
    from .menu import ToolMenu, ToolPresentation
    from .plugin import ToolCatalog, ToolView


ContentCheck = Callable[[ContentPart], ContentReferences]


class _Catalog(Protocol):
    """菜单工厂需要的 tools owner 能力。"""

    def execution(
        self,
        authorize: Authorize,
        *,
        child_permit: Callable[[], ExternalRootPermit] | None = None,
    ) -> ToolExecution: ...

class ToolProgramFactory:
    """返回真实 ToolMenu，并由 tools owner 构造每次调用的 MessageReply。"""

    def __init__(self, ctx: Context, catalog: _Catalog):
        self._ctx = ctx
        self._catalog = catalog

    def bind_reply(
        self,
        reader: MessageReader,
        source: str,
        *,
        content: Mapping[str, ContentCheck],
        check_start: Callable[[], None],
    ) -> Callable[[CallRef], MessageReply]:
        """固定来源和 scoped writer，返回原 owner 的实际 MessageReply。"""
        if not isinstance(source, str) or not source:
            raise ValueError("工具回执来源不能为空")
        if not callable(check_start):
            raise TypeError("工具回执必须有启动检查器")
        writers = self._ctx.require(MESSAGE_WRITERS)
        writer: Callable[..., MessageWriter] = writers.bind(
            self._ctx,
            author="tool",
            source=source,
            body_types=(ToolResult,),
            content=content,
        )

        def reply(call_ref: CallRef) -> MessageReply:
            return MessageReply(
                result_message_id(call_ref),
                call_ref,
                reader,
                writer(reader.session_id, call_ref=call_ref),
                check_start,
            )

        return reply

    def create_menu(
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
        from .menu import ToolMenu

        bindings = self._ctx.require(BINDINGS)
        reply = self.bind_reply(
            reader,
            source,
            content=content,
            check_start=check_start,
        )
        return ToolMenu(
            cast("ToolCatalog", self._catalog),
            bindings,
            self._catalog.execution(authorize, child_permit=child_permit),
            reply,
            view=view,
            limit=limit,
            fixed_bindings=fixed_bindings,
            presentation=presentation,
        )


TOOL_PROGRAM = ServiceKey[ToolProgramFactory]("tools.program.v1")
