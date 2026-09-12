from __future__ import annotations


from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from typing import Any, Protocol
from collections.abc import Mapping

from agent.plugin_composition import Context, ServiceKey

from agent.plugin_composition.channels import AttachmentRef, ChannelAttachmentReadPort

from agent.plugin_composition.models import BoundChatModel, ChatModelSelection, ModelRequest, StreamCallback
from agent.plugin_composition.tasks import Task, ExternalRootPermit
from agent.plugin_composition.messages import MessageReader
from agent.plugin_contracts import CallRef, ContentPart, ContentReferences, Message

Materials = Mapping[str, object]
Summary = Mapping[str, object]
Reminder = Mapping[str, object]
ToolView = object
ToolPresentation = object
Authorize = Callable[[str, Mapping[str, object]], Awaitable[Mapping[str, object] | str]]
Preview = Callable[[str], AbstractContextManager[StreamCallback]]


class ContextModel(Protocol):
    @property
    def context_window(self) -> int | None: ...
    @property
    def max_tool_schemas(self) -> int | None: ...
    def estimate(self, request: ModelRequest) -> int: ...
    def render(self, messages: tuple[Message, ...], *, after_seq: int,
               summary_reference: str | None = None, fresh: bool = False) -> ModelRequest: ...


class ContentView(Protocol):
    @property
    def checks(self) -> Mapping[str, Callable[[ContentPart], ContentReferences]]: ...
    @property
    def prompts(self) -> tuple[str, ...]: ...
    def check_metadata(self, metadata: Mapping[str, object]) -> None: ...


class Content(Protocol):
    def bind(self) -> AbstractAsyncContextManager[ContentView]: ...


class ContextBuilder(Protocol):
    def check_summary(self, part: ContentPart) -> ContentReferences: ...
    def summary_range(self, snapshot: tuple[Message, ...], source_message_ids: tuple[str, ...]) -> range: ...


class MaterialView(Protocol):
    async def prepare(self, snapshot: tuple[Message, ...], source: str, *,
                      caller: Context | None = None, reminders: tuple[Reminder, ...] = ()) -> Materials: ...
    async def reduce(self, snapshot: tuple[Message, ...], materials: Materials,
                     request: ModelRequest, model: BoundChatModel, projection: ContextModel,
                     *, source: str, force: bool) -> Summary | None: ...


class ContextMaterials(Protocol):
    def bind(self, *, exclude: frozenset[str] = frozenset()) -> AbstractAsyncContextManager[MaterialView]: ...


class ProjectedTurn(Protocol):
    @property
    def message_ids(self) -> tuple[str, ...]: ...
    @property
    def status(self) -> str: ...


class TurnProjection(Protocol):
    def project(self, messages: tuple[Message, ...], source: str) -> tuple[ProjectedTurn, ...]: ...


class ToolCatalog(Protocol):
    async def drain_calls(self, calls: tuple[CallRef, ...]) -> None: ...


class ToolMenu(Protocol):
    @property
    def names(self) -> frozenset[str]: ...
    @property
    def system_prompt(self) -> str: ...
    def check_call(self, call: Any) -> None: ...
    def name(self, binding_id: str) -> str: ...


class ToolProgram(Protocol):
    def create_menu(self, reader: MessageReader, source: str, *,
                    content: Mapping[str, Callable[[ContentPart], ContentReferences]],
                    check_start: Callable[[], None], authorize: Authorize,
                    view: object | None = None, fixed_bindings: Mapping[str, str] | None = None,
                    limit: int | None = None, presentation: object | None = None,
                    child_permit: Callable[[], ExternalRootPermit] | None = None) -> ToolMenu: ...


ContentRenderer = Callable[[ContentPart], Sequence[Mapping[str, Any]]]
CallReader = Callable[[str], Mapping[str, Any]]


class ModelSelection(Protocol):
    def read_saved(self, metadata: Mapping[str, object]) -> ChatModelSelection: ...
    def read(self, messages: Sequence[Message]) -> ChatModelSelection | None: ...


class ModelContent(Protocol):
    def render(
        self, part: ContentPart, *, artifacts: Mapping[str, tuple[Mapping[str, Any], ...]],
        read_message: Callable[[str], Message | None] | None = None,
    ) -> tuple[Mapping[str, Any], ...]: ...

    async def load_artifacts(
        self, reader: ChannelAttachmentReadPort, refs: Sequence[AttachmentRef], *, accepts_images: bool,
    ) -> Mapping[str, tuple[Mapping[str, Any], ...]]: ...


class ModelChecks(Protocol):
    def check_facts(self, part: ContentPart) -> ContentReferences: ...
    def check_tool_rejection(self, part: ContentPart) -> ContentReferences: ...


class ModelProjections(Protocol):
    def create(
        self, model: BoundChatModel, *, source: str, render_content: ContentRenderer,
        tool_name: Callable[[str], str], read_call: CallReader,
        check_summary: Callable[[ContentPart], ContentReferences], keep_input_ids: tuple[str, ...] = (),
    ) -> ContextModel: ...


MODEL_SELECTION = ServiceKey[ModelSelection]("models.selection.v1")
MODEL_CONTENT = ServiceKey[ModelContent]("models.content.v1")
MODEL_CHECKS = ServiceKey[ModelChecks]("models.message-checks.v1")
MODEL_PROJECTION = ServiceKey[ModelProjections]("models.projection.v1")


class ToolCleanup(Protocol):
    """程序消费者提供本次工具 owner 的真实收尾边界。"""

    def __call__(
        self,
        ctx: Context,
        reader: MessageReader,
        source: str,
        from_seq: int,
        *,
        task: Task,
        drain: Callable[[tuple[CallRef, ...]], Awaitable[None]],
    ) -> AbstractAsyncContextManager[None]: ...


CONTENT = ServiceKey[Content]("content.v2")
CONTEXT = ServiceKey[ContextBuilder]("context.v2")
MATERIALS = ServiceKey[ContextMaterials]("context.materials.v3")
TOOLS = ServiceKey[ToolCatalog]("tools.v1")
TOOL_PROGRAM = ServiceKey[ToolProgram]("tools.program.v1")
TOOL_CLEANUP = ServiceKey[ToolCleanup]("tools.cleanup.v1")
TURN_PROJECTION = ServiceKey[TurnProjection]("turn.projection.v1")
REACT = ServiceKey[Callable[..., Awaitable[Message]]]("react.v2")
MODEL_CALLS = ServiceKey[CallReader]("models.calls.v1")

SOURCE_CHECK = ServiceKey[Callable[[Task, MessageReader, str, int], None]]("source.check.v1")
