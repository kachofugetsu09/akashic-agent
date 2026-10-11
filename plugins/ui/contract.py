"""UI 提供方公开合同；登记、资产和请求执行由 provider 实现。"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import aclosing
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from pathlib import Path
from types import ModuleType
from typing import Literal, Protocol, cast

from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.routing import WebSocketRoute

from agent.plugin_composition.context import Context
from agent.plugin_composition.requests import RequestContext
from agent.plugin_composition.effect import Effect
from agent.plugin_composition.model import ServiceKey

from plugins.ledger.contract import MessagePage, MessageReader, SessionEntry
from plugins.ledger.contract import ContentPart, Control, Input, Message, Output, ToolCall, ToolResult, json_value

DashboardRoute = APIRoute | WebSocketRoute

@dataclass(frozen=True)
class WebModuleAsset:
    module: str
    module_sha256: str
    module_bytes: int
    stylesheet: str
    stylesheet_sha256: str | None
    stylesheet_bytes: int
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    contract_digests: tuple[tuple[str, str], ...] = ()
    contract_sha256: str = ""

@dataclass(frozen=True)
class WebModuleDescriptor:
    plugin_id: str
    registration_uuid: str
    generation_id: str
    asset: WebModuleAsset

@dataclass(frozen=True)
class WebUiCatalog:
    identity: str
    modules: tuple[WebModuleDescriptor, ...]


@dataclass(frozen=True)
class DashboardBinding:
    plugin_id: str
    app: FastAPI
    routes: tuple[DashboardRoute, ...]
    context: Context
    generation_id: str
    has_web: bool
    runtime_workspace: Path
    runtime_data_root: Path
    module_name: str


class UiRegistry(Protocol):
    @property
    def root_instance_token(self) -> object: ...

    async def register(
        self, ctx: Context, *, web: str | None = None, dashboard: Callable[[], ModuleType] | None = None,
        requires: tuple[str, ...] = (), provides: tuple[str, ...] = (),
        contract_digests: Mapping[str, str] | None = None,
    ) -> Effect: ...

    def register_configuration(
        self, app: FastAPI, context: RequestContext,
        key: ServiceKey[Configuration], prefix: str,
    ) -> None: ...

    def catalog(self) -> WebUiCatalog: ...

    def bindings(self) -> tuple[DashboardBinding, ...]: ...

class WebUiProvider(Protocol):
    async def bootstrap(self) -> bytes: ...

    async def state(self) -> dict[str, str | bool]: ...


UI = ServiceKey[UiRegistry]("ui.v1")
WEB_UI = ServiceKey[WebUiProvider]("ui.web.v1")

@dataclass(frozen=True)
class PluginUiAsset:
    module: str
    module_sha256: str
    module_bytes: int
    stylesheet: str
    stylesheet_sha256: str | None
    stylesheet_bytes: int
    navigation_label: str | None
    navigation_description: str | None
    slots: tuple[str, ...]


PluginUiSlot = Literal[
    "turn.before_reasoning",
    "turn.before_tool",
    "turn.after_answer",
    "drawer.panel",
]

PLUGIN_UI_SLOTS = frozenset(
    {
        "turn.before_reasoning",
        "turn.before_tool",
        "turn.after_answer",
        "drawer.panel",
    }
)

class PluginUiQueryHandler(Protocol):
    def __call__(
        self,
        method: str,
        payload: dict[str, object],
        *,
        session_id: str | None,
        turn_id: str | None,
    ) -> object | Awaitable[object]:
        """Sync handlers run in bounded workers; async handlers keep the owner scope."""
        ...

class PluginUiRpcInvalidRequest(ValueError):
    """Signal a request rejected by the plugin-owned UI projection."""

class PluginUiPluginUnavailable(LookupError):
    """Signal that the requested Plugin UI owner is not available."""

class PluginUiStaleRevision(LookupError):
    """Signal that a Plugin UI request names an old registration revision."""

class PluginUiQueryTimeout(TimeoutError):
    """Signal that a Plugin UI query exceeded its caller-visible deadline."""

class PluginUiQueryOverloaded(RuntimeError):
    """Signal that the bounded Plugin UI query admission is full."""

class PluginUiRpcExecutionError(RuntimeError):
    """Signal that a Plugin UI handler failed while executing its RPC."""

@dataclass(frozen=True, slots=True)
class PluginUiNavigation:
    label: str
    description: str

@dataclass(frozen=True, slots=True)
class PluginUiDefinition:
    module: str
    stylesheet: str | None = None
    navigation: PluginUiNavigation | None = None
    slots: tuple[PluginUiSlot, ...] = ()

@dataclass(frozen=True, slots=True)
class PluginUiDescriptor:
    """Describe immutable plugin assets without retaining executable handlers."""

    owner: str
    module_sha256: str
    module_bytes: int
    stylesheet_sha256: str | None
    stylesheet_bytes: int
    navigation_label: str | None
    navigation_description: str | None
    slots: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class PluginUiBinding:
    """Bind one descriptor and its handlers to one exact contributor Context."""

    descriptor: PluginUiDescriptor
    asset: PluginUiAsset
    query: PluginUiQueryHandler
    available: Callable[[], bool]
    context: Context
    registration_uuid: str

class UiSlots(Protocol):
    """Expose the current plugin registrations owned by the UI provider."""

    @property
    def root_instance_token(self) -> object: ...

    async def register_plugin_ui(
        self, ctx: Context, definition: PluginUiDefinition, *,
        query: PluginUiQueryHandler, available: Callable[[], bool] | None = None,
    ) -> Effect: ...

    def bindings(self) -> tuple[PluginUiBinding, ...]: ...


UI_SLOTS = ServiceKey[UiSlots]("ui.slots.v1")


class Configuration(Protocol):
    async def read(self) -> dict[str, object]: ...
    async def save(self, request_id: str, expected_input: str, values: dict[str, object]) -> dict[str, object]: ...
    def receipt(self, request_id: str) -> dict[str, object]: ...


# 工具内容 owner 提供派生展示；读取回调只允许当前消息之前的同 Session 内容。
ToolResultDisplayProvider = Callable[
    [ContentPart, Callable[[str], Awaitable[Message | None]]], Awaitable[object]
]


class MessageDisplayReader(Protocol):
    """在自己的资源作用域内投影一页，不让客户端持有插件回调。"""

    async def __call__(
        self, page: MessagePage, *, display_only: bool
    ) -> list[dict[str, object]]: ...


class PluginUiProvider(Protocol):
    async def catalog(self) -> dict[str, object]: ...

    async def asset(
        self,
        plugin_id: str,
        plugin_revision: str,
        kind: str,
        sha256: str,
    ) -> dict[str, object]: ...

    async def query(
        self,
        plugin_id: str,
        plugin_revision: str,
        method: str,
        payload: dict[str, object],
        *,
        session_id: str | None,
        turn_id: str | None,
    ) -> dict[str, object]: ...


MESSAGE_DISPLAY = ServiceKey[MessageDisplayReader]("ui.messages.v1")
PLUGIN_UI = ServiceKey[PluginUiProvider]("ui.plugin.v1")


PartDisplayProvider = Callable[[ContentPart], Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class MessageDisplayProviders:
    """一次页面投影使用的只读回调；调用者负责同代 lease。"""

    tool_name: Callable[[str], str] | None = None
    part_display: Mapping[str, PartDisplayProvider] = field(default_factory=dict)
    result_values: Mapping[tuple[str, int], object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "part_display", MappingProxyType(dict(self.part_display)))
        object.__setattr__(self, "result_values", MappingProxyType(dict(self.result_values)))


async def read_message_rows(
    page: MessagePage, *, display_only: bool = False,
    reader: MessageDisplayReader | None = None,
) -> list[dict[str, object]]:
    """展示缺少业务 provider 时保留明确的 unavailable 状态。"""
    if reader is None:
        return message_rows(page, display_only=display_only)
    return await reader(page, display_only=display_only)


def session_row(entry: SessionEntry) -> dict[str, object]:
    """列表只读首条消息；无文字的会话也保留入口。"""
    first = entry.first_message
    text = "" if first is None or isinstance(first.body, Control) else "\n".join(
        cast(str, part.value) for part in first.body.parts
        if isinstance(part, ContentPart) and part.kind == "text"
    )
    return {
        "key": entry.session_id,
        "created_at": entry.created_at.isoformat(),
        "updated_at": entry.updated_at.isoformat(),
        "message_count": entry.message_count,
        "head_seq": entry.head_seq,
        "first_message_content": text,
        "scope": dict(entry.attributes.scope),
        "title": entry.title,
    }


def message_rows(
    page: MessagePage,
    *,
    display_only: bool = False,
    providers: MessageDisplayProviders | None = None,
) -> list[dict[str, object]]:
    """把固定消息页转为两端共用的展示数据，不读取或修改运行状态。"""
    display = MessageDisplayProviders() if providers is None else providers
    return [
        _message_row(message, page, display_only=display_only, providers=display)
        for message in page.messages
    ]


async def follow_messages(
    reader: MessageReader,
    *,
    after_seq: int,
    display_only: bool = False,
    reader_display: MessageDisplayReader | None = None,
) -> AsyncGenerator[dict[str, object], None]:
    """从 seq 续读完整展示页；唤醒通知不携带第二份消息正文。"""
    # 1. 先注册日志通知，再按页补齐附件和 binding 展示字段。
    async with aclosing(reader.follow(after_seq=after_seq)) as follower:
        async for message in follower:
            if message.seq <= after_seq:
                continue
            page = reader.read_page(after_seq=after_seq, limit=50)
            while page.messages:
                next_seq = page.messages[-1].seq
                yield {"version": 2, "session_id": reader.session_id,
                       "items": await read_message_rows(page, display_only=display_only, reader=reader_display), "after_seq": after_seq,
                       "through_seq": page.through_seq, "next_after_seq": next_seq,
                       "has_more": page.has_more}
                after_seq = next_seq
                if not page.has_more:
                    break
                # 2. 当前批次固定 head；随后到达的事实由外层 follow 继续追赶。
                page = reader.read_page(after_seq=after_seq, through_seq=page.through_seq, limit=50)


def _message_row(
    message: Message,
    page: MessagePage,
    *,
    display_only: bool,
    providers: MessageDisplayProviders,
) -> dict[str, object]:
    """保留真实类型、顺序和引用，页面不推断执行结果或重新分配作者。"""
    # 1. 身份和消息用途分别呈现；Control 与晚到结果仍是独立行。
    body = message.body

    row: dict[str, object] = {
        "id": message.message_id,
        "session_id": message.session_id,
        "seq": message.seq,
        "timestamp": message.recorded_at.isoformat(),
        "author": message.author,
        "source": message.source,
        "metadata": json_value(message.metadata),
        "attachments": [asdict(ref) for ref in page.attachments[message.message_id]],
    }
    if isinstance(body, Control):
        row["body"] = {"kind": "control", "action": body.action,
                       "through_seq": body.through_seq, "reason": body.reason}
        return row
    parts = [
        {"kind": part.kind, "display": "data", "value": json_value(part.value),
         "rendered": json_value(providers.result_values[message.message_id, index])}
        if isinstance(part, ContentPart) and (message.message_id, index) in providers.result_values
        else _part(part, display_only=display_only, providers=providers, tool_result=isinstance(body, ToolResult))
        if isinstance(part, ContentPart)
        else _tool_call(part, providers=providers)
        for index, part in enumerate(body.parts)
    ]
    if isinstance(body, Input):
        row["body"] = {"kind": "input", "parts": parts}
    elif isinstance(body, Output):
        row["body"] = {"kind": "output", "parts": parts, "finish": body.finish}
    else:
        row["body"] = {"kind": "tool_result", "parts": parts,
                       "call_ref": asdict(body.call_ref), "outcome": body.outcome}
    return row


def _tool_call(part: ToolCall, *, providers: MessageDisplayProviders) -> dict[str, object]:
    """只通过工具 owner 的名称回调展示 binding，不解析或重开工具。"""
    name_reader = providers.tool_name
    if name_reader is None:
        return {
            "kind": "tool_call",
            "binding_id": part.binding_id,
            "display": "unavailable",
        }
    return {
        "kind": "tool_call",
        "binding_id": part.binding_id,
        "name": name_reader(part.binding_id),
        "arguments": json_value(part.arguments),
    }


def _part(
    part: ContentPart,
    *,
    display_only: bool,
    providers: MessageDisplayProviders,
    tool_result: bool,
) -> dict[str, object]:
    """工具结果保留通用数据；其他内部内容只公开 owner 允许的字段。"""
    # 1. 业务字段由插件 callback 选择；Core 不识别模型或工具的字段名。
    provider = providers.part_display.get(part.kind)
    if provider is not None:
        value = json_value(provider(part))
        return {"kind": part.kind, "value": value, **({"display": "data"} if tool_result else {})}
    # 2. 展示端保留原 part 下标；不可展示的归档只传类型，权威正文不变。
    if display_only and part.kind in {"history.provenance", "history.record", "history.turn_input"}:
        return {"kind": part.kind, "display": "unavailable"}
    # 旧客户端和旧下载摘要仍使用原表示；可见 transcript 始终完整。
    if part.kind in {"history.provenance", "history.transcript", "history.record", "history.turn_input"}:
        return {"kind": part.kind, "archive": json_value(part.value)}
    if part.kind in {"text", "artifact_ref", "reply_ref"}:
        return {"kind": part.kind, "value": json_value(part.value)}
    # 工具输出有通用数据展示；Input/Output 的内部执行材料仍须 owner 明确公开。
    if tool_result:
        return {"kind": part.kind, "display": "data", "value": json_value(part.value)}
    return {"kind": part.kind, "display": "unavailable"}

# message_display.py 按内容 kind 借用 message.display:* 与 message.result_display:*。
# Context、Models 与 ContentView 分别拥有自己的 renderer；没有中央 key 列表或默认补位。
