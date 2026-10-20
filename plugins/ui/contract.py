"""UI 提供方公开合同；登记、资产和请求执行由 provider 实现。"""

from __future__ import annotations

import asyncio
import gzip
import os
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import aclosing
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from pathlib import Path
from types import ModuleType
from typing import Literal, Protocol, cast

from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import FileResponse, Response
from starlette.routing import WebSocketRoute
from starlette.staticfiles import NotModifiedResponse, StaticFiles
from starlette.types import Receive, Scope, Send

from core.common.file_io import run_file_io

from agent.plugin_composition.context import Context
from agent.plugin_composition.requests import RequestContext
from agent.plugin_composition.effect import Effect
from agent.plugin_composition.model import ServiceKey

from plugins.ledger.contract import MessagePage, MessageReader, SessionEntry
from plugins.ledger.contract import ContentPart, Control, Input, Message, Output, ToolCall, ToolResult, json_value
from plugins.tools.contract import ToolDisplay

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

    async def module_asset(self, plugin_id: str, sha256: str) -> str:
        """返回当前目录中该插件的 web module 源码；插件缺失或摘要不匹配时抛 KeyError。"""
        ...


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
    slots: tuple[str, ...]


PluginUiSlot = Literal[
    "turn.before_reasoning",
    "turn.after_answer",
    "drawer.panel",
    "tool.result",
    "message.attachment",
    "chat.notice",
    "header.end",
    "composer.toolbar",
    "composer.status",
    "composer.suggest",
    "scope.fields",
    "scope.badge",
    "navigation.groups",
]

PLUGIN_UI_SLOTS = frozenset(
    {
        "turn.before_reasoning",
        "turn.after_answer",
        "drawer.panel",
        "tool.result",
        "message.attachment",
        "chat.notice",
        "header.end",
        "composer.toolbar",
        "composer.status",
        "composer.suggest",
        "scope.fields",
        "scope.badge",
        "navigation.groups",
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

class PluginUiFollowHandler(Protocol):
    def __call__(self, session_id: str) -> AsyncGenerator[Mapping[str, object], None]:
        """Yield this plugin's latest session state; ending means the state is no longer valid."""
        ...


# 插件推给自己界面的一帧：value 为 None 表示该插件的会话状态已失效（插件卸载或代际结束）。
PluginUiEmit = Callable[[dict[str, object]], Awaitable[None]]


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
class PluginUiDefinition:
    module: str
    stylesheet: str | None = None
    slots: tuple[PluginUiSlot, ...] = ()

@dataclass(frozen=True, slots=True)
class PluginUiDescriptor:
    """Describe immutable plugin assets without retaining executable handlers."""

    owner: str
    module_sha256: str
    module_bytes: int
    stylesheet_sha256: str | None
    stylesheet_bytes: int
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
    follow: PluginUiFollowHandler | None = None

class UiSlots(Protocol):
    """Expose the current plugin registrations owned by the UI provider."""

    @property
    def root_instance_token(self) -> object: ...

    async def register_plugin_ui(
        self, ctx: Context, definition: PluginUiDefinition, *,
        query: PluginUiQueryHandler, available: Callable[[], bool] | None = None,
        follow: PluginUiFollowHandler | None = None,
    ) -> Effect: ...

    def bindings(self) -> tuple[PluginUiBinding, ...]: ...

    def changed(self) -> asyncio.Event:
        """Set once when the registrations change; read it again for the next change."""
        ...


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

    async def follow(self, session_id: str, emit: PluginUiEmit) -> None:
        """Forward every plugin's session stream until cancelled or the UI owner closes."""
        ...


MESSAGE_DISPLAY = ServiceKey[MessageDisplayReader]("ui.messages.v1")
PLUGIN_UI = ServiceKey[PluginUiProvider]("ui.plugin.v1")


# owner 把某段内容声明为思考过程：宿主按通用的思考行展示 text，value 是 owner 公开给其他客户端的字段。
@dataclass(frozen=True, slots=True)
class ThinkingDisplay:
    text: str | None
    value: Mapping[str, object]


PartDisplayProvider = Callable[[ContentPart], Mapping[str, object] | ThinkingDisplay]


# 内容 owner 以 message.display:<kind> 提供展示回调；没有 owner 回调的内容显示为“无法展示”。
def part_display_key(kind: str) -> ServiceKey[PartDisplayProvider]:
    return ServiceKey[PartDisplayProvider](f"message.display:{kind}")


# owner 把自己的内部材料声明为不进聊天正文：`ctx.provide(part_display_key(kind), hidden_part)`。
def hidden_part(part: ContentPart) -> Mapping[str, object]:
    return {}


@dataclass(frozen=True, slots=True)
class MessageDisplayProviders:
    """一次页面投影使用的只读回调；调用者负责同代 lease。"""

    tool_display: Callable[[str], ToolDisplay] | None = None
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
    """只通过 tools 的展示回调读取名称与 owner，不解析或重开工具；owner 决定谁能渲染结果。"""
    reader = providers.tool_display
    if reader is None:
        return {
            "kind": "tool_call",
            "binding_id": part.binding_id,
            "display": "unavailable",
        }
    display = reader(part.binding_id)
    return {
        "kind": "tool_call",
        "binding_id": part.binding_id,
        "name": display.name,
        **({"owner": display.owner} if display.owner is not None else {}),
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
    if provider is hidden_part:
        return {"kind": part.kind, "display": "hidden"}
    if provider is not None:
        shown = provider(part)
        if isinstance(shown, ThinkingDisplay):
            return {"kind": part.kind, "display": "thinking", "text": shown.text, "value": json_value(shown.value)}
        value = json_value(shown)
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

# ---- 压缩静态资产 ----
#
# 浏览器到插件监听器的带宽是冷启动瀑布的主体（入口 JS/CSS 合计约 1.5MB 未压缩），
# StaticFiles 不支持内容编码。这里按 (路径, mtime, 大小) 缓存 gzip 变体：哈希产物
# 内容不变可永久命中，稳定入口随构建换 key 自然失效。首次压缩排水到文件 worker，
# 不占事件循环；范围请求与不接受 gzip 的客户端走父类原路径，304 语义不变。
_COMPRESSIBLE_SUFFIXES = (".css", ".html", ".js", ".json", ".map", ".svg", ".txt", ".wasm")
_GZIP_MIN_BYTES = 8 * 1024
_STATIC_GZIP_CACHE_LIMIT = 64
_static_gzip_cache: dict[str, bytes] = {}


def accepts_gzip_header(value: str) -> bool:
    """Accept-Encoding 明确接受 gzip 时压缩；无效权重按不接受处理。"""
    for token in value.split(","):
        parts = token.split(";")
        if parts[0].strip().lower() != "gzip":
            continue
        for parameter in parts[1:]:
            name, _, parameter_value = parameter.partition("=")
            if name.strip().lower() == "q":
                try:
                    quality = float(parameter_value.strip())
                except ValueError:
                    return False
                if not 0 < quality <= 1:
                    return False
        return True
    return False


def _accepts_gzip(scope: Scope) -> bool:
    return accepts_gzip_header(Headers(scope=scope).get("accept-encoding", ""))


class CompressedFileResponse(FileResponse):
    """发送已选定的 gzip 变体；协商和条件校验由静态路由完成。"""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.headers.get("content-encoding") != "gzip":
            await super().__call__(scope, receive, send)
            return
        stat_result = self.stat_result
        assert stat_result is not None
        cache_key = f"{self.path}:{stat_result.st_mtime_ns}:{stat_result.st_size}"
        body = _static_gzip_cache.get(cache_key)
        if body is None:
            body = await run_file_io(lambda: gzip.compress(Path(self.path).read_bytes(), compresslevel=9, mtime=0))
            _static_gzip_cache[cache_key] = body
            while len(_static_gzip_cache) > _STATIC_GZIP_CACHE_LIMIT:
                del _static_gzip_cache[next(iter(_static_gzip_cache))]
        headers = MutableHeaders(raw=list(self.raw_headers))
        headers["content-length"] = str(len(body))
        await send({"type": "http.response.start", "status": self.status_code, "headers": headers.raw})
        payload = b"" if scope["method"].upper() == "HEAD" else body
        await send({"type": "http.response.body", "body": payload, "more_body": False})
        if self.background is not None:
            await self.background()


class CompressedStaticFiles(StaticFiles):
    """StaticFiles 外加按文件身份缓存的 gzip 变体；缓存头与 304 行为与父类一致。"""

    def file_response(
        self,
        full_path: os.PathLike[str] | str,
        stat_result: os.stat_result,
        scope: Scope,
        status_code: int = 200,
    ) -> Response:
        response = CompressedFileResponse(full_path, status_code=status_code, stat_result=stat_result)
        # 条件校验先选表示：不同编码不能共用强 ETag，否则 gzip 的 If-Range
        # 会错误命中 identity 的字节范围。所有变体（含 304）均声明 Vary。
        if stat_result.st_size >= _GZIP_MIN_BYTES and str(full_path).endswith(_COMPRESSIBLE_SUFFIXES):
            response.headers.add_vary_header("Accept-Encoding")
            if "range" not in Headers(scope=scope) and _accepts_gzip(scope):
                response.headers["content-encoding"] = "gzip"
                response.headers["etag"] = response.headers["etag"][:-1] + '-gzip"'
        if self.is_not_modified(response.headers, Headers(scope=scope)):
            return NotModifiedResponse(response.headers)
        return response
# Context、Models 与 ContentView 分别拥有自己的 renderer；没有中央 key 列表或默认补位。
