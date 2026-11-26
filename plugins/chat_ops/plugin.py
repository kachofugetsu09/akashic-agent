from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from pathlib import PurePosixPath

from agent.plugin_composition import Context
from plugins.akashic_clients.contract import CHAT_SESSIONS, ChatRequestError, ChatSessions, ChatSessionsHost
from plugins.host_execution.contract import FILES, FileByteReader, FileError, Files
from plugins.ledger.contract import ContentPart
from plugins.operations.contract import (
    session_of,
    OPERATIONS_V1, OperationRequest, OperationResult, OperationSpec, OptionSpec,
)

api_version = 3
name = "chat_ops"
version = "1.0.0"
desc = "把 Akashic 聊天目录（会话、消息、置顶、附件）的界面入口发布为 Self-CLI 命令"

_ERROR_CODES = {400: "rejected", 404: "not_found", 413: "too_large", 415: "rejected", 422: "invalid_arguments", 503: "unavailable"}
_TEXT_LIMIT = 64 * 1024
_SESSION_KEY = OptionSpec("session_key", "会话 ID（侧栏目录里的 key，如 akashic:...）", required=True)
_PAGE_SIZE = OptionSpec("page_size", "每页条数 1–200", value_type="int", default=50)

Call = Callable[[ChatSessions], Awaitable[Mapping[str, object]]]


def _text(req: OperationRequest, key: str) -> str | None:
    value = req.args.get(key)
    return value if isinstance(value, str) else None


def _int(req: OperationRequest, key: str) -> int | None:
    value = req.args.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _int_or(req: OperationRequest, key: str, default: int) -> int:
    value = _int(req, key)
    return default if value is None else value


def _session(req: OperationRequest) -> str:
    return _text(req, "session_key") or ""


# 统一把领域拒绝与通道换代映射为 operation 错误；成功时原样返回界面同款载荷。
async def _run(host: ChatSessionsHost, summary: str, call: Call) -> OperationResult:
    try:
        data = await call(host.current())
    except ChatRequestError as error:
        return OperationResult(False, error.detail, error_code=_ERROR_CODES.get(error.status, "rejected"))
    except RuntimeError as error:
        return OperationResult(False, f"聊天通道暂不可用: {error}", error_code="unavailable")
    return OperationResult(True, summary, data=data)


def _session_read_specs(host: ChatSessionsHost) -> tuple[OperationSpec, ...]:
    async def list_sessions(req: OperationRequest) -> OperationResult:
        return await _run(host, "会话目录", lambda chat: chat.list_sessions(
            page_size=_int_or(req, "page_size", 50), after_time=_text(req, "after_time"), after_key=_text(req, "after_key")))

    # 只读消息与查找先按侧栏可见性（本通道前缀 + listed）核对；助手只能读自己所在的会话（跨会话读回复用 session send --wait）。
    async def messages(req: OperationRequest) -> OperationResult:
        target = session_of(req, "session_key")

        async def call(chat: ChatSessions) -> Mapping[str, object]:
            await chat.ensure_listed(target)
            return await chat.list_messages(
                target, page_size=_int_or(req, "page_size", 50), before_seq=_int(req, "before_seq"),
                through_seq=_int(req, "through_seq"), after_seq=_int(req, "after_seq"))
        return await _run(host, "会话消息页", call)

    async def search(req: OperationRequest) -> OperationResult:
        target = session_of(req, "session_key")

        async def call(chat: ChatSessions) -> Mapping[str, object]:
            await chat.ensure_listed(target)
            return await chat.search_messages(target, q=_text(req, "q") or "", limit=_int_or(req, "limit", 200))
        return await _run(host, "会话内查找结果", call)

    return (
        OperationSpec(
            operation_id="chat_ops.session.list", path=("session", "list"),
            summary="列出侧栏会话目录（本通道 listed 会话，按更新时间倒序，游标分页）", handler=list_sessions,
            options=(_PAGE_SIZE, OptionSpec("after_time", "游标：上一页 next_cursor.updated_at"),
                     OptionSpec("after_key", "游标：上一页 next_cursor.session_id")),
            mirrors=("http:akashic_clients:GET /api/chat/sessions",),
        ),
        OperationSpec(
            operation_id="chat_ops.session.messages", path=("session", "messages"),
            summary="读取会话消息页（默认从尾部往前翻；after_seq 往新的方向读）", handler=messages,
            options=(_SESSION_KEY, _PAGE_SIZE,
                     OptionSpec("before_seq", "读取此 seq 之前的较早消息", value_type="int"),
                     OptionSpec("through_seq", "固定读取上界 seq（-1 表示空）", value_type="int"),
                     OptionSpec("after_seq", "读取此 seq 之后的较新消息；不能与 before_seq 同用", value_type="int")),
            mirrors=("http:akashic_clients:GET /api/chat/sessions/{session_key:path}/messages",),
        ),
        OperationSpec(
            operation_id="chat_ops.session.search", path=("session", "search"),
            summary="在会话完整历史里查找正文，返回命中位置与摘要", handler=search,
            options=(_SESSION_KEY, OptionSpec("q", "查找文本（≤200 字；空串只预热索引）", default=""),
                     OptionSpec("limit", "最多返回命中数 1–500", value_type="int", default=200)),
            mirrors=("http:akashic_clients:GET /api/chat/sessions/{session_key:path}/search",),
        ),
    )


def _session_admin_specs(host: ChatSessionsHost) -> tuple[OperationSpec, ...]:
    async def rename(req: OperationRequest) -> OperationResult:
        return await _run(host, "会话标题已更新", lambda chat: chat.rename(_session(req), _text(req, "title") or ""))

    async def delete(req: OperationRequest) -> OperationResult:
        return await _run(host, "会话已移入已删除", lambda chat: chat.set_deleted(_session(req), deleted=True))

    async def undelete(req: OperationRequest) -> OperationResult:
        return await _run(host, "会话已恢复", lambda chat: chat.set_deleted(_session(req), deleted=False))

    return (
        OperationSpec(
            operation_id="chat_ops.session.rename", path=("session", "rename"),
            summary="覆盖会话显示标题；空标题清除覆盖（幂等）", handler=rename,
            options=(_SESSION_KEY, OptionSpec("title", "新标题（≤512 字，空串清除覆盖）", default="")),
            mirrors=("http:akashic_clients:POST /api/chat/sessions/{session_key:path}/rename",),
        ),
        OperationSpec(
            operation_id="chat_ops.session.delete", path=("session", "delete"),
            summary="软删会话（幂等；消息物理保留，可恢复）", handler=delete, options=(_SESSION_KEY,),
            mirrors=("http:akashic_clients:POST /api/chat/sessions/{session_key:path}/delete",),
        ),
        OperationSpec(
            operation_id="chat_ops.session.undelete", path=("session", "undelete"),
            summary="恢复已软删的会话（幂等）", handler=undelete, options=(_SESSION_KEY,),
            mirrors=("http:akashic_clients:POST /api/chat/sessions/{session_key:path}/undelete",),
        ),
    )


def _nav_specs(host: ChatSessionsHost) -> tuple[OperationSpec, ...]:
    async def pins(_req: OperationRequest) -> OperationResult:
        return await _run(host, "侧栏置顶", lambda chat: chat.read_pins())

    async def pin(req: OperationRequest) -> OperationResult:
        pinned = req.args.get("pinned") is not False
        return await _run(host, "已置顶" if pinned else "已取消置顶", lambda chat: chat.update_pin(
            kind=_text(req, "kind") or "session", id=_text(req, "id") or "", pinned=pinned))

    return (
        OperationSpec(
            operation_id="chat_ops.nav.pins", path=("nav", "pins"),
            summary="读取侧栏置顶（会话与会话维度值，按置顶顺序）", handler=pins,
            mirrors=("http:akashic_clients:GET /api/chat/navigation/pins",),
        ),
        OperationSpec(
            operation_id="chat_ops.nav.pin", path=("nav", "pin"),
            summary="置顶或取消置顶一个会话或会话维度值（如项目）；已置顶项重放不再校验", handler=pin,
            options=(OptionSpec("id", "会话 ID，或维度值（如项目 ID）", required=True),
                     OptionSpec("kind", "session 或会话维度名（如 project）", default="session"),
                     OptionSpec("pinned", "true 置顶 / false 取消", value_type="bool", default=True)),
            mirrors=("http:akashic_clients:POST /api/chat/navigation/pins",),
        ),
    )


def _artifact_view(artifact_id: str, data: bytes, media_type: str | None, filename: str | None) -> OperationResult:
    view: dict[str, object] = {"artifact_id": artifact_id, "filename": filename, "media_type": media_type,
                               "size_bytes": len(data)}
    parts: tuple[ContentPart, ...] = ()
    if media_type is not None and media_type.startswith("image/"):
        # 图片交给模型投影原附件，不把字节塞进正文。
        parts = (ContentPart("artifact_ref", artifact_id),)
        view["encoding"] = "image"
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            view["encoding"] = "binary"
        else:
            view.update(encoding="utf-8", text=text[:_TEXT_LIMIT], truncated=len(text) > _TEXT_LIMIT)
    return OperationResult(True, f"附件 {filename or artifact_id}", data=view, parts=parts)


def _artifact_spec(host: ChatSessionsHost) -> OperationSpec:
    async def read(req: OperationRequest) -> OperationResult:
        try:
            item = await host.current().read_artifact(_text(req, "artifact_id") or "")
        except ChatRequestError as error:
            return OperationResult(False, error.detail, error_code=_ERROR_CODES.get(error.status, "rejected"))
        return _artifact_view(item.artifact_id, item.data, item.media_type, item.filename)

    return OperationSpec(
        operation_id="chat_ops.artifact.read", path=("artifact", "read"),
        summary="读取一个聊天附件：文本给出正文（≤64KiB），图片作为附件引用返回", handler=read,
        options=(OptionSpec("artifact_id", "附件 ID（消息里的 artifact_id）", required=True),),
        mirrors=("http:akashic_clients:GET /api/chat/artifacts/{artifact_id}",),
    )


# 按块读取执行主机文件；固定首块指纹，读取期间文件变化即拒绝。
async def _host_chunks(reader: FileByteReader, path: str) -> AsyncIterator[bytes]:
    offset, version = 0, None
    while True:
        chunk = await reader.read(path, offset=offset, version=version)
        if isinstance(chunk, FileError):
            raise ChatRequestError(400, chunk.text)
        version = chunk.version
        if chunk.data:
            yield chunk.data
        offset += len(chunk.data)
        if not chunk.data or offset >= chunk.size:
            return


def _attachment_spec(host: ChatSessionsHost, files: Files) -> OperationSpec:
    async def add(req: OperationRequest) -> OperationResult:
        path = _text(req, "path") or ""
        if not (path.startswith("/") or path.startswith("~")):
            return OperationResult(False, "需要执行主机上的绝对路径", error_code="invalid_arguments")
        filename = _text(req, "filename") or PurePosixPath(path).name

        async def call(chat: ChatSessions) -> Mapping[str, object]:
            async with files.bytes() as reader:
                return await chat.save_upload(_host_chunks(reader, path), filename)
        return await _run(host, f"已登记附件 {filename}，可随消息发送", call)

    return OperationSpec(
        operation_id="chat_ops.attachment.add", path=("attachment", "add"),
        summary="把执行主机上的一个文件登记为可随消息发送的聊天附件（与界面上传同一登记，≤50MB）", handler=add,
        options=(OptionSpec("path", "执行主机上的绝对路径", required=True),
                 OptionSpec("filename", "附件显示名；默认取路径文件名")),
        mirrors=("http:akashic_clients:POST /api/chat/uploads",),
    )


async def apply(ctx: Context) -> None:
    """聊天通道与 Operation Bus 都在时注册命令；任一缺席都不影响对方。"""

    async def register_chat(child: Context) -> None:
        host = child.require(CHAT_SESSIONS)
        operations = child.require(OPERATIONS_V1)
        for spec in (*_session_read_specs(host), *_session_admin_specs(host), *_nav_specs(host), _artifact_spec(host)):
            _ = await operations.register(child, spec)

    async def register_attachment(child: Context) -> None:
        spec = _attachment_spec(child.require(CHAT_SESSIONS), child.require(FILES))
        _ = await child.require(OPERATIONS_V1).register(child, spec)

    _ = await ctx.inject((OPERATIONS_V1, CHAT_SESSIONS), register_chat, name="operations")
    _ = await ctx.inject((OPERATIONS_V1, CHAT_SESSIONS, FILES), register_attachment, name="operations-attachment")
