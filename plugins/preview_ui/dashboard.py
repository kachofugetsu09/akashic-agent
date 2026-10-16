"""预览标签的浏览器入口：只读会话里某条消息的附件开头，解成文本返回。"""
from __future__ import annotations

import codecs

from fastapi import FastAPI, HTTPException, Query

from agent.plugin_composition import DashboardContext
from plugins.ledger.contract import ARTIFACT_READ, MESSAGE_CATALOG, AttachmentRef

inject = (MESSAGE_CATALOG, ARTIFACT_READ)
# 与聊天端文本预览同一上限：大日志只看开头，其余交给下载。
PREVIEW_BYTES = 512 * 1024


# 每个参数各用一个 Query：同一个 FieldInfo 被多个参数共用时，FastAPI 会把它们都绑定到第一个参数名上。
def _id() -> str:
    return Query(min_length=1, max_length=512)


def register(app: FastAPI, context: DashboardContext) -> None:
    @app.get("/api/dashboard/preview_ui/text")
    async def text(session: str = _id(), message: str = _id(), artifact: str = _id()) -> dict[str, object]:
        ref = _attachment(context, session, message, artifact)
        raw = await _read_prefix(context, ref)
        complete = ref.size_bytes <= PREVIEW_BYTES
        decoded = _decode(raw, complete=complete)
        if decoded is None:
            raise HTTPException(415, "这个附件不是 UTF-8 文本，无法预览")
        return {
            "artifact_id": ref.artifact_id, "name": ref.filename or "附件", "media_type": ref.media_type,
            "size_bytes": ref.size_bytes, "complete": complete, "text": decoded,
        }


# 附件必须属于这个会话里的这条消息；其它会话的消息与不在消息上的附件一律当作不存在。
def _attachment(context: DashboardContext, session: str, message: str, artifact: str) -> AttachmentRef:
    try:
        refs = context.require(MESSAGE_CATALOG).reader(session).attachments(message)
    except (LookupError, ValueError) as error:
        raise HTTPException(404, "没有找到这条消息") from error
    found = next((ref for ref in refs if ref.artifact_id == artifact), None)
    if found is None:
        raise HTTPException(404, "这条消息里没有这个附件")
    return found


async def _read_prefix(context: DashboardContext, ref: AttachmentRef) -> bytes:
    try:
        lease = await context.require(ARTIFACT_READ).acquire(ref)
        try:
            return await lease.read_chunk(offset=0, max_bytes=min(ref.size_bytes, PREVIEW_BYTES))
        finally:
            await lease.aclose()
    except (ValueError, OSError) as error:
        raise HTTPException(410, "附件已不可读") from error


# 带 NUL 或非法字节的视为二进制；截断处落在多字节字符中间时只丢掉未写完的尾部。
def _decode(raw: bytes, *, complete: bool) -> str | None:
    if b"\x00" in raw:
        return None
    try:
        return codecs.getincrementaldecoder("utf-8")().decode(raw, final=complete)
    except UnicodeDecodeError:
        return None
