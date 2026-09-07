"""原生发送读取公开正文和原消息的已验证附件。"""
from dataclasses import dataclass
from typing import cast

from agent.plugin_composition.artifacts import ArtifactRead
from session.artifacts import AttachmentRef
from session.log import MessageCatalog
from session.message import ContentPart, Control, Message


class AttachmentReadError(ValueError):
    """已验证引用的字节当前不可读取；修复后可重新准备发送。"""


@dataclass(frozen=True, slots=True)
class File:
    ref: AttachmentRef
    data: bytes


async def read_content(message: Message, catalog: MessageCatalog, artifacts: ArtifactRead) -> tuple[str | File, ...]:
    """先读完全部附件，避免发送正文后才发现文件损坏。"""
    if isinstance(message.body, Control):
        return ()
    refs = {ref.artifact_id: ref for ref in catalog.reader(message.session_id).attachments(message.message_id)}
    parts: list[str | File] = []
    for part in message.body.parts:
        if not isinstance(part, ContentPart):
            continue
        if part.kind == "text":
            text = cast(str, part.value)
            if text.strip():
                parts.append(text)
        elif part.kind == "artifact_ref":
            ref = refs[cast(str, part.value)]
            try:
                lease = await artifacts.acquire(ref)
                try:
                    data = await lease.read_bytes(max_bytes=ref.size_bytes)
                finally:
                    await lease.aclose()
            except (ValueError, OSError) as error:
                raise AttachmentReadError(f"{type(error).__name__}: {error}") from error
            parts.append(File(ref, data))
    return tuple(parts)
