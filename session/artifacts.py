from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

_ATTACHMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
_MEDIA_TYPE = re.compile(r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AttachmentKind(StrEnum):
    FILE = "file"
    IMAGE = "image"


@dataclass(frozen=True, slots=True)
class AttachmentRef:
    """标识不可变附件，不暴露存储路径。"""

    artifact_id: str
    kind: AttachmentKind
    filename: str | None
    media_type: str | None
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _ = check_artifact_id(self.artifact_id)
        if not isinstance(self.kind, AttachmentKind):
            raise TypeError("kind 必须是 AttachmentKind")
        _ = _attachment_filename(self.filename)
        _ = _attachment_media_type(self.media_type)
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise TypeError("size_bytes 必须是 int")
        if self.size_bytes < 0:
            raise ValueError("size_bytes 不能是负数")
        if not isinstance(self.sha256, str):
            raise TypeError("sha256 必须是 str")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("sha256 必须是 64 位小写十六进制字符串")



def check_artifact_id(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("artifact_id 必须是 str")
    if _ATTACHMENT_ID.fullmatch(value) is None:
        raise ValueError("artifact_id 必须是安全的 opaque id")
    return value


def _attachment_filename(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("filename 必须是 str 或 None")
    if (
        not value
        or value != value.strip()
        or len(value) > 255
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError("filename 必须是 1..255 字符的纯文件名")
    return value


def _attachment_media_type(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("media_type 必须是 str 或 None")
    if len(value) > 255 or _MEDIA_TYPE.fullmatch(value) is None:
        raise ValueError("media_type 必须是合法 MIME type")
    return value



class AttachmentReadLease(Protocol):
    @property
    def ref(self) -> AttachmentRef: ...

    async def read_bytes(self, *, max_bytes: int) -> bytes: ...

    async def aclose(self) -> None: ...


class AttachmentReadPort(Protocol):
    async def acquire(self, ref: AttachmentRef) -> AttachmentReadLease: ...
