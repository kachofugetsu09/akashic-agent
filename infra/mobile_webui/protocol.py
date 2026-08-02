from __future__ import annotations

import unicodedata
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Digest = Annotated[str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")]
GitSha1 = Annotated[str, Field(min_length=40, max_length=40, pattern=r"^[0-9a-f]{40}$")]
ServerId = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
PlatformToken = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]{1,64}$")]
Token = Annotated[str, Field(min_length=1, max_length=2048, pattern=r"^\S+$")]
MimeType = Literal[
    "application/json",
    "application/octet-stream",
    "application/wasm",
    "font/otf",
    "font/ttf",
    "font/woff",
    "font/woff2",
    "image/avif",
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/svg+xml",
    "image/webp",
    "text/css",
    "text/html",
    "text/javascript",
    "text/plain",
    "video/mp4",
    "audio/mpeg",
    "audio/ogg",
    "audio/wav",
]
ErrorCode = Literal[
    "capability_required",
    "invalid_range",
    "invalid_ticket",
    "range_precondition_failed",
    "release_store_corrupt",
    "resource_not_found",
    "rollback_unavailable",
    "target_changed",
    "target_not_found",
]


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DirtyProvenanceWire(WireModel):
    base_commit: GitSha1
    tracked_patch_digest: Digest
    untracked_tree_digest: Digest


class BuilderIdentityWire(WireModel):
    node_version: Token = Field(max_length=64)
    npm_version: Token = Field(max_length=64)
    package_lock_digest: Digest
    build_script_digest: Digest


class WebUiFileWire(WireModel):
    path: str = Field(min_length=1, max_length=512, pattern=r"^[A-Za-z0-9._-]{1,128}(?:/[A-Za-z0-9._-]{1,128})*$")
    sha256: Digest
    size_bytes: int = Field(ge=0, le=8 * 1024 * 1024)
    mime: MimeType

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if unicodedata.normalize("NFC", value) != value or any(segment in {".", ".."} for segment in value.split("/")):
            raise ValueError("path 必须是 NFC 安全相对路径")
        if value.rsplit("/", 1)[-1].lower().endswith((".dex", ".jar", ".so", ".apk", ".aab")):
            raise ValueError("path 后缀不允许进入 WebUI")
        return value


class WebUiManifestWire(WireModel):
    schema_version: Literal[2]
    generation_id: Digest
    entrypoint: Literal["mobile.html"]
    files: list[WebUiFileWire] = Field(min_length=1, max_length=2048)
    bridge_protocol_min: int = Field(ge=1)
    bridge_protocol_max: int = Field(ge=1)
    snapshot_protocol_min: int = Field(ge=1)
    snapshot_protocol_max: int = Field(ge=1)
    minimum_native_build: int = Field(ge=1)
    platforms: list[PlatformToken] = Field(min_length=1, max_length=64)
    source_repository: Token = Field(max_length=2048)
    source_commit: GitSha1
    source_tree: GitSha1
    input_digest: Digest
    build_context_digest: Digest
    dirty_provenance: DirtyProvenanceWire | None
    reproducible: bool
    builder_identity: BuilderIdentityWire
    unpacked_size_bytes: int = Field(ge=0, le=64 * 1024 * 1024)
    file_count: int = Field(ge=1, le=2048)

    @model_validator(mode="after")
    def validate_manifest_contract(self) -> WebUiManifestWire:
        _validate_protocol_window(self.bridge_protocol_min, self.bridge_protocol_max, "bridge_protocol")
        _validate_protocol_window(self.snapshot_protocol_min, self.snapshot_protocol_max, "snapshot_protocol")
        _validate_sorted_tokens(self.platforms, "platforms")
        if self.file_count != len(self.files):
            raise ValueError("file_count 与 files 不一致")
        if tuple(item.path for item in self.files) != tuple(sorted((item.path for item in self.files), key=lambda value: value.encode("utf-8"))):
            raise ValueError("files 必须按 UTF-8 path 顺序排列")
        folded = [item.path.lower() for item in self.files]
        if len(set(folded)) != len(folded):
            raise ValueError("files 存在大小写折叠冲突")
        digest_metadata: dict[str, tuple[int, MimeType]] = {}
        for item in self.files:
            metadata = (item.size_bytes, item.mime)
            previous_metadata = digest_metadata.setdefault(item.sha256, metadata)
            if previous_metadata != metadata:
                raise ValueError("同一 generation 内 digest 不得映射多个 size/mime")
        total = sum(item.size_bytes for item in self.files)
        if total != self.unpacked_size_bytes or total > 64 * 1024 * 1024:
            raise ValueError("unpacked_size_bytes 与 files 不一致")
        if self.reproducible != (self.dirty_provenance is None):
            raise ValueError("reproducible 必须与 dirty_provenance 一致")
        return self


class WebUiTargetWire(WireModel):
    target_key: Digest
    generation_id: Digest
    manifest_digest: Digest
    manifest_size_bytes: int = Field(ge=1, le=1024 * 1024)
    bridge_protocol_min: int = Field(ge=1)
    bridge_protocol_max: int = Field(ge=1)
    snapshot_protocol_min: int = Field(ge=1)
    snapshot_protocol_max: int = Field(ge=1)
    minimum_native_build: int = Field(ge=1)
    platforms: list[PlatformToken] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_target_contract(self) -> WebUiTargetWire:
        _validate_protocol_window(self.bridge_protocol_min, self.bridge_protocol_max, "bridge_protocol")
        _validate_protocol_window(self.snapshot_protocol_min, self.snapshot_protocol_max, "snapshot_protocol")
        _validate_sorted_tokens(self.platforms, "platforms")
        return self


class ReleaseViewWire(WireModel):
    server_id: ServerId
    release_epoch: str = Field(
        min_length=36,
        max_length=36,
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    )
    sequence: int = Field(ge=0)
    selection_digest: Digest
    stable: WebUiTargetWire | None
    preview: WebUiTargetWire | None

    @model_validator(mode="after")
    def validate_release_epoch(self) -> ReleaseViewWire:
        try:
            parsed = UUID(self.release_epoch)
        except ValueError as error:
            raise ValueError("release_epoch 必须是 UUID4") from error
        if parsed.version != 4 or str(parsed) != self.release_epoch:
            raise ValueError("release_epoch 必须是规范小写 UUID4")
        return self


class PrepareReplyWire(WireModel):
    target_key: Digest
    manifest_digest: Digest
    ticket: str = Field(min_length=1, max_length=4096)
    expires_at: str = Field(min_length=1, max_length=64)


class ErrorReplyWire(WireModel):
    code: ErrorCode
    message: str = Field(min_length=1, max_length=512)


class HttpErrorBodyWire(WireModel):
    error: ErrorReplyWire


def _validate_protocol_window(minimum: int, maximum: int, label: str) -> None:
    if minimum > maximum:
        raise ValueError(f"{label} 兼容窗口无效")


def _validate_sorted_tokens(values: list[str], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} 不得重复")
    if tuple(values) != tuple(sorted(values, key=lambda value: value.encode("utf-8"))):
        raise ValueError(f"{label} 必须按 UTF-8 顺序排列")


__all__ = [
    "BuilderIdentityWire",
    "DirtyProvenanceWire",
    "ErrorReplyWire",
    "HttpErrorBodyWire",
    "PrepareReplyWire",
    "ReleaseViewWire",
    "WebUiFileWire",
    "WebUiManifestWire",
    "WebUiTargetWire",
]
