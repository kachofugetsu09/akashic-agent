from __future__ import annotations

import hashlib
import json
import re
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping

MAX_MANIFEST_BYTES = 1 * 1024 * 1024
MAX_FILES = 2048
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_UNPACKED_BYTES = 64 * 1024 * 1024
MANIFEST_SCHEMA_VERSION = 2

_MIME_ALLOWLIST = frozenset(
    {
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
    }
)
_FORBIDDEN_SUFFIXES = frozenset({".dex", ".jar", ".so", ".apk", ".aab"})
_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SERVER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PLATFORM_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_MIME_BY_SUFFIX = {
    ".html": "text/html",
    ".htm": "text/html",
    ".css": "text/css",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".cjs": "text/javascript",
    ".json": "application/json",
    ".map": "application/json",
    ".wasm": "application/wasm",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".otf": "font/otf",
    ".avif": "image/avif",
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
    ".txt": "text/plain",
}


class ManifestError(ValueError):
    """表示 WebUI manifest 或文件集合违反发布合同。"""


@dataclass(frozen=True, slots=True)
class WebUiFile:
    path: str
    sha256: str
    size_bytes: int
    mime: str

    def as_json(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "mime": self.mime,
        }


@dataclass(frozen=True, slots=True)
class WebUiTarget:
    target_key: str
    generation_id: str
    manifest_digest: str
    manifest_size_bytes: int
    bridge_protocol_min: int
    bridge_protocol_max: int
    snapshot_protocol_min: int
    snapshot_protocol_max: int
    minimum_native_build: int
    platforms: tuple[str, ...]

    def as_json(self) -> dict[str, object]:
        return {
            "target_key": self.target_key,
            "generation_id": self.generation_id,
            "manifest_digest": self.manifest_digest,
            "manifest_size_bytes": self.manifest_size_bytes,
            "bridge_protocol_min": self.bridge_protocol_min,
            "bridge_protocol_max": self.bridge_protocol_max,
            "snapshot_protocol_min": self.snapshot_protocol_min,
            "snapshot_protocol_max": self.snapshot_protocol_max,
            "minimum_native_build": self.minimum_native_build,
            "platforms": list(self.platforms),
        }


@dataclass(frozen=True, slots=True)
class WebUiManifest:
    generation_id: str
    entrypoint: str
    files: tuple[WebUiFile, ...]
    bridge_protocol_min: int
    bridge_protocol_max: int
    snapshot_protocol_min: int
    snapshot_protocol_max: int
    minimum_native_build: int
    platforms: tuple[str, ...]
    source_repository: str
    source_commit: str
    source_tree: str
    input_digest: str
    build_context_digest: str
    dirty_provenance: Mapping[str, object] | None
    reproducible: bool
    builder_identity: Mapping[str, str]
    unpacked_size_bytes: int
    file_count: int

    def as_json_without_digest(self) -> dict[str, object]:
        """Return the complete canonical manifest payload."""

        value: dict[str, object] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "generation_id": self.generation_id,
            "entrypoint": self.entrypoint,
            "files": [item.as_json() for item in self.files],
            "bridge_protocol_min": self.bridge_protocol_min,
            "bridge_protocol_max": self.bridge_protocol_max,
            "snapshot_protocol_min": self.snapshot_protocol_min,
            "snapshot_protocol_max": self.snapshot_protocol_max,
            "minimum_native_build": self.minimum_native_build,
            "platforms": list(self.platforms),
            "source_repository": self.source_repository,
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
            "input_digest": self.input_digest,
            "build_context_digest": self.build_context_digest,
            "dirty_provenance": self.dirty_provenance,
            "reproducible": self.reproducible,
            "builder_identity": self.builder_identity,
            "unpacked_size_bytes": self.unpacked_size_bytes,
            "file_count": self.file_count,
        }
        return value

    def as_json(self) -> dict[str, object]:
        return self.as_json_without_digest()

    def target(self, server_id: str, digest: str | None = None) -> WebUiTarget:
        if _SERVER_ID_RE.fullmatch(server_id) is None:
            raise ManifestError("server_id 无效")
        body = canonical_manifest_bytes(self)
        actual_digest = hashlib.sha256(body).hexdigest()
        if digest is not None and digest != actual_digest:
            raise ManifestError("manifest digest 与 canonical 内容不一致")
        return WebUiTarget(
            target_key=derive_target_key(server_id, self.generation_id, actual_digest),
            generation_id=self.generation_id,
            manifest_digest=actual_digest,
            manifest_size_bytes=len(body),
            bridge_protocol_min=self.bridge_protocol_min,
            bridge_protocol_max=self.bridge_protocol_max,
            snapshot_protocol_min=self.snapshot_protocol_min,
            snapshot_protocol_max=self.snapshot_protocol_max,
            minimum_native_build=self.minimum_native_build,
            platforms=self.platforms,
        )


def manifest_from_json(payload: object) -> WebUiManifest:
    """把数据库或 HTTP 读出的 manifest object 严格恢复为领域值。"""

    if not isinstance(payload, dict):
        raise ManifestError("manifest 顶层必须是 object")
    required = {
        "schema_version", "generation_id", "entrypoint", "files",
        "bridge_protocol_min", "bridge_protocol_max", "snapshot_protocol_min",
        "snapshot_protocol_max", "minimum_native_build", "platforms",
        "source_repository", "source_commit", "source_tree", "input_digest",
        "build_context_digest", "dirty_provenance", "reproducible",
        "builder_identity", "unpacked_size_bytes", "file_count",
    }
    if set(payload) != required:
        raise ManifestError("manifest 字段集合无效")
    raw_files = payload["files"]
    raw_platforms = payload["platforms"]
    if not isinstance(raw_files, list) or not isinstance(raw_platforms, list):
        raise ManifestError("manifest files/platforms 类型无效")
    files: list[WebUiFile] = []
    for raw in raw_files:
        if not isinstance(raw, dict) or set(raw) != {"path", "sha256", "size_bytes", "mime"}:
            raise ManifestError("manifest file 项字段无效")
        files.append(
            WebUiFile(
                path=_require_string(raw["path"], "file.path"),
                sha256=_require_string(raw["sha256"], "file.sha256"),
                size_bytes=_require_int(raw["size_bytes"], "file.size_bytes"),
                mime=_require_string(raw["mime"], "file.mime"),
            )
        )
    dirty = payload["dirty_provenance"]
    if dirty is not None and not isinstance(dirty, dict):
        raise ManifestError("dirty_provenance 必须是 object 或 null")
    builder = payload["builder_identity"]
    if not isinstance(builder, dict):
        raise ManifestError("builder_identity 必须是 object")
    manifest = WebUiManifest(
        generation_id=_require_string(payload["generation_id"], "generation_id"),
        entrypoint=_require_string(payload["entrypoint"], "entrypoint"),
        files=tuple(files),
        bridge_protocol_min=_require_int(payload["bridge_protocol_min"], "bridge_protocol_min"),
        bridge_protocol_max=_require_int(payload["bridge_protocol_max"], "bridge_protocol_max"),
        snapshot_protocol_min=_require_int(payload["snapshot_protocol_min"], "snapshot_protocol_min"),
        snapshot_protocol_max=_require_int(payload["snapshot_protocol_max"], "snapshot_protocol_max"),
        minimum_native_build=_require_int(payload["minimum_native_build"], "minimum_native_build"),
        platforms=tuple(_require_string(item, "platform") for item in raw_platforms),
        source_repository=_require_string(payload["source_repository"], "source_repository"),
        source_commit=_require_string(payload["source_commit"], "source_commit"),
        source_tree=_require_string(payload["source_tree"], "source_tree"),
        input_digest=_require_string(payload["input_digest"], "input_digest"),
        build_context_digest=_require_string(payload["build_context_digest"], "build_context_digest"),
        dirty_provenance=dirty,
        reproducible=_require_bool(payload["reproducible"], "reproducible"),
        builder_identity=_require_string_map(builder, "builder_identity"),
        unpacked_size_bytes=_require_int(payload["unpacked_size_bytes"], "unpacked_size_bytes"),
        file_count=_require_int(payload["file_count"], "file_count"),
    )
    if payload["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ManifestError("manifest schema_version 不受支持")
    validate_manifest(manifest)
    return manifest


def canonical_manifest_bytes(manifest: WebUiManifest) -> bytes:
    """编码完整 manifest，manifest_digest 是此结果的 sha256。"""

    validate_manifest(manifest)
    encoded = json.dumps(
        manifest.as_json_without_digest(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise ManifestError("manifest 超过 1 MiB 上限")
    return encoded


def canonical_generation_identity_bytes(manifest: WebUiManifest) -> bytes:
    """编码决定 generation_id 的 manifest 内容（不含 generation_id）。"""

    body = manifest.as_json_without_digest().copy()
    _ = body.pop("generation_id")
    return json.dumps(
        body,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def generation_id_for_manifest(manifest: WebUiManifest) -> str:
    return hashlib.sha256(canonical_generation_identity_bytes(manifest)).hexdigest()


def derive_target_key(server_id: str, generation_id: str, digest: str) -> str:
    body = json.dumps(
        {"server_id": server_id, "generation_id": generation_id, "manifest_digest": digest},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def manifest_digest(manifest: WebUiManifest) -> str:
    return hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()


def validate_manifest(manifest: WebUiManifest) -> None:
    """在发布和 HTTP 读取边界统一校验 manifest 不变量。"""

    if not _valid_digest(manifest.generation_id):
        raise ManifestError("generation_id 必须是 64 位小写 sha256")
    if generation_id_for_manifest(manifest) != manifest.generation_id:
        raise ManifestError("generation_id 必须由 manifest 内容确定性派生")
    if manifest.entrypoint != "mobile.html":
        raise ManifestError("entrypoint 必须是 mobile.html")
    if manifest.entrypoint not in {item.path for item in manifest.files}:
        raise ManifestError("entrypoint 不在文件清单中")
    for minimum, maximum, label in (
        (manifest.bridge_protocol_min, manifest.bridge_protocol_max, "bridge_protocol"),
        (manifest.snapshot_protocol_min, manifest.snapshot_protocol_max, "snapshot_protocol"),
    ):
        if not isinstance(minimum, int) or isinstance(minimum, bool) or not isinstance(maximum, int) or isinstance(maximum, bool) or minimum < 1 or minimum > maximum:
            raise ManifestError(f"{label} 兼容窗口无效")
    if not isinstance(manifest.minimum_native_build, int) or isinstance(manifest.minimum_native_build, bool) or manifest.minimum_native_build < 1:
        raise ManifestError("minimum_native_build 必须为正整数")
    _validate_dirty_provenance(manifest.dirty_provenance)
    if manifest.reproducible != (manifest.dirty_provenance is None):
        raise ManifestError("reproducible 必须与 dirty_provenance 一致")
    _validate_builder_identity(manifest.builder_identity)
    for value, label in (
        (manifest.source_repository, "source_repository"),
        (manifest.source_commit, "source_commit"),
        (manifest.source_tree, "source_tree"),
        (manifest.input_digest, "input_digest"),
        (manifest.build_context_digest, "build_context_digest"),
    ):
        _require_nonempty_token(value, label)
    _require_git_object(manifest.source_commit, "source_commit")
    _require_git_object(manifest.source_tree, "source_tree")
    for value, label in (
        (manifest.input_digest, "input_digest"),
        (manifest.build_context_digest, "build_context_digest"),
    ):
        if not _valid_digest(value):
            raise ManifestError(f"{label} 必须是 sha256")
    _validate_platforms(manifest.platforms)

    if not manifest.files or len(manifest.files) > MAX_FILES:
        raise ManifestError("files 数量超出范围")
    if tuple(item.path for item in manifest.files) != tuple(
        sorted((item.path for item in manifest.files), key=lambda value: value.encode("utf-8"))
    ):
        raise ManifestError("files 必须按 NFC UTF-8 path 顺序排列")
    if manifest.file_count != len(manifest.files):
        raise ManifestError("file_count 与 files 不一致")
    seen: set[str] = set()
    seen_folded: set[str] = set()
    digest_metadata: dict[str, tuple[int, str]] = {}
    total = 0
    for item in manifest.files:
        normalized = _normalise_path(item.path)
        if normalized != item.path:
            raise ManifestError(f"文件路径必须使用 NFC POSIX 形式: {item.path}")
        folded = normalized.lower()
        if normalized in seen or folded in seen_folded:
            raise ManifestError(f"文件路径冲突: {item.path}")
        seen.add(normalized)
        seen_folded.add(folded)
        if not _valid_digest(item.sha256):
            raise ManifestError(f"文件 sha256 无效: {item.path}")
        if not isinstance(item.size_bytes, int) or isinstance(item.size_bytes, bool) or not 0 <= item.size_bytes <= MAX_FILE_BYTES:
            raise ManifestError(f"文件大小超出上限: {item.path}")
        if item.mime not in _MIME_ALLOWLIST:
            raise ManifestError(f"MIME 不在 allowlist: {item.path}")
        expected_mime = _MIME_BY_SUFFIX.get(Path(item.path).suffix.lower(), "application/octet-stream")
        if item.mime != expected_mime:
            raise ManifestError(f"MIME 与固定后缀映射不一致: {item.path}")
        metadata = (item.size_bytes, item.mime)
        previous_metadata = digest_metadata.setdefault(item.sha256, metadata)
        if previous_metadata != metadata:
            raise ManifestError("同一 generation 内 digest 不得映射多个 size/mime")
        if Path(item.path).suffix.lower() in _FORBIDDEN_SUFFIXES:
            raise ManifestError(f"文件类型禁止进入 WebUI: {item.path}")
        total += item.size_bytes
    if manifest.unpacked_size_bytes != total:
        raise ManifestError("unpacked_size_bytes 与 files 不一致")
    if total > MAX_UNPACKED_BYTES:
        raise ManifestError("manifest unpacked 总大小超过 64 MiB")


def manifest_from_directory(
    root: Path,
    *,
    entrypoint: str = "mobile.html",
    bridge_protocol_min: int = 1,
    bridge_protocol_max: int = 1,
    snapshot_protocol_min: int = 7,
    snapshot_protocol_max: int = 7,
    minimum_native_build: int = 45,
    platforms: tuple[str, ...] = ("android",),
    source_repository: str,
    source_commit: str,
    source_tree: str,
    input_digest: str,
    build_context_digest: str,
    dirty_provenance: Mapping[str, object] | None,
    reproducible: bool,
    builder_identity: Mapping[str, str],
) -> tuple[WebUiManifest, dict[str, bytes]]:
    """从已构建目录读取普通文件并生成确定性 manifest 与内容。"""

    if not root.is_dir():
        raise ManifestError(f"WebUI build 目录不存在: {root}")
    files: list[WebUiFile] = []
    contents: dict[str, bytes] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix().encode("utf-8")):
        if path.is_symlink():
            raise ManifestError(f"WebUI build 不允许 symlink: {path}")
        if path.is_dir():
            continue
        try:
            mode = path.stat().st_mode
        except OSError as error:
            raise ManifestError(f"WebUI build 文件无法读取: {path}") from error
        if not stat.S_ISREG(mode):
            raise ManifestError(f"WebUI build 只允许普通文件: {path}")
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        mime = _MIME_BY_SUFFIX.get(Path(relative).suffix.lower(), "application/octet-stream")
        files.append(WebUiFile(relative, digest, len(data), mime))
        contents[relative] = data
    draft = WebUiManifest(
        generation_id="0" * 64,
        entrypoint=entrypoint,
        files=tuple(files),
        bridge_protocol_min=bridge_protocol_min,
        bridge_protocol_max=bridge_protocol_max,
        snapshot_protocol_min=snapshot_protocol_min,
        snapshot_protocol_max=snapshot_protocol_max,
        minimum_native_build=minimum_native_build,
        platforms=platforms,
        source_repository=source_repository,
        source_commit=source_commit,
        source_tree=source_tree,
        input_digest=input_digest,
        build_context_digest=build_context_digest,
        dirty_provenance=dirty_provenance,
        reproducible=reproducible,
        builder_identity=builder_identity,
        unpacked_size_bytes=sum(item.size_bytes for item in files),
        file_count=len(files),
    )
    manifest = WebUiManifest(
        generation_id=generation_id_for_manifest(draft),
        entrypoint=draft.entrypoint,
        files=draft.files,
        bridge_protocol_min=draft.bridge_protocol_min,
        bridge_protocol_max=draft.bridge_protocol_max,
        snapshot_protocol_min=draft.snapshot_protocol_min,
        snapshot_protocol_max=draft.snapshot_protocol_max,
        minimum_native_build=draft.minimum_native_build,
        platforms=draft.platforms,
        source_repository=draft.source_repository,
        source_commit=draft.source_commit,
        source_tree=draft.source_tree,
        input_digest=draft.input_digest,
        build_context_digest=draft.build_context_digest,
        dirty_provenance=draft.dirty_provenance,
        reproducible=draft.reproducible,
        builder_identity=draft.builder_identity,
        unpacked_size_bytes=draft.unpacked_size_bytes,
        file_count=draft.file_count,
    )
    validate_manifest(manifest)
    return manifest, contents


def _normalise_path(value: str) -> str:
    if not value or "\\" in value:
        raise ManifestError(f"文件路径无效: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ManifestError(f"文件路径必须是安全相对路径: {value!r}")
    if "/".join(path.parts) != value:
        raise ManifestError(f"文件路径包含非规范分隔符: {value!r}")
    if not 1 <= len(value.encode("utf-8")) <= 512:
        raise ManifestError("文件路径 UTF-8 长度必须在 1..512 bytes")
    for segment in path.parts:
        if not 1 <= len(segment.encode("utf-8")) <= 128 or _PATH_SEGMENT_RE.fullmatch(segment) is None:
            raise ManifestError(f"文件路径 segment 只能使用 ASCII [A-Za-z0-9._-]: {value!r}")
    normalized = unicodedata.normalize("NFC", value)
    if normalized != value or normalized.startswith("/"):
        raise ManifestError(f"文件路径不是 NFC 相对形式: {value!r}")
    return normalized


def _require_nonempty_token(value: str, label: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 2048 or any(char.isspace() for char in value):
        raise ManifestError(f"{label} 无效")


def _valid_digest(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _validate_platforms(platforms: tuple[str, ...]) -> None:
    if not platforms or len(platforms) > 64 or len(set(platforms)) != len(platforms):
        raise ManifestError("platforms 必须为非空无重复列表")
    if tuple(platforms) != tuple(sorted(platforms, key=lambda value: value.encode("utf-8"))):
        raise ManifestError("platforms 必须按规范顺序排列")
    for value in platforms:
        if not isinstance(value, str) or _PLATFORM_RE.fullmatch(value) is None:
            raise ManifestError("platform 必须是 ASCII token")


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ManifestError(f"{label} 必须是字符串")
    return value


def _require_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{label} 必须是整数")
    return value


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ManifestError(f"{label} 必须是 bool")
    return value


def _require_string_map(value: dict[object, object], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ManifestError(f"{label} 必须是 string map")
        result[key] = item
    return result


def _validate_dirty_provenance(value: Mapping[str, object] | None) -> None:
    if value is None:
        return
    if set(value) != {"base_commit", "tracked_patch_digest", "untracked_tree_digest"}:
        raise ManifestError("dirty_provenance 字段集合无效")
    for key, item in value.items():
        if not isinstance(item, str) or not item:
            raise ManifestError(f"dirty_provenance.{key} 无效")
        if key != "base_commit" and not _valid_digest(item):
            raise ManifestError(f"dirty_provenance.{key} 必须是 sha256")
        if key == "base_commit":
            _require_git_object(item, "dirty_provenance.base_commit")


def _validate_builder_identity(value: Mapping[str, str]) -> None:
    if set(value) != {"node_version", "npm_version", "package_lock_digest", "build_script_digest"}:
        raise ManifestError("builder_identity 字段集合无效")
    for key, item in value.items():
        if not isinstance(item, str) or not item or len(item) > 64 or any(char.isspace() for char in item):
            raise ManifestError(f"builder_identity.{key} 无效")
    for key in ("package_lock_digest", "build_script_digest"):
        if not _valid_digest(value[key]):
            raise ManifestError(f"builder_identity.{key} 必须是 sha256")


def _require_git_object(value: str, label: str) -> None:
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ManifestError(f"{label} 必须是 git commit/tree 摘要")


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "MAX_FILE_BYTES",
    "MAX_FILES",
    "MAX_MANIFEST_BYTES",
    "MAX_UNPACKED_BYTES",
    "ManifestError",
    "WebUiFile",
    "WebUiManifest",
    "WebUiTarget",
    "canonical_generation_identity_bytes",
    "canonical_manifest_bytes",
    "derive_target_key",
    "generation_id_for_manifest",
    "manifest_digest",
    "manifest_from_directory",
    "manifest_from_json",
    "validate_manifest",
]
