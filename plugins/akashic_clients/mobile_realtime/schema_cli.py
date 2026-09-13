from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import types
from pathlib import Path

def _add_explicit_core_root() -> None:
    """Add only the explicitly selected Core root for direct CLI execution."""

    configured = os.environ.get("AKASHIC_CORE_ROOT", "").strip()
    candidate = Path(configured).expanduser() if configured else Path.cwd()
    if configured or (candidate / "agent" / "plugin_composition").is_dir():
        root = candidate.resolve(strict=True)
        if not (root / "agent" / "plugin_composition").is_dir():
            raise RuntimeError(f"Core root 缺少 agent/plugin_composition: {root}")
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)


def _direct_package_identity() -> str:
    """Create a collision-checked package identity rooted at this artifact."""

    artifact_root = Path(__file__).resolve().parents[1]
    token = hashlib.sha256(str(artifact_root).encode("utf-8")).hexdigest()[:20]
    package_name = f"_akashic_clients_cli_{token}"
    existing = sys.modules.get(package_name)
    if existing is not None:
        paths = getattr(existing, "__path__", ())
        if tuple(paths) != (str(artifact_root),):
            raise RuntimeError(f"插件 CLI package identity 冲突: {package_name}")
        return package_name
    package = types.ModuleType(package_name)
    package.__path__ = [str(artifact_root)]  # type: ignore[attr-defined]
    package.__package__ = package_name
    sys.modules[package_name] = package
    return package_name


if not __package__:
    _add_explicit_core_root()
    __package__ = f"{_direct_package_identity()}.mobile_realtime"

from .protocol import (
    COMMAND_TYPES,
    CONTROL_TYPES,
    EVENT_TYPES,
    FRAME_ADAPTER,
    MAX_JSON_FRAME_BYTES,
    PRE_AUTH_CONTROL_TYPES,
    PROTOCOL_VERSION,
)
from .attachments import MAX_ATTACHMENT_CHUNK_BYTES
from ..mobile_webui.protocol import (
    BuilderIdentityWire,
    DirtyProvenanceWire,
    ErrorReplyWire,
    HttpErrorBodyWire,
    PrepareReplyWire,
    ReleaseViewWire,
    WebUiFileWire,
    WebUiManifestWire,
    WebUiTargetWire,
)
from pydantic import TypeAdapter


def _source_schema_output() -> Path | None:
    """Resolve a writable source output without reaching outside this plugin."""

    configured = os.environ.get("AKASHIC_SCHEMA_OUTPUT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve(strict=False)
    candidate = Path.cwd() / "schema" / "mobile-realtime-v1.json"
    return candidate if candidate.parent.is_dir() else None


def build_schema() -> dict[str, object]:
    """从服务端帧模型生成确定性的移动协议 schema。"""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Akashic Mobile Realtime Protocol v1",
        "protocolVersion": PROTOCOL_VERSION,
        "transport": "WebSocket JSON text frames and attachment binary chunks",
        "maxJsonFrameBytes": MAX_JSON_FRAME_BYTES,
        "maxAttachmentChunkBytes": MAX_ATTACHMENT_CHUNK_BYTES,
        "attachmentBinaryFrame": {
            "byteOrder": "big-endian",
            "layout": [
                "uint32 header_length",
                "header_length bytes UTF-8 JSON header",
                "remaining bytes chunk payload",
            ],
            "uploadHeaderSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["attachment_id", "offset"],
                "properties": {
                    "attachment_id": {
                        "type": "string",
                        "minLength": 26,
                        "maxLength": 36,
                    },
                    "offset": {"type": "integer", "minimum": 0},
                },
            },
            "downloadHeaderSchema": {
                "type": "object", "additionalProperties": False,
                "required": ["artifact_id", "offset"],
                "properties": {
                    "artifact_id": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$"},
                    "offset": {"type": "integer", "minimum": 0},
                },
            },
            "downloadPayloadMinimumBytes": 0,
            "uploadPayloadMinimumBytes": 1,
            "maxHeaderBytes": 1024,
            "payloadOffsetSemantics": "absolute byte offset",
        },
        "commandTypes": sorted(COMMAND_TYPES),
        "eventTypes": sorted(EVENT_TYPES),
        "controlTypes": sorted(CONTROL_TYPES),
        "preAuthControlTypes": sorted(PRE_AUTH_CONTROL_TYPES),
        "messageLog": {
            "version": 2,
            "historyGet": {
                "command": "history.get",
                "required": ["message_log_version"],
                "optional": ["page_size", "direction", "after_seq", "before_seq", "through_seq", "around_id", "display_only"],
                "direction": "forward (legacy default) or backward",
                "backward": "latest page when no cursor; before_seq is exclusive; around_id ends a page at that Message or returns message_not_found",
                "exclusive": "backward forbids after_seq; forward forbids before_seq and around_id; before_seq and around_id are mutually exclusive",
                "pageSize": {"minimum": 1, "maximum": 200},
            },
            "historyPage": {
                "event": "history.page",
                "common": ["version", "items", "after_seq", "next_after_seq", "through_seq", "has_more"],
                "backward": ["direction", "before_seq", "next_before_seq", "request_id"],
                "range": "all manifests in (after_seq,next_after_seq] have been delivered; this does not mean bodies or attachments are downloaded",
                "snapshot": "through_seq is the fixed Session head; backward next_after_seq equals before_seq-1",
                "older": "next_before_seq is the first delivered seq; has_more refers to older messages; after_seq=-1 only when the start is reached",
                "budget": "240 KiB; backward pages retain the newest suffix when the frame budget is reached",
            },
            "displayOnly": {
                "requestField": "display_only",
                "default": False,
                "parts": "history.provenance, history.record and history.turn_input retain kind and display=unavailable; part indexes and history.transcript stay intact",
                "follow": "session.follow accepts the same display_only flag",
                "reference": "message_ref.display_only defaults to false; byte_length and sha256 select the exact legacy or display representation",
                "download": "message.content.prepare and authenticated Range reads honor either known representation digest",
            },
        },
        "mobileWebUi": {
            "capability": "mobile-webui-ota-v1",
            "commandTypes": [
                "mobile.webui.release.get",
                "mobile.webui.content.prepare",
            ],
            "controlType": "mobile.webui.release.changed",
            "contentPrepareReply": {
                "type": "mobile.webui.content.prepare.ok",
                "fields": ["target_key", "manifest_digest", "ticket", "expires_at"],
                "paths": {
                    "manifest": "/mobile/webui/v1/manifest/{manifest_digest}",
                    "blob": "/mobile/webui/v1/blob/{blob_digest}",
                },
            },
            "schemas": {
                "ReleaseView": TypeAdapter(ReleaseViewWire).json_schema(),
                "Target": TypeAdapter(WebUiTargetWire).json_schema(),
                "Manifest": TypeAdapter(WebUiManifestWire).json_schema(),
                "ManifestFile": TypeAdapter(WebUiFileWire).json_schema(),
                "DirtyProvenance": TypeAdapter(DirtyProvenanceWire).json_schema(),
                "BuilderIdentity": TypeAdapter(BuilderIdentityWire).json_schema(),
                "PrepareReply": TypeAdapter(PrepareReplyWire).json_schema(),
                "ErrorReply": TypeAdapter(ErrorReplyWire).json_schema(),
                "HttpErrorBody": TypeAdapter(HttpErrorBodyWire).json_schema(),
            },
            "releaseView": {
                "fields": [
                    "server_id", "release_epoch", "sequence", "selection_digest",
                    "stable", "preview",
                ],
                "stable": "Target|null",
                "preview": "Target|null",
                "selectionDigest": "sha256(canonical UTF-8 JSON {server_id,stable_target_key,preview_target_key}; null keys are present)",
                "sequenceSemantics": "audit-only; clients never order or choose by sequence/time/semver",
                "noPublication": "stable=null and preview=null is the desired baseline",
            },
            "target": {
                "fields": [
                    "target_key", "generation_id", "manifest_digest", "manifest_size_bytes",
                    "bridge_protocol_min", "bridge_protocol_max", "snapshot_protocol_min",
                    "snapshot_protocol_max", "minimum_native_build", "platforms",
                ],
                "targetKey": "sha256(canonical UTF-8 JSON {server_id,generation_id,manifest_digest})",
            },
            "manifest": {
                "schemaVersion": 2,
                "fields": [
                    "schema_version", "generation_id", "entrypoint", "files",
                    "bridge_protocol_min", "bridge_protocol_max", "snapshot_protocol_min",
                    "snapshot_protocol_max", "minimum_native_build", "platforms",
                    "source_repository", "source_commit", "source_tree", "input_digest",
                    "build_context_digest", "dirty_provenance", "reproducible",
                    "builder_identity", "unpacked_size_bytes", "file_count",
                ],
                "generationId": "sha256(canonical complete manifest with generation_id omitted)",
                "manifestDigest": "sha256(canonical complete manifest)",
                "canonical": "UTF-8 JSON, sorted object keys, no insignificant whitespace, NFC paths, files/platforms UTF-8 order; duplicate/unknown fields rejected",
                "limits": {
                    "manifestBytes": 1048576,
                    "files": 2048,
                    "fileBytes": 8388608,
                    "unpackedBytes": 67108864,
                },
                "provenance": {
                    "dirty_provenance": "null or {base_commit,tracked_patch_digest,untracked_tree_digest}",
                    "builder_identity": "{node_version,npm_version,package_lock_digest,build_script_digest}",
                    "stable": "reproducible=true and dirty_provenance=null",
                },
            },
            "ticket": {
                "audience": "mobile-webui-v1",
                "ttlSeconds": 300,
                "claims": [
                    "aud", "v", "server_id", "device_id", "connection_epoch", "target_key",
                    "generation_id", "manifest_digest", "selection_digest", "release_epoch", "iat", "exp",
                ],
                "scope": "one target manifest plus all blobs listed by that target manifest",
                "recheck": "signature, server/device/revoke, connection_epoch, release_epoch, selection_digest and target membership on every HTTP request",
            },
            "http": {
                "manifest": "/mobile/webui/v1/manifest/{manifest_digest}",
                "blob": "/mobile/webui/v1/blob/{blob_digest}",
                "manifestCacheControl": "no-store",
                "blobCacheControl": "immutable",
                "range": "one bytes range, response <= 8388608 bytes",
                "statuses": {
                    "invalid_ticket": 401,
                    "target_changed": 409,
                    "resource_not_found": 404,
                    "invalid_range": 416,
                    "range_precondition_failed": 412,
                    "release_store_corrupt": 500,
                },
            },
        },
        "frame": FRAME_ADAPTER.json_schema(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    encoded = (
        json.dumps(build_schema(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )
    output = _source_schema_output()
    if args.check:
        # Installed artifacts carry the generator and protocol modules, while
        # the checked-in generated file belongs only to the source checkout.
        matches = output is None or (
            output.is_file() and output.read_text(encoding="utf-8") == encoded
        )
        return 0 if matches else 1
    if output is None:
        raise RuntimeError(
            "生成 schema 需要 source schema 目录或 AKASHIC_SCHEMA_OUTPUT"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    _ = output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
