from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from infra.mobile_realtime.protocol import (
    COMMAND_TYPES,
    CONTROL_TYPES,
    EVENT_TYPES,
    FRAME_ADAPTER,
    MAX_JSON_FRAME_BYTES,
    PRE_AUTH_CONTROL_TYPES,
    PROTOCOL_VERSION,
)
from infra.mobile_realtime.attachments import MAX_ATTACHMENT_CHUNK_BYTES
from infra.mobile_webui.protocol import (
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


OUTPUT = ROOT / "schema" / "mobile-realtime-v1.json"


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
            "headerSchema": {
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
            "maxHeaderBytes": 1024,
            "payloadOffsetSemantics": "absolute byte offset",
        },
        "commandTypes": sorted(COMMAND_TYPES),
        "eventTypes": sorted(EVENT_TYPES),
        "controlTypes": sorted(CONTROL_TYPES),
        "preAuthControlTypes": sorted(PRE_AUTH_CONTROL_TYPES),
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
    if args.check:
        matches = OUTPUT.is_file() and OUTPUT.read_text(encoding="utf-8") == encoded
        return 0 if matches else 1
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _ = OUTPUT.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
