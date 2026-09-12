"""用真实 Gateway、MessageLog 和 ArtifactStore 验证 Android 历史消费；只创建隔离数据。"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import closing
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from infra.channels.artifacts import ChannelAttachmentArtifactStore
from infra.channels.base import AttachmentStore
from plugins.akashic_clients.mobile_realtime.gateway import build_mobile_gateway_runtime, build_mobile_gateway_server
from session.artifact_store import ArtifactStore
from session.artifacts import AttachmentKind
from session.log import MessageLog
from session.message import ContentPart, ContentReferences, Input, Output
from tests_scenarios.mobile_isolated_gateway import EphemeralMasterKeys, approve_pairing, build_config

SESSION = "akashic:00000000000070008000000000000001"
OTHER_SESSION = "akashic:00000000000070008000000000000002"
TAIL_SESSION = "akashic:00000000000070008000000000000003"


class HistoryBus:
    def bind_mobile_channel_inbound_recoverer(self, recoverer: object) -> None:
        self.recoverer = recoverer


async def run(root: Path, port: int, tail_count: int = 0) -> None:
    """固定小型旧历史与分片附件，启动可配对的真实 TLS 服务。"""
    # 1. 目录必须全新；不接触已有 workspace 或正式服务。
    root.mkdir(parents=True, exist_ok=False)
    with closing(MessageLog(root / "sessions.db")) as log, closing(ArtifactStore(root / "sessions.db")) as metadata:
        store = ChannelAttachmentArtifactStore(workspace=root, metadata_store=metadata)
        content = bytes(range(256)) * 520
        ref = await store.import_bytes(content, kind=AttachmentKind.FILE, filename="旧附件.bin", media_type="application/octet-stream")
        empty = await store.import_bytes(b"", kind=AttachmentKind.FILE, filename=None, media_type=None)
        checks = {kind: lambda part: ContentReferences() for kind in ("text", "history.provenance", "history.transcript", "history.record")}
        checks["artifact_ref"] = lambda part: ContentReferences(artifact_ids=(part.value,))

        def append(sid: str, mid: str, body: Input | Output) -> None:
            log.writer(sid, author="legacy-attribution-unknown", source="legacy-unattributed", body_types=(type(body),), content=checks).append(mid, body)

        transcript = {"schema": "sessions.messages.tool_chain.v0", "raw": json.dumps([
            {"text": "原工具说明", "reasoning_content": "原思考记录", "calls": [{"name": "shell", "arguments": {"command": "pwd"}, "result": "/fixture"}]}
        ]), "completeness": "unknown"}
        append(SESSION, "legacy-question", Input((ContentPart("text", "旧历史修复验证"), ContentPart("history.provenance", {"role": "user"}))))
        append(SESSION, "legacy-answer", Output((ContentPart("text", "原回答完整保留"), ContentPart("history.transcript", transcript)), "complete"))
        append(SESSION, "legacy-file", Output((ContentPart("artifact_ref", ref.artifact_id),), "complete"))
        append(SESSION, "legacy-empty", Output((ContentPart("artifact_ref", empty.artifact_id),), "complete"))
        append(SESSION, "legacy-record", Output((ContentPart("history.record", {"schema": "sessions.turns.v0", "row": {"status": "failed"}}),), "quiet"))
        append(OTHER_SESSION, "other-reference", Input((ContentPart("artifact_ref", ref.artifact_id),)))
        for index in range(tail_count):
            append(TAIL_SESSION, f"tail-{index}", Input((ContentPart("text", f"按需历史 {index}"),
                ContentPart("history.provenance", {"raw": "x" * 8192}))))
        manifest = {"session_id": SESSION, "other_session_id": OTHER_SESSION, "artifact_id": ref.artifact_id,
                    "sha256": hashlib.sha256(content).hexdigest(), "empty_artifact_id": empty.artifact_id, "head_seq": 4,
                    "tail_session_id": TAIL_SESSION, "tail_count": tail_count}
        (root / "fixture.json").write_text(json.dumps(manifest), encoding="utf-8")
        # 2. 使用正式认证、分页、完整性校验和二进制发送路径，不安装回复执行者。
        runtime, keyset = build_mobile_gateway_runtime(build_config(root, "127.0.0.1", port), root, master_keys=EphemeralMasterKeys())
        runtime.channel.bind_messages(log.catalog())
        runtime.channel.bind_channel_attachment_store(store)
        await runtime.channel.start(cast(Any, SimpleNamespace(bus=HistoryBus(), attachment_store=AttachmentStore(root / "drafts"), command_catalog_provider=lambda: ())))
        offer = runtime.admin.create_offer()
        (root / "pairing-offer.json").write_text(json.dumps(offer), encoding="utf-8")
        approval = asyncio.create_task(approve_pairing(runtime, cast(str, offer["pairing_id"])))
        print(f"fixture_ready={root}", flush=True)
        try:
            await build_mobile_gateway_server(runtime, keyset).serve()
        finally:
            approval.cancel()
            await asyncio.gather(approval, return_exceptions=True)
            await runtime.channel.stop()
            runtime.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--port", type=int, default=16339)
    parser.add_argument("--tail-count", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(run(args.root, args.port, args.tail_count))
