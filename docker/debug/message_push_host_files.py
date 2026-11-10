"""真实 UDS、文件与 Ledger 集成验收；只使用一次性目录。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.plugin_composition import CompositionRoot
from plugins.host_execution.bridge.client import HostBridgeShellProcessManager
from plugins.host_execution.bridge.server import HostBridgeService
from plugins.host_execution.bridge.transport import Server
from plugins.host_execution.contract import FILES, FileError, FILE_IMPORT_BYTES
from plugins.host_execution.files import Files
from plugins.ledger.artifact_services import ArtifactImport
from plugins.ledger.artifact_store import ArtifactStore
from plugins.ledger.attachment_import import ChannelOutboundAttachmentImporter
from plugins.ledger.attachments import ChannelAttachmentArtifactStore
from plugins.ledger.contract import ARTIFACT_IMPORT, decode_body
from plugins.message_push.tool import MessagePush


async def run(folder: Path) -> None:
    """经真实 Bridge 准备附件，核对大文件、错误、内容保全与来源漂移。"""
    socket = folder / "bridge.sock"
    commit, digest, token = "a" * 40, "b" * 64, "isolated-file-scenario"
    service = HostBridgeService(token, 30, folder / "bridge-output",
        release_commit=commit, toolchain_digest=digest, runtime_checkout=ROOT,
        bridge_python=Path(sys.executable))
    server = Server(service)
    await server.start(socket)
    control = HostBridgeShellProcessManager(socket, "files", token, commit, digest)
    await control.claim_boot()
    environment = {
        "AKASHIC_EXECUTION_MODE": "host-bridge", "AKASHIC_HOST_BRIDGE_SOCKET": str(socket),
        "AKASHIC_BOOT_ID": "files", "AKASHIC_HOST_BRIDGE_TOKEN": token,
        "AKASHIC_RUNTIME_COMMIT": commit, "AKASHIC_HOST_TOOLCHAIN_DIGEST": digest,
    }
    previous = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    store = ArtifactStore(folder / "artifacts.db")
    artifacts = ChannelAttachmentArtifactStore(workspace=folder, metadata_store=store)
    root = CompositionRoot("message-push-files")
    contexts = []

    async def provide(ctx):
        await ctx.provide(FILES, Files())
        await ctx.provide(ARTIFACT_IMPORT, ArtifactImport(ChannelOutboundAttachmentImporter(artifacts).import_source))

    async def consume(ctx):
        contexts.append(ctx)

    try:
        await root.mount(provide, name="services")
        await root.mount(consume, name="consumer", inject=(FILES, ARTIFACT_IMPORT))
        ctx = contexts[0]
        async with ctx.runtime_scope():
            push = MessagePush(ctx, {"test": "test-binding"})
            source = folder / "host-file.bin"
            data = bytes(range(256)) * (FILE_IMPORT_BYTES // 256)
            source.write_bytes(data)
            arguments = {"target_channel": "test", "target_chat_id": "room", "file": str(source)}
            prepared = await push.prepare(arguments)
            assert not isinstance(prepared, str), prepared
            body = decode_body(prepared["body"])
            ref = artifacts.resolve_refs((body.parts[0].value,))[0]
            assert ref.filename == source.name and ref.size_bytes == len(data)
            assert ref.sha256 == hashlib.sha256(data).hexdigest()
            # 来源后续改变不得反写已经发布的附件。
            source.write_bytes(b"changed")
            lease = await artifacts.acquire(ref)
            try:
                assert await lease.read_bytes(max_bytes=FILE_IMPORT_BYTES) == data
            finally:
                await lease.aclose()
            # 跨块读取必须拒绝混用两个文件版本。
            async with Files().bytes() as reader:
                first = await reader.read(str(source))
                assert not isinstance(first, FileError)
                source.write_bytes(b"new source")
                changed = await reader.read(str(source), offset=1, version=first.version)
                assert isinstance(changed, FileError) and "变化" in changed.text
            fifo = folder / "pipe"
            os.mkfifo(fifo)
            large = folder / "too-large"
            with large.open("wb") as output:
                output.truncate(FILE_IMPORT_BYTES + 1)
            for path in (folder / "missing/absent", folder, fifo, large):
                failed = await asyncio.wait_for(push.prepare({**arguments, "file": str(path)}), 5)
                assert isinstance(failed, str) and "消息尚未发送" in failed, failed
            remote = await push.prepare({**arguments, "file": "https://127.0.0.1/blocked"})
            assert isinstance(remote, str) and "非公网" in remote, remote
            # 一个输入失败之后，下一次调用仍可正常准备。
            again = await push.prepare(arguments)
            assert not isinstance(again, str), again
            source.write_bytes(b"")
            empty = await push.prepare(arguments)
            assert not isinstance(empty, str), empty
            empty_id = decode_body(empty["body"]).parts[0].value
            assert artifacts.resolve_refs((empty_id,))[0].size_bytes == 0
            assert artifacts.resolve_refs((ref.artifact_id,)) == (ref,)
            print(json.dumps({"result": "passed", "bridge_max_bytes": len(data),
                "immutable_content": True, "source_change_rejected": True,
                "invalid_sources": 5, "next_call_succeeded": True}))
    finally:
        await root.dispose()
        store.close()
        await control.close_transport()
        await service.shutdown()
        await server.stop()
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    with TemporaryDirectory(prefix="message-push-files-") as temporary:
        asyncio.run(run(Path(temporary)))
