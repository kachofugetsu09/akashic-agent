"""真实附件文件的流顺序、完整性与取消排空；不访问网络。"""

import asyncio
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from unittest.mock import patch

import httpcore

from plugins.ledger.base import AttachmentStore
from infra.channels import remote_media


async def response():
    yield b"first"
    yield b"second"


async def persist(store):
    return await remote_media._persist_response(
        httpcore.Response(200, headers=[(b"content-length", b"11")], content=response()),
        "https://example.org/test.bin", store, max_bytes=100,
    )


async def check(root: Path) -> None:
    """暂停真实 fsync 后取消，确认清理不会抢在 worker 之前运行。"""
    store = AttachmentStore(root)
    snapshot = await persist(store)
    assert snapshot.path.read_bytes() == b"firstsecond"
    assert snapshot.sha256 == hashlib.sha256(b"firstsecond").hexdigest()
    assert snapshot.size_bytes == 11
    snapshot.path.unlink()
    release = threading.Event()
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()
    original = remote_media.os.fsync

    def blocked(fd):
        loop.call_soon_threadsafe(entered.set)
        if not release.wait(5):
            raise RuntimeError("附件 fsync 阻塞了事件循环")
        original(fd)

    with patch.object(remote_media.os, "fsync", blocked):
        download = asyncio.create_task(persist(store))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            download.cancel()
            checkpoint = loop.create_future()
            loop.call_soon(checkpoint.set_result, None)
            await checkpoint
            assert not download.done()
            assert tuple(root.iterdir())
            release.set()
            try:
                await download
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("下载取消丢失")
            assert not tuple(root.iterdir())
        finally:
            release.set()
            await asyncio.gather(download, return_exceptions=True)


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-media-io-") as directory:
        asyncio.run(check(Path(directory)))
    print("PASS: ordered bytes/hash; cancellation drains fsync before deleting files")
