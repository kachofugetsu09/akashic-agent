"""用真实双文件和 receipt 验证恢复期间的取消与读者隔离。"""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from unittest.mock import patch

from plugins.markdown_memory.plugin import profile_lock, start_store
from plugins.markdown_memory.store import MarkdownProfileStore


async def check(root: Path) -> None:
    """取消不能提前释放两文件锁；恢复成功的 receipt 仍然可查。"""
    store = MarkdownProfileStore(root / "MEMORY.md", root / "SELF.md", root / "writes.db")
    before_self = store.read_self()
    draft: dict[str, object] = {"memory_before": "", "self_before": before_self,
             "memory": "new memory", "self": before_self + "\n- new self\n"}
    store.write_draft("ref", draft, session_key="session", generation=1)
    lock = root / "profile.lock"
    release = threading.Event()
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()
    original = store._apply_document

    def blocked(source_ref, document, path):
        original(source_ref, document, path)
        if document == "memory":
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(5):
                raise RuntimeError("档案恢复阻塞了事件循环")

    async def read():
        async with profile_lock(lock):
            return store.read_memory(), store.read_self()

    with patch.object(store, "_apply_document", blocked):
        writer = asyncio.create_task(start_store(
            store, lock, root / "PENDING.md", root / "snapshot.md", root / "retired.md"))
        reader = None
        try:
            await asyncio.wait_for(entered.wait(), 2)
            writer.cancel()
            reader = asyncio.create_task(read())
            # 一个 loop 回合足以传播取消；物理写仍在受控屏障内。
            checkpoint = loop.create_future()
            loop.call_soon(checkpoint.set_result, None)
            await checkpoint
            assert not writer.done() and not reader.done()
            release.set()
            try:
                await writer
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("取消丢失")
            assert await reader == (draft["memory"], draft["self"])
            assert store.is_applied("ref")
            assert store.read_backup("ref", "memory") == ""
            assert store.read_backup("ref", "self") == before_self
        finally:
            release.set()
            await asyncio.gather(writer, *(() if reader is None else (reader,)), return_exceptions=True)


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-markdown-io-") as directory:
        asyncio.run(check(Path(directory)))
    print("PASS: cancellation drains both profiles before unlocking; receipts and backups preserved")
