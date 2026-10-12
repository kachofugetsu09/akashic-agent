"""用真实档案文件和操作 receipt 验证恢复期间的取消与读者隔离。"""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from unittest.mock import patch

from plugins.markdown_memory import store as store_module
from plugins.markdown_memory.documents import MEMORY_TEMPLATE
from plugins.markdown_memory.plugin import profile_lock, start_store
from plugins.markdown_memory.store import MarkdownProfileStore


async def check(root: Path) -> None:
    """取消不能提前释放文件锁；恢复成功的 receipt 与首次 before-image 仍然可查。"""
    store = MarkdownProfileStore(root / "memory", root / "writes.db")
    before_self = store.read("SELF.md")
    draft: dict[str, object] = {"version": 3, "scope": None, "ops": [
        {"op": "add", "doc": "MEMORY.md", "section": "## 用户偏好", "line": "- new memory", "message_ids": ["m"]},
        {"op": "add", "doc": "SELF.md", "section": "## 人格与形象", "line": "- new self", "message_ids": ["m"]},
    ]}
    store.write_draft("ref", draft, session_key="session", generation=1)
    lock = root / "profile.lock"
    release = threading.Event()
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()
    original = store_module.atomic_write_text

    def blocked(path, text, **kwargs):
        original(path, text, **kwargs)
        if path.name == "MEMORY.md":
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(5):
                raise RuntimeError("档案恢复阻塞了事件循环")

    async def read():
        async with profile_lock(lock):
            return store.read("MEMORY.md"), store.read("SELF.md")

    with patch.object(store_module, "atomic_write_text", blocked):
        writer = asyncio.create_task(start_store(store, lock))
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
            profile, self_profile = await reader
            assert "## 用户偏好\n- new memory" in profile and "- new self" in self_profile
            assert store.is_applied("ref")
            backup = store._read_receipt("ref", "markdown_before_v1")
            assert backup == {"MEMORY.md": MEMORY_TEMPLATE, "SELF.md": before_self}
            # 重放已应用操作保持幂等，文件不再变化。
            store.apply("ref")
            assert (store.read("MEMORY.md"), store.read("SELF.md")) == (profile, self_profile)
        finally:
            release.set()
            await asyncio.gather(writer, *(() if reader is None else (reader,)), return_exceptions=True)


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-markdown-io-") as directory:
        asyncio.run(check(Path(directory)))
    print("PASS: cancellation drains profile writes before unlocking; receipts and before-image preserved")
