from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import suppress
from dataclasses import asdict

from agent.plugins.snapshot import RuntimeSnapshotStore
from plugins.reply.status import REPLY_STATUS, ReplyActivity


class RuntimeReplyStatus:
    """订阅当前回复的只读状态；客户端连接不占用插件执行 lease。"""

    def __init__(self, store: RuntimeSnapshotStore):
        self._store = store

    async def follow(self, session_id: str) -> AsyncGenerator[dict[str, object], None]:
        """切换 generation 时丢弃旧预览；无回复插件与空闲状态明确区分。"""
        while True:
            # 1. 同步取出窄读取接口，不在 Root 上跨 await 执行业务。
            snapshot = self._store.current
            if snapshot is None or snapshot.composition_root is None:
                raise RuntimeError("回复状态需要已发布的插件 Root")
            read = snapshot.composition_root.context.get(REPLY_STATUS)
            base: dict[str, object] = {
                "version": 2, "session_id": session_id,
                "snapshot_id": snapshot.snapshot_id,
            }
            if read is None:
                yield {**base, "available": False, "items": []}
                _ = await self._store.wait_for_stable_change(snapshot)
                continue

            # 2. 通知只提示重新读取当前状态；旧 generation 的 token 不重放。
            changed = asyncio.create_task(self._store.wait_for_stable_change(snapshot))
            pending: asyncio.Task[tuple[ReplyActivity, ...]] | None = None
            follower = read.follow(session_id)
            try:
                while self._store.current is snapshot:
                    pending = asyncio.create_task(anext(follower))
                    done, _ = await asyncio.wait((pending, changed), return_when=asyncio.FIRST_COMPLETED)
                    if changed in done:
                        _ = changed.result()
                        break
                    try:
                        items = pending.result()
                    except StopAsyncIteration:
                        yield {**base, "available": False, "items": []}
                        _ = await changed
                        break
                    yield {**base, "available": True, "items": [asdict(item) for item in items]}
            finally:
                # 3. 切页、断线和卸载结束正在等的读取，不能留下后台订阅。
                if pending is not None:
                    _ = pending.cancel()
                    with suppress(asyncio.CancelledError, StopAsyncIteration):
                        _ = await pending
                _ = changed.cancel()
                with suppress(asyncio.CancelledError):
                    _ = await changed
                await follower.aclose()
