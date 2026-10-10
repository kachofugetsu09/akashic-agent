"""临时 SQLite 与真实插件装配；本地模型边界控制成功、失败和取消时序。"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
from pathlib import Path
import sys
import threading
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.plugin_composition import CHAT_MODELS, CompositionRoot, PluginRuntime
from agent.plugin_composition.messages import SESSION_ADMIN, SessionAdmin
from agent.plugin_composition.models import LLMResponse, TransportError
from agent.plugin_contracts import ContentPart, ContentReferences, Input
from plugins.sources.contract import (
    SOURCE_CHANGED_V3,
    SourceChangedV3,
)
from plugins.session_title import plugin
from scripts.install_plugin_distribution import _load_profile
from session.log import MessageLog, SessionAttributes


class LocalModels:
    """只替代外部模型，不替代注册、调度、管理接口或数据库。"""

    def __init__(self):
        self.calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.drained = asyncio.Event()
        self.reply = '"自动短标题"'
        self.failure = None
        self.block = False

    @asynccontextmanager
    async def independent_execution(self):
        yield self

    def chat(self, role):
        assert role == "fast"
        return self

    async def complete(self, request):
        self.calls.append(request)
        self.started.set()
        if self.block:
            try:
                await self.release.wait()
            finally:
                self.cancelled.set()
                await self.drained.wait()
        if self.failure is not None:
            raise self.failure
        return LLMResponse(content=self.reply, usage=None)


class FileQueue(ThreadPoolExecutor):
    """把写入排队与用户提交确定性地交错，不更换生产存储函数。"""

    def __init__(self, loop):
        super().__init__(max_workers=1)
        self.loop, self.queued = loop, asyncio.Event()
        self.watch = False

    def submit(self, fn, /, *args, **kwargs):
        future = super().submit(fn, *args, **kwargs)
        if self.watch:
            self.loop.call_soon_threadsafe(self.queued.set)
        return future


async def drain():
    """给通知队列一次调度机会，然后显式等待真实子任务，不靠定时睡眠。"""
    await asyncio.sleep(0)
    jobs = [task for task in asyncio.all_tasks() if task.get_name().startswith("session-title:")]
    async with asyncio.timeout(20):
        await asyncio.gather(*jobs)


async def run(path):
    """覆盖正常命名、当前值竞争和卸载排空，同时核对既有消息及目录排序。"""
    log = MessageLog(path / "sessions.db")
    root, models = CompositionRoot("session-title-scenario"), LocalModels()
    admin = SessionAdmin(log)
    messages = []
    updates = {}

    async def provide(ctx):
        await ctx.provide(SESSION_ADMIN, admin)
        await ctx.provide(CHAT_MODELS, models)

    await root.mount(provide, name="services")
    fiber = await root.mount(plugin.apply, name="session_title", inject=plugin.inject,
        runtime=PluginRuntime("session_title", "scenario", path, path, path, {}))
    assert fiber.error is None, fiber.error

    def append(key, text, *, source="conversation", emit=True):
        log.ensure_session(key, SessionAttributes())
        reader = log.reader(key)
        writer = log.writer(key, author="user", source=source, body_types=(Input,),
            content={"text": lambda part: ContentReferences()})
        message = writer.append(f"{key}:{reader.head() + 1}", Input((ContentPart("text", text),)))
        messages.append(message)
        with log._read():
            updates[key] = log._connection.execute("SELECT updated_at FROM sessions WHERE key=?", (key,)).fetchone()[0]
        if emit:
            root.context.emit(SOURCE_CHANGED_V3, SourceChangedV3(reader, source, True))
        return reader

    try:
        # 1. 首条触发且重复通知合并；输入输出有限，正文与时间不变。
        first = append("normal", "用户问题" * 1000)
        root.context.emit(SOURCE_CHANGED_V3, SourceChangedV3(first, "conversation", True))
        await drain()
        assert first.title == "自动短标题" and len(models.calls) == 1
        assert len(models.calls[0].messages[-1]["content"]) <= 2000
        await admin.set_title("normal", None)
        append("normal", "第二个问题")
        old = append("old", "旧首条", emit=False)
        append("old", "旧会话新消息")
        append("excluded", "内部调用", source="programmatic")
        await drain()
        assert first.title is None and old.title is None and len(models.calls) == 1

        # 2. 已有标题跳过模型；模型已开始时手动改名或删除仍胜出。
        named = append("named", "命名前已改名", emit=False)
        await admin.set_title("named", "用户标题")
        root.context.emit(SOURCE_CHANGED_V3, SourceChangedV3(named, "conversation", True))
        await drain()
        assert named.title == "用户标题" and len(models.calls) == 1
        for key, action in (("rename", "rename"), ("deleted", "delete")):
            models.block = True
            models.started.clear()
            models.release.clear()
            models.drained.set()
            reader = append(key, "等待期间发生管理操作")
            await asyncio.wait_for(models.started.wait(), 3)
            if action == "rename":
                await admin.set_title(key, "手动优先")
            else:
                await admin.set_deleted(key, deleted=True)
            models.release.set()
            await drain()
            assert reader.title == ("手动优先" if action == "rename" else None)
        models.block = False

        # 3. 预期故障和空输出使用有界首句；不把内部错误伪装成功。
        models.failure = TransportError("本地模型连接失败").exception()
        fallback = append("fallback", "   首句  " * 40)
        await drain()
        assert fallback.title == (" ".join(("   首句  " * 40).split())[:24]).rstrip()
        models.failure = None
        models.reply = '"   "'
        empty = append("empty", "空响应使用这句")
        await drain()
        assert empty.title == "空响应使用这句"
        blank = append("blank", "   ")
        await drain()
        assert blank.title is None

        # 4. 自动写入已排队之后，另一连接提交手动标题，条件 UPDATE 必须放弃。
        loop = asyncio.get_running_loop()
        pool = FileQueue(loop)
        loop.set_default_executor(pool)
        models.block = True
        models.started.clear()
        models.release.clear()
        late = append("late-rename", "已经完成最后一次读取")
        await asyncio.wait_for(models.started.wait(), 3)
        release_io = threading.Event()
        blocker = loop.run_in_executor(None, release_io.wait)
        pool.watch = True
        models.release.set()
        try:
            await asyncio.wait_for(pool.queued.wait(), 3)
            other = MessageLog(path / "sessions.db")
            try:
                other.set_session_title("late-rename", "排队期间用户改名")
            finally:
                other.close()
        finally:
            release_io.set()
            pool.watch = False
        await blocker
        await drain()
        assert late.title == "排队期间用户改名"
        models.block = False

        # 5. 两个独立连接竞争只允许一个空值写入，不需要生成锁。
        raced = append("cas", "并发候选", emit=False)
        other = MessageLog(path / "sessions.db")
        try:
            results = await asyncio.gather(admin.set_title_if_unset("cas", "候选甲"),
                SessionAdmin(other).set_title_if_unset("cas", "候选乙"))
            assert sum(results) == 1 and raced.title in ("候选甲", "候选乙")
        finally:
            other.close()

        # 6. 卸载必须等模型取消清理完成，之后不得留下写入任务。
        models.block = True
        models.started.clear()
        models.release.clear()
        models.cancelled.clear()
        models.drained.clear()
        stopped = append("stop", "卸载期间的输入")
        await asyncio.wait_for(models.started.wait(), 3)
        closing = asyncio.create_task(fiber.dispose())
        await asyncio.wait_for(models.cancelled.wait(), 3)
        assert not closing.done()
        models.drained.set()
        await asyncio.wait_for(closing, 3)
        assert stopped.title is None
        assert not any(t.get_name().startswith("session-title:") for t in asyncio.all_tasks())
        for message in messages:
            assert log.reader(message.session_id).get(message.message_id) == message
        with log._read():
            after = dict(log._connection.execute("SELECT key,updated_at FROM sessions"))
        assert after == updates
        print(json.dumps({"status": "passed", "model_calls": len(models.calls),
            "messages_preserved": len(messages), "updated_at_preserved": True,
            "cas_winners": sum(results), "unload_drained": True}))
    finally:
        models.release.set()
        models.drained.set()
        await root.dispose()
        log.close()


if __name__ == "__main__":
    _load_profile(Path(__file__).resolve().parents[2] / "docker/host-runtime/profiles/default.json")
    with TemporaryDirectory() as temporary:
        asyncio.run(run(Path(temporary)))
