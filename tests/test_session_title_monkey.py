"""Monkey 测试与异常注入：session_title 插件鲁棒性、故障隔离与清洗。"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from typing import Any, AsyncIterator

import pytest

from agent.plugin_composition import CompositionRoot, PluginRuntime
from agent.plugin_composition.plugin_config import PLUGIN_CONFIG, PluginConfig
from agent.plugins.bundles import load_bundle
from plugins.ledger.contract import (
    ContentPart,
    ContentReferences,
    Input,
    SESSION_ADMIN,
)
from plugins.ledger.log import MessageLog, SessionAttributes
from plugins.ledger.services import SessionAdmin
from plugins.models.contract import (
    CHAT_MODELS,
    LLMResponse,
    MODEL_CATALOG,
    ModelCatalogSnapshot,
    ModelChoice,
    ModelRequest,
    ModelUse,
    TransportError,
)
from plugins.session_title import plugin
from plugins.sources.contract import SOURCE_CHANGED_V3, SourceChangedV3


class MonkeyCatalog:
    def snapshot(self) -> ModelCatalogSnapshot:
        return ModelCatalogSnapshot(
            revision=0,
            connections=(),
            models=(),
            role_bindings={},
            default_embedding_model_id=None,
        )

    def validate_chat_selection(self, selection: Any) -> Any:
        return selection


class MonkeyModels:
    """可注入任意响应、异常和延迟的猴子测试模型桩。"""

    def __init__(self) -> None:
        self.calls: list[ModelRequest] = []
        self.behavior: Any = "自动短标题"
        self.delay: float = 0.0

    @asynccontextmanager
    async def independent_execution(self):
        yield self

    async def use(self, use: ModelUse, choice: ModelChoice) -> MonkeyModels:
        return self

    async def complete(self, request: ModelRequest) -> LLMResponse:
        self.calls.append(request)
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        val = self.behavior
        if callable(val):
            val = val(request)
        if isinstance(val, BaseException):
            raise val
        return LLMResponse(content=val, usage=None)


@asynccontextmanager
async def scenario_context() -> AsyncIterator[dict[str, Any]]:
    bundle_path = Path(__file__).resolve().parents[1] / "bundles/base.toml"
    if bundle_path.exists():
        load_bundle(bundle_path)

    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir)
        log = MessageLog(path / "sessions.db")
        root = CompositionRoot("monkey-test")
        models = MonkeyModels()
        admin = SessionAdmin(log)

        async def provide(ctx):
            await ctx.provide(SESSION_ADMIN, admin)
            await ctx.provide(CHAT_MODELS, models)
            await ctx.provide(PLUGIN_CONFIG, PluginConfig(None))
            await ctx.provide(MODEL_CATALOG, MonkeyCatalog())

        await root.mount(provide, name="services")
        fiber = await root.mount(
            plugin.apply,
            name="session_title",
            inject=plugin.inject,
            runtime=PluginRuntime("session_title", "monkey", path, path, path, {}),
        )

        def append(key: str, text: str, *, source: str = "conversation", emit: bool = True):
            log.ensure_session(key, SessionAttributes())
            reader = log.reader(key)
            writer = log.writer(
                key,
                author="user",
                source=source,
                body_types=(Input,),
                content={"text": lambda _: ContentReferences()},
            )
            writer.append(f"{key}:{reader.head() + 1}", Input((ContentPart("text", text),)))
            if emit:
                root.context.emit(SOURCE_CHANGED_V3, SourceChangedV3(reader, source, True))
            return reader

        async def drain():
            await asyncio.sleep(0)
            jobs = [task for task in asyncio.all_tasks() if task.get_name().startswith("session-title:")]
            if jobs:
                await asyncio.gather(*jobs, return_exceptions=True)

        try:
            yield {
                "root": root,
                "fiber": fiber,
                "models": models,
                "log": log,
                "admin": admin,
                "append": append,
                "drain": drain,
            }
        finally:
            await root.dispose()
            log.close()


@pytest.mark.asyncio
async def test_none_content_fallback_and_worker_survival() -> None:
    """1. 验证 response.content 为 None 时不崩，降级首句，且后续会话仍能正常生成标题。"""
    async with scenario_context() as env:
        append = env["append"]
        drain = env["drain"]
        models = env["models"]
        fiber = env["fiber"]

        # 第一次响应返回 None
        models.behavior = None
        r1 = append("s_none", "用户发的第一句话")
        await drain()

        # 应降级为首句
        assert r1.title == "用户发的第一句话"
        assert fiber.error is None

        # 关键点：后续新会话到达，worker 依然活着且能正常生成标题！
        models.behavior = "正常的后续标题"
        r2 = append("s_alive", "另一段完全不同的长文本")
        await drain()

        assert r2.title == "正常的后续标题"


@pytest.mark.asyncio
async def test_title_cleaning_various_formats() -> None:
    """2. 各种怪异前缀、引号及 Markdown 格式清洗。"""
    async with scenario_context() as env:
        append = env["append"]
        drain = env["drain"]
        models = env["models"]

        test_cases = [
            ('"带双引号的标题"', "带双引号的标题"),
            ('“带中文双引号”', "带中文双引号"),
            ("'带单引号'", "带单引号"),
            ("`反引号标题`", "反引号标题"),
            ("标题：冒号前缀测试", "冒号前缀测试"),
            ("标题: 英文冒号测试", "英文冒号测试"),
            ("Title: English Prefix", "English Prefix"),
            ('标题："引用内的标题"', "引用内的标题"),
            ("   前后多余空白与换行  \n  ", "前后多余空白与换行"),
        ]

        for i, (reply, expected) in enumerate(test_cases):
            models.behavior = reply
            reader = append(f"s_clean_{i}", f"测试输入 {i}")
            await drain()
            assert reader.title == expected, f"Failed for reply: {reply}"


@pytest.mark.asyncio
async def test_unexpected_exceptions_remain_visible() -> None:
    """内部模型错误必须进入真实后台任务诊断，不能伪装成成功标题。"""
    for failure in (ZeroDivisionError("模型内部错误"), TypeError("模型类型错误")):
        async with scenario_context() as env:
            worker = next(task for task in asyncio.all_tasks()
                          if task.get_name() == "plugin-task:session-title")
            env["models"].behavior = failure
            reader = env["append"]("broken", "不应写成成功标题")
            with pytest.raises(ExceptionGroup):
                await asyncio.wait_for(asyncio.shield(worker), 3)
            assert reader.title is None
            assert any(item.kind == "task_failure" for item in env["root"].recent_incidents())


@pytest.mark.asyncio
async def test_monkey_stress_concurrency() -> None:
    """4. Monkey 压力测试：并发随机注入 None、异常、超时、正常响应及用户主动改名。"""
    async with scenario_context() as env:
        append = env["append"]
        drain = env["drain"]
        models = env["models"]
        admin = env["admin"]

        rng = random.Random(42)
        session_count = 30
        readers = []

        def random_behavior(req: ModelRequest) -> Any:
            choice = rng.randint(0, 5)
            if choice == 0:
                return None
            elif choice == 1:
                return "   "  # 纯空格
            elif choice == 2:
                return TransportError("模拟网络断开").exception()
            elif choice == 3:
                return TimeoutError("模型请求超时")
            elif choice == 4:
                return f"标题：Monkey生成_{rng.randint(100, 999)}"
            else:
                return f"“优雅标题_{rng.randint(100, 999)}”"

        models.behavior = random_behavior

        for i in range(session_count):
            key = f"monkey_{i}"
            text = f"这是会话 {i} 的第一条随机输入内容_{rng.randint(1000, 9999)}"
            reader = append(key, text)
            readers.append((key, text, reader))

            # 随机模拟用户提前改名
            if rng.random() < 0.2:
                await admin.set_title(key, f"用户手动改名_{i}")

        await drain()

        # 验收每一个会话状态
        for key, text, reader in readers:
            title = reader.title
            assert title is not None, f"会话 {key} 标题不应为空"
            # 长度不能超过 24
            assert len(title) <= 24
            # 不能包含未剥离的包裹引号
            assert not (title.startswith('"') and title.endswith('"'))
            assert not (title.startswith('“') and title.endswith('”'))
            assert not title.startswith("标题：")
