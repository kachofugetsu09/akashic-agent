"""首条用户输入生成短标题；后台生成，空标题条件写入。"""
from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition import Context
from plugins.models.contract import CHAT_MODELS
from plugins.ledger.contract import SESSION_ADMIN, MessageReader
from plugins.models.contract import ModelRequest
from plugins.models.contract import ModelError
from plugins.ledger.contract import Input
from plugins.sources.contract import (
    SOURCE_CHANGED_V3,
    SourceChangedV3,
)

logger = logging.getLogger(__name__)
api_version = 3
name = "session_title"
version = "1.0.0"
desc = "首条用户消息自动生成短标题，保留用户改名"
inject = (SESSION_ADMIN, CHAT_MODELS)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sources: tuple[str, ...] = Field(default=("conversation",), min_length=1)
    max_title_chars: int = Field(default=24, gt=0, le=200)


TITLE_PROMPT = (
    "根据用户的第一条消息拟一个简短会话标题，使用用户的语言。"
    "只输出标题，不要引号、前缀、解释或 Markdown。中文以 4 到 15 字为宜。"
)


def _first_text(reader: MessageReader, source: str) -> str:
    """只读取首条正文；已有标题或已删除的会话不调用模型。"""
    if reader.title is not None or reader.deleted:
        return ""
    messages = reader.read(through_seq=0, limit=1, source=source)
    if not messages or not isinstance(messages[0].body, Input):
        return ""
    return " ".join(
        part.value.strip() for part in messages[0].body.parts
        if part.kind == "text" and isinstance(part.value, str)
    ).strip()


async def apply(ctx: Context) -> None:
    """同步通知只筛选首条输入，Fiber 拥有并排空全部后台生成。"""
    config = Config.model_validate(ctx.config)
    admin = ctx.require(SESSION_ADMIN)
    models = ctx.require(CHAT_MODELS)
    queue: asyncio.Queue[tuple[MessageReader, str]] = asyncio.Queue()
    pending: set[str] = set()

    async def generate(reader: MessageReader, source: str) -> None:
        """模型预期失败时截断首句；写入竞争只放弃，不重试。"""
        try:
            # 1. 消息不可变，异步读取固定首条；后续输入不改变命名材料。
            text = await reader.read_async(lambda current: _first_text(current, source))
            if not text:
                return
            title = ""
            try:
                # 2. 小任务使用 fast 角色和局部预算，不占用会话的模型执行。
                async with asyncio.timeout(15), models.independent_execution() as execution:
                    response = await execution.chat("fast").complete(ModelRequest(
                        messages=(
                            {"role": "system", "content": TITLE_PROMPT},
                            {"role": "user", "content": text[:2000]},
                        ),
                        max_output_tokens=64,
                        disable_reasoning=True,
                    ))
                    title = " ".join(response.content.strip().strip('"\'`“”‘’').split())
            except (RuntimeError, TimeoutError) as error:
                if not (ModelError.matches(error) or isinstance(error, TimeoutError)):
                    raise
                logger.warning("会话 %s 标题生成失败，使用首句: %s", reader.session_id, error)
            title = " ".join((title or text).split())[:config.max_title_chars].rstrip()
            # 3. 不锁住生成过程；存储内一次条件更新保护已经提交的手动改名。
            await admin.set_title_if_unset(reader.session_id, title)
        finally:
            pending.remove(reader.session_id)

    async def run() -> None:
        """结构化管理并行小任务，卸载时取消并等待所有子任务完成。"""
        async with asyncio.TaskGroup() as tasks:
            while True:
                reader, source = await queue.get()
                tasks.create_task(generate(reader, source), name=f"session-title:{reader.session_id}")

    def changed(event: SourceChangedV3) -> None:
        if not event.pending or event.source not in config.sources:
            return
        reader = event.reader
        # 序号从零开始且不复用；普通消息只做索引头查询，不重读历史或标题。
        if reader.session_id in pending or reader.head() != 0:
            return
        pending.add(reader.session_id)
        queue.put_nowait((reader, event.source))

    await ctx.spawn(run(), name="session-title")
    await ctx.on(SOURCE_CHANGED_V3, changed)
