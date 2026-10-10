"""会话开始时自动推导并生成可辨识标题；用户覆盖优先。

遵循 Decision 0087（docs/decisions/0087-session-title-override.md）：
- 标题写入 sessions.title，通过 SESSION_ADMIN.set_title 管理。
- 仅在会话标题尚未设置（None）时触发自动生成。
- 若已有标题（用户显式设置或先前已生成），则跳过并取消 pending，防止覆盖用户输入。
- 若模型调用失败或不可用，回退到首条有效用户消息的截断文本，确保可靠降级。
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition import (
    CHAT_MODELS,
    RUNTIME_STARTED,
    RUNTIME_STOPPING,
    Context,
)
from agent.plugin_composition.messages import (
    MESSAGE_CATALOG,
    SESSION_ADMIN,
    MessageReader,
)
from agent.plugin_composition.models import (
    BoundChatModel,
    ContextLengthError,
    ModelRequest,
    ModelTimeoutError,
    RateLimitError,
    TransportError,
)
from agent.plugin_contracts import ContentPart, Input, Message
from agent.plugin_contracts.sources import (
    SOURCE_CHANGED_V3 as SOURCE_CHANGED,
    SourceChangedV3 as SourceChanged,
)

logger = logging.getLogger("plugins.session_title")

api_version = 3
name = "session_title"
version = "1.0.0"
desc = "新会话自动生成简明标题，用户手动重命名优先"
inject = (MESSAGE_CATALOG, SESSION_ADMIN, CHAT_MODELS)

MAX_TITLE_CHARS = 36
FALLBACK_CHARS = 24


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sources: tuple[str, ...] = Field(
        default=("conversation",),
        min_length=1,
        description="监听并为其生成标题的消息来源",
    )
    max_title_chars: int = Field(
        default=MAX_TITLE_CHARS,
        gt=0,
        le=200,
        description="自动生成标题的最大字符数",
    )


TITLE_SYSTEM_PROMPT = """你是一个会话标题提炼专家。
请根据用户的初始消息，生成一个极其简明、精炼的会话标题。
规则要求：
1. 语言与用户消息一致（中文优先使用中文，英文使用英文）。
2. 只输出标题本身，不要加任何标点、前缀、引号、解释或 Markdown 格式。
3. 严格控制在 4 到 15 个字以内，最长不得超过 24 个字。
4. 概括会话核心主题或意图，禁止废话（例如禁止“关于...的讨论”）。
"""


def extract_user_text(message: Message) -> str:
    """提取 Input 消息中的纯文本内容。"""
    body = message.body
    if not isinstance(body, Input):
        return ""
    texts: list[str] = []
    for part in body.parts:
        if isinstance(part, ContentPart) and part.kind == "text":
            if isinstance(part.value, str):
                text = part.value.strip()
                if text:
                    texts.append(text)
    return "\n".join(texts).strip()


def derive_fallback_title(text: str, max_chars: int = FALLBACK_CHARS) -> str:
    """当模型不可用或生成失败时的纯文本截断回退标题。"""
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return "新会话"
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


async def generate_title_with_model(
    model: BoundChatModel,
    user_prompt: str,
    max_chars: int,
) -> str | None:
    """调用辅助模型生成标题，失败返回 None。"""
    request = ModelRequest(
        messages=(
            {"role": "system", "content": TITLE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ),
        max_output_tokens=64,
        disable_reasoning=True,
    )
    try:
        response = await model.complete(request)
        title = response.content.strip()
        # 去除可能包裹的引号或反引号
        title = title.strip('"\'`“”‘’«»')
        title = " ".join(title.split()).strip()
        if not title:
            return None
        if len(title) > max_chars:
            title = title[:max_chars].rstrip()
        return title
    except (
        ContextLengthError,
        ModelTimeoutError,
        RateLimitError,
        TransportError,
        Exception,
    ) as error:
        logger.warning("模型自动生成会话标题失败，将使用回退标题: %s", error)
        return None


async def apply(ctx: Context) -> None:
    config = Config.model_validate(ctx.config)
    session_admin = ctx.require(SESSION_ADMIN)
    catalog = ctx.require(MESSAGE_CATALOG)
    chat_models = ctx.require(CHAT_MODELS)

    # 记录正在生成的 session，避免重复排队
    pending_tasks: dict[str, asyncio.Task[None]] = {}

    async def generate_for_session(session_id: str, prompt_text: str) -> None:
        try:
            # 1. 再次确认标题是否仍未设置（可能在排队期间已被用户改名）
            reader = catalog.reader(session_id)
            if reader.title is not None:
                return

            generated: str | None = None
            try:
                async with chat_models.independent_execution() as execution:
                    model = execution.chat("default")
                    generated = await generate_title_with_model(
                        model, prompt_text, config.max_title_chars
                    )
            except Exception as error:
                logger.warning("无法取得模型 execution，准备使用回退标题: %s", error)

            final_title = generated or derive_fallback_title(
                prompt_text, max_chars=min(config.max_title_chars, FALLBACK_CHARS)
            )

            # 2. 最终写入前再次核验 reader.title
            if reader.title is not None:
                return

            _ = await session_admin.set_title(session_id, final_title)
            logger.info("已为会话 %s 自动设置标题: %s", session_id, final_title)
        except Exception as error:
            logger.warning("为会话 %s 自动设置标题异常终止: %s", session_id, error)
        finally:
            pending_tasks.pop(session_id, None)

    async def on_source_changed(event: SourceChanged) -> None:
        # 只处理配置关注的来源
        if event.source not in config.sources:
            return

        reader = event.reader
        session_id = reader.session_id

        # 若会话已经显式设置过标题，或者正在生成，直接跳过
        if reader.title is not None:
            # 若之前有 pending 的任务，取消之（用户显式重命名优先）
            task = pending_tasks.pop(session_id, None)
            if task is not None and not task.done():
                task.cancel()
            return

        if session_id in pending_tasks:
            return

        # 检查是否为新会话的首条用户消息
        messages = reader.read(limit=10, source=event.source)
        user_messages = [m for m in messages if isinstance(m.body, Input)]
        if not user_messages:
            return

        first_user_text = extract_user_text(user_messages[0])
        if not first_user_text:
            return

        # 仅针对第一条有效输入触发标题生成
        # 异步后台执行，不阻塞当前消息循环
        task = asyncio.create_task(
            generate_for_session(session_id, first_user_text),
            name=f"session_title_{session_id}",
        )
        pending_tasks[session_id] = task

    def cancel_all_pending() -> None:
        for task in pending_tasks.values():
            if not task.done():
                task.cancel()
        pending_tasks.clear()

    async def start(_event: object) -> None:
        logger.info("session_title 插件已启动")

    async def stop(_event: object) -> None:
        cancel_all_pending()

    _ = await ctx.on(SOURCE_CHANGED, on_source_changed)
    _ = await ctx.on(RUNTIME_STARTED, start)
    _ = await ctx.on(RUNTIME_STOPPING, stop)
