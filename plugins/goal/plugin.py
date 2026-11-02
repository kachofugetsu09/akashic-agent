"""目标插件：持久目标、/goal 命令、模型工具与同来源自动续跑。"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager

from agent.plugin_composition import (
    RUNTIME_STARTED,
    RUNTIME_STOPPING,
    Context,
    stop_spawned,
)
from plugins.commands.contract import COMMANDS, CommandDefinition
from plugins.content.contract import CONTENT
from plugins.ledger.contract import (
    MESSAGE_CATALOG,
    MESSAGE_WRITERS,
    OWNER_STATE,
)
from plugins.models.contract import MODEL_CALL_STATS
from plugins.tools.contract import TOOLS, ProviderBoundTool

from .command import GoalCommand
from .contract import GOAL_STATUS
from .runtime import GoalRuntime
from .store import GoalStore
from .tools import (
    CREATE_GOAL_TOOL,
    GET_GOAL_TOOL,
    UPDATE_GOAL_TOOL,
    CreateGoal,
    GetGoal,
    UpdateGoal,
)

api_version = 3
name = "goal"
version = "1.0.0"
desc = "持久目标：/goal 设置后自动续跑，直到完成、阻塞或暂停"

inject = (
    COMMANDS,
    CONTENT,
    MESSAGE_CATALOG,
    MESSAGE_WRITERS,
    MODEL_CALL_STATS,
    OWNER_STATE,
    TOOLS,
)

_GET_SCHEMA: Mapping[str, object] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}
_CREATE_SCHEMA: Mapping[str, object] = {
    "type": "object",
    "properties": {
        "objective": {
            "type": "string",
            "minLength": 1,
            "maxLength": 4000,
            "description": "要开始追求的具体目标。只有用户或系统明确要求时才创建目标。",
        },
        "token_budget": {
            "type": "integer",
            "minimum": 1,
            "description": "可选的正整数 token 预算；只在用户明确给出预算时填写。",
        },
    },
    "required": ["objective"],
    "additionalProperties": False,
}
_UPDATE_SCHEMA: Mapping[str, object] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["complete", "blocked", "paused"],
            "description": (
                "complete 只在目标真正完成且逐项验证通过后使用；"
                "blocked 只在同一阻塞连续至少 3 个目标轮次后使用；"
                "paused 只在用户明确要求暂停时使用。"
            ),
        },
    },
    "required": ["status"],
    "additionalProperties": False,
}


async def apply(ctx: Context) -> None:
    store = GoalStore(ctx)
    _ = await ctx.provide(GOAL_STATUS, store)

    _ = await ctx.require(TOOLS).declare_group(ctx, description=desc)

    @asynccontextmanager
    async def open_get(_state: Mapping[str, object]) -> AsyncGenerator[ProviderBoundTool, None]:
        yield GetGoal(store)

    @asynccontextmanager
    async def open_create(_state: Mapping[str, object]) -> AsyncGenerator[ProviderBoundTool, None]:
        yield CreateGoal(store)

    @asynccontextmanager
    async def open_update(_state: Mapping[str, object]) -> AsyncGenerator[ProviderBoundTool, None]:
        yield UpdateGoal(store)

    _ = await ctx.require(TOOLS).register(
        ctx,
        name=GET_GOAL_TOOL,
        description=(
            "读取当前会话的目标：状态、objective、token 预算与用量、续跑次数、最近原因。"
            "目标未完成时不要假装它不存在。"
        ),
        parameters=_GET_SCHEMA,
        open=open_get,
        idempotent=True,
        parallel=True,
    )
    _ = await ctx.require(TOOLS).register(
        ctx,
        name=CREATE_GOAL_TOOL,
        description=(
            "创建一个持久目标，让 Akashic 自动续跑直到完成。"
            "只有用户或系统明确要求时才创建；普通任务不要推断成目标。"
            "当前目标未完成时会失败；只有 complete 的目标可以被替换。"
        ),
        parameters=_CREATE_SCHEMA,
        open=open_create,
        idempotent=True,
    )
    _ = await ctx.require(TOOLS).register(
        ctx,
        name=UPDATE_GOAL_TOOL,
        description=(
            "更新当前目标状态。只接受 complete / blocked / paused。"
            "complete 需要目标真正完成且逐项验证通过；"
            "blocked 需要同一阻塞连续至少 3 个目标轮次；"
            "paused 只在用户明确要求暂停时使用。"
        ),
        parameters=_UPDATE_SCHEMA,
        open=open_update,
        idempotent=True,
    )

    command = GoalCommand(store)
    await ctx.require(COMMANDS).register(ctx, CommandDefinition(
        name="goal",
        description="设置持久目标并自动续跑，直到完成、阻塞或暂停",
        handler=command,
        recover=command,
        input_hint="[<objective> | pause | resume | clear | budget <n> | budget clear]",
    ))

    follower = None
    worker = None

    async def start(_event: object) -> None:
        nonlocal follower, worker
        runtime = GoalRuntime(
            store, ctx.require(MESSAGE_CATALOG), ctx.require(MODEL_CALL_STATS),
        )
        follower = await ctx.spawn(runtime.follow(), name="goal-follow")
        worker = await ctx.spawn(runtime.run(), name="goal-runtime")

    async def stop(_event: object) -> None:
        try:
            await stop_spawned(worker)
        finally:
            await stop_spawned(follower)

    _ = await ctx.on(RUNTIME_STARTED, start)
    _ = await ctx.on(RUNTIME_STOPPING, stop)
