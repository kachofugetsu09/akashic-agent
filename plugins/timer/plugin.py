"""Timer provider 拥有全部一次等待的生命周期。"""
from agent.plugin_composition import Context
from plugins.timer.contract import TIMERS

from .timer import AsyncioOneShotTimer

api_version = 3
name = "timer"
version = "1.0.0"
desc = "单次截止时间等待，不拥有调度或来源语义"


async def apply(ctx: Context) -> None:
    timer = AsyncioOneShotTimer()
    await ctx.effect(lambda: timer.close, label="timers")
    await ctx.provide(TIMERS, timer)
