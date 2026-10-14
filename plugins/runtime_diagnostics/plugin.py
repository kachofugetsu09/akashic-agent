"""随发行版提供本地诊断；外部监控和 Web UI 都不是采样前置条件。"""
from __future__ import annotations

from agent.plugin_composition import Context
from plugins.runtime_diagnostics.contract import DIAGNOSTICS
from .profiling import PROFILES, Profiles
from .records import Records
from .runtime import Monitor

api_version = 3
name = "runtime_diagnostics"
version = "1.0.0"
desc = "本地运行曲线、插件操作与限时内存剖析，无需外部监控服务"


async def apply(ctx: Context) -> None:
    """先登记资源关闭责任，再发布只读接口与独立剖析控制。"""
    records = Records(ctx.runtime.plugin_id)
    monitor = Monitor(records)
    profiles = Profiles()
    await ctx.effect(monitor.start, label="diagnostics.records")
    await ctx.effect(lambda: profiles.close, label="diagnostics.profiles")
    await ctx.provide(DIAGNOSTICS, records)
    await ctx.provide(PROFILES, profiles)
    await ctx.spawn(monitor.run(), name="diagnostics-sampling")
