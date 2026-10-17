"""随发行版提供本地诊断；外部监控和 Web UI 都不是采样前置条件。"""
from __future__ import annotations

from agent.plugin_composition import Context
from plugins.runtime_diagnostics.contract import DIAGNOSTICS
from plugins.ui.contract import UI
from . import dashboard
from .profiling import PROFILES, Profiles
from .records import Records
from .runtime import Monitor
from .stalls import StallWatch

api_version = 3
name = "runtime_diagnostics"
version = "1.0.0"
desc = "运行指标、事件循环停顿现场、插件操作与限时内存剖析，供运维读取"


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
    await ctx.spawn(StallWatch(records).run(), name="diagnostics-stall-watch")

    async def register_http(child: Context) -> None:
        await child.require(UI).register(child, dashboard=lambda: dashboard)

    await ctx.inject((UI, DIAGNOSTICS, PROFILES), register_http, name="diagnostics-http")
