"""普通插件贡献配置入口；引导只取得元数据和只读状态。"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from agent.plugin_composition import Context, Effect, ServiceKey


@dataclass(frozen=True)
class Step:
    key: str
    title: str
    group: str
    route: str
    status: Callable[[], Awaitable[dict[str, object]]]
    function_fibers: tuple[int, ...] = ()


class Onboarding(Protocol):
    async def register(self, ctx: Context, step: Step) -> Effect: ...
    async def group(self, ctx: Context, key: str, title: str) -> Effect: ...
    async def catalog(self) -> dict[str, object]: ...
    async def status(self, key: str) -> dict[str, object]: ...


ONBOARDING = ServiceKey[Onboarding]("onboarding.steps.v1")
