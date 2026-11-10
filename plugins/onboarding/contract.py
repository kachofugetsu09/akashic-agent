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


# 能力卡上的一段示例：说话人与一行文字，纯文本。
@dataclass(frozen=True)
class PreviewLine:
    speaker: str
    text: str


# 分组面向用户的说明：引导按分组展示能力卡，插件名只作来源小字。
@dataclass(frozen=True)
class Ability:
    pitch: str
    benefit: str
    preview: tuple[PreviewLine, ...] = ()
    required: bool = False


class Onboarding(Protocol):
    async def register(self, ctx: Context, step: Step) -> Effect: ...
    async def group(self, ctx: Context, key: str, title: str, ability: Ability | None = None) -> Effect: ...
    async def catalog(self) -> dict[str, object]: ...
    async def status(self, key: str) -> dict[str, object]: ...


ONBOARDING = ServiceKey[Onboarding]("onboarding.steps.v1")
