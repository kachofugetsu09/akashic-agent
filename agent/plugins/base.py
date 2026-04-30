from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class Plugin(ABC):
    name: str | None = None
    version: str | None = None
    desc: str | None = None
    author: str | None = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        from agent.plugins.registry import plugin_registry
        plugin_registry.register_class(cls)

    async def initialize(self) -> None: ...
    async def terminate(self) -> None: ...
