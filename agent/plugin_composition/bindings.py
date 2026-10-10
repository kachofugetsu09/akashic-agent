from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol, TypeVar

from agent.plugin_composition.context import Context
from agent.plugin_composition.model import ServiceKey

_T = TypeVar("_T")


class Bindings(Protocol):
    """固定业务选择与来源证据，并借用当前 provider。"""

    def bind(
        self, service: ServiceKey[Any], metadata: Mapping[str, object], *,
        contributors: tuple[Context, ...] = (),
    ) -> str: ...

    def describe(self, identity: str, service: ServiceKey[Any]) -> Mapping[str, object]: ...

    def open(
        self, identity: str, service: ServiceKey[_T],
    ) -> AbstractAsyncContextManager[tuple[_T, Mapping[str, object]]]: ...


BINDINGS = ServiceKey[Bindings]("core.bindings")
