"""Expose a narrow read-only catalog for the current runtime scope.

公开合同只含 ServiceKey、DTO Reader 与失败类型；snapshot→catalog 投影归 Core
(`agent.plugins.runtime_catalog`)，插件不得接触 RuntimeSnapshot 或其 lease。
"""

from __future__ import annotations

from collections.abc import Callable

from agent.plugin_composition.model import ServiceKey


RuntimeCatalogReader = Callable[[], dict[str, object]]
RUNTIME_CATALOG = ServiceKey[RuntimeCatalogReader]("core.runtime_catalog.v1")


class RuntimeCatalogUnavailable(RuntimeError):
    """Report a catalog section that cannot be projected yet."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


__all__ = [
    "RUNTIME_CATALOG",
    "RuntimeCatalogReader",
    "RuntimeCatalogUnavailable",
]
