from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping


@dataclass(frozen=True)
class MigrationContext:
    config_path: Path
    workspace: Path
    # Each artifact migration receives only its own declared data root.  The
    # mapping is keyed by the immutable bundle owner, so a migration cannot
    # guess another plugin's marketplace directory.
    bundle_data_roots: Mapping[str, Path]


_CURRENT_CONTEXT: ContextVar[MigrationContext | None] = ContextVar(
    "akashic_migration_context",
    default=None,
)


@contextmanager
def bind_migration_context(
    *,
    config_path: Path,
    workspace: Path,
    bundle_data_roots: Mapping[str, Path] | None = None,
) -> Iterator[MigrationContext]:
    """在 Yoyo 调用迁移回调期间暴露当前安装上下文。"""

    context = MigrationContext(
        config_path=config_path,
        workspace=workspace,
        bundle_data_roots=MappingProxyType(dict(bundle_data_roots or {})),
    )
    token = _CURRENT_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_CONTEXT.reset(token)


def current_migration_context() -> MigrationContext:
    context = _CURRENT_CONTEXT.get()
    if context is None:
        raise RuntimeError("migration callback 缺少 Akashic installation context")
    return context
