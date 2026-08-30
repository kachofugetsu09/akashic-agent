from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar

running_turn_id: ContextVar[str] = ContextVar("running_turn_id", default="")
_plugin_child_capability_minters: dict[str, Callable[[str], str]] = {}


def register_plugin_child_capability_minter(
    owner_turn_id: str,
    minter: Callable[[str], str],
) -> None:
    current = _plugin_child_capability_minters.get(owner_turn_id)
    if current is not None and current != minter:
        raise RuntimeError(f"插件 child capability minter 已存在: {owner_turn_id}")
    _plugin_child_capability_minters[owner_turn_id] = minter


def unregister_plugin_child_capability_minter(
    owner_turn_id: str,
    minter: Callable[[str], str],
) -> None:
    if _plugin_child_capability_minters.get(owner_turn_id) == minter:
        del _plugin_child_capability_minters[owner_turn_id]


def mint_plugin_child_capability(owner_turn_id: str) -> str | None:
    minter = _plugin_child_capability_minters.get(owner_turn_id)
    return None if minter is None else minter(owner_turn_id)
