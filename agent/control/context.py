from __future__ import annotations

from contextvars import ContextVar

running_turn_id: ContextVar[str] = ContextVar("running_turn_id", default="")
_plugin_child_capability_minters: dict[str, object] = {}


def register_plugin_child_capability_minter(owner_turn_id: str, minter: object) -> None:
    current = _plugin_child_capability_minters.get(owner_turn_id)
    if current is not None and current is not minter:
        raise RuntimeError(f"插件 child capability minter 已存在: {owner_turn_id}")
    _plugin_child_capability_minters[owner_turn_id] = minter


def unregister_plugin_child_capability_minter(
    owner_turn_id: str,
    minter: object,
) -> None:
    if _plugin_child_capability_minters.get(owner_turn_id) is minter:
        _ = _plugin_child_capability_minters.pop(owner_turn_id, None)


def mint_plugin_child_capability(owner_turn_id: str) -> str:
    minter = _plugin_child_capability_minters.get(owner_turn_id)
    if not callable(minter):
        return ""
    capability = minter(owner_turn_id)
    if not isinstance(capability, str):
        raise RuntimeError("插件 child capability minter 返回值无效")
    return capability
