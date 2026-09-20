from __future__ import annotations

from collections.abc import Callable
from typing import cast

from agent.plugin_composition import ServiceKey
from agent.plugins.snapshot import RuntimeSnapshotStore, lease_runtime_snapshot
from infra.channels.message_view import MessageDisplayProviders, PartDisplayProvider, message_rows
from session.log import MessagePage


class RuntimeMessageDisplay:
    """每页解析并调用同代投影；连接和下载票据不持有插件实现。"""

    def __init__(self, store: RuntimeSnapshotStore):
        self._store = store

    async def __call__(self, page: MessagePage, *, display_only: bool) -> list[dict[str, object]]:
        """插件以内容 kind 声明展示；重复 key 沿用组合层冲突语义。"""
        async with lease_runtime_snapshot(self._store) as snapshot:
            root = snapshot.composition_root
            if root is None:
                raise RuntimeError("消息展示需要已发布的插件 Root")
            providers = root.provided_services()
            renderers = {
                key.name.removeprefix("message.display:"): cast(PartDisplayProvider, value)
                for key, value in providers.items()
                if key.name.startswith("message.display:")
            }
            tool_name = root.context.get(ServiceKey[Callable[[str], str]]("tools.display-name.v1"))
            return message_rows(page, display_only=display_only, providers=MessageDisplayProviders(
                tool_name=tool_name, part_display=renderers,
            ))
