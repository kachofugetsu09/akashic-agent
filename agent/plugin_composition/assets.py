"""普通插件共享的资产注册与只读合同；不解释资产内容。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agent.plugin_composition.context import Context
from agent.plugin_composition.effect import Effect

from agent.plugin_composition.model import ServiceKey


@dataclass(frozen=True, slots=True)
class InstalledAsset:
    """冻结一个插件声明的资产集合；不解释其文件内容。"""

    owner_id: str
    category: str
    root_dir: Path

    def __post_init__(self) -> None:
        if not self.owner_id or self.owner_id.strip() != self.owner_id:
            raise ValueError("资产 owner_id 必须是非空字符串")
        if not self.category or self.category.strip() != self.category:
            raise ValueError("资产 category 必须是非空字符串")
        if not self.root_dir.is_absolute():
            raise ValueError("资产 root_dir 必须是绝对路径")


class InstalledAssets(Protocol):
    """读取精确作用域的资产，或登记调用方自己的代码目录。"""

    def __call__(self) -> tuple[InstalledAsset, ...]: ...

    async def register(
        self, ctx: Context, category: str, relative_path: str,
    ) -> Effect: ...


INSTALLED_ASSETS = ServiceKey[InstalledAssets]("core.installed_assets.v1")


__all__ = ["INSTALLED_ASSETS", "InstalledAsset", "InstalledAssets"]
