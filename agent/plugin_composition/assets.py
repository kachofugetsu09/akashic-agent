"""来源中立的 generation 固定插件声明资产读取口。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

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


INSTALLED_ASSETS = ServiceKey[
    Callable[[], tuple[InstalledAsset, ...]]
]("core.installed_assets.v1")


__all__ = ["INSTALLED_ASSETS", "InstalledAsset"]
