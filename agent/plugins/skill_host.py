from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import tempfile
import weakref
from typing import TYPE_CHECKING

from agent.plugin_composition.assets import InstalledAsset

if TYPE_CHECKING:
    from agent.plugins.generation import PluginGeneration


@dataclass(frozen=True)
class PreparedAssetCatalog:
    """一个 generation 的固定声明资产树；宿主不解释树内文件。"""

    generation_id: str
    snapshot: AssetSnapshot
    assets: tuple[InstalledAsset, ...]


class AssetSnapshot:
    def __init__(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="akashic-asset-catalog-"))
        self._finalizer = weakref.finalize(
            self,
            _remove_snapshot_tree,
            self.root,
        )

    def cleanup(self) -> None:
        if not self._finalizer.alive:
            return
        try:
            _remove_snapshot_tree(self.root)
        except FileNotFoundError:
            pass
        _ = self._finalizer.detach()


def _remove_snapshot_tree(root: Path) -> None:
    """Remove a private snapshot even when its immutable source modes were copied."""

    # 1. Snapshot copytree preserves a read-only image source, so restore owner writes.
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            continue
        path.chmod(path.stat().st_mode | 0o200)
    if root.exists():
        root.chmod(root.stat().st_mode | 0o200)

    # 2. Delete only the private mkdtemp tree owned by this snapshot.
    shutil.rmtree(root)


class PluginAssetHost:
    """固定插件声明目录并按 generation 保存其生命周期。"""

    def __init__(self) -> None:
        self._catalogs: dict[str, PreparedAssetCatalog] = {}

    def prepare(
        self,
        generation_id: str,
        *,
        asset_roots: dict[str, tuple[tuple[str, tuple[Path, ...]], ...]],
    ) -> PreparedAssetCatalog:
        snapshot = AssetSnapshot()
        try:
            assets = self._snapshot_assets(snapshot.root, asset_roots)
        except BaseException:
            snapshot.cleanup()
            raise
        catalog = PreparedAssetCatalog(
            generation_id=generation_id,
            snapshot=snapshot,
            assets=assets,
        )
        self._catalogs[generation_id] = catalog
        return catalog

    def get(self, generation_id: str) -> PreparedAssetCatalog | None:
        return self._catalogs.get(generation_id)

    def close(self, generation_id: str) -> None:
        catalog = self._catalogs.get(generation_id)
        if catalog is not None:
            catalog.snapshot.cleanup()
            del self._catalogs[generation_id]

    @staticmethod
    def roots_for(
        generations: list[PluginGeneration],
    ) -> dict[str, tuple[tuple[str, tuple[Path, ...]], ...]]:
        """返回 generation 声明的类别到固定目录映射。"""

        return {
            generation.plugin_id: generation.contributions.asset_roots
            for generation in generations
        }

    @staticmethod
    def _snapshot_assets(
        snapshot_root: Path,
        declared: dict[str, tuple[tuple[str, tuple[Path, ...]], ...]],
    ) -> tuple[InstalledAsset, ...]:
        """复制目录树并返回宿主拥有的固定值，不读取目录内容。"""

        assets: list[InstalledAsset] = []
        for owner_id, categories in sorted(declared.items()):
            owner_key = hashlib.sha256(owner_id.encode()).hexdigest()[:12]
            for category, roots in categories:
                category_key = hashlib.sha256(category.encode()).hexdigest()[:12]
                for index, root in enumerate(roots):
                    target = snapshot_root / owner_key / category_key / str(index)
                    _ = shutil.copytree(root, target)
                    assets.append(
                        InstalledAsset(
                            owner_id=owner_id,
                            category=category,
                            root_dir=target,
                        )
                    )
        return tuple(assets)
