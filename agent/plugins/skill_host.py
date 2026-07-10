from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING

from agent.skills import SkillIndex, SkillsLoader

if TYPE_CHECKING:
    from agent.plugins.generation import PluginGeneration


@dataclass(frozen=True)
class PreparedSkillCatalog:
    generation_id: str
    snapshot_root: Path
    normal: SkillIndex
    drift: SkillIndex

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted({*self.normal.records, *self.drift.records}))


class PluginSkillHost:
    def __init__(self, workspace: Path | None) -> None:
        self._workspace = workspace
        self._catalogs: dict[str, PreparedSkillCatalog] = {}

    def prepare(
        self,
        generation_id: str,
        *,
        normal_roots: dict[str, tuple[Path, ...]],
        drift_roots: dict[str, tuple[Path, ...]],
        ignored_normal_roots: tuple[Path, ...],
        ignored_drift_roots: tuple[Path, ...],
    ) -> PreparedSkillCatalog:
        self._validate_unique_names(normal_roots)
        self._validate_unique_names(drift_roots)
        workspace = self._workspace or Path("/__akashic_no_workspace__")
        snapshot_root = Path(tempfile.mkdtemp(prefix="akashic-skill-catalog-"))
        try:
            frozen_normal = self._snapshot_roots(
                snapshot_root / "normal",
                normal_roots,
            )
            frozen_drift = self._snapshot_roots(
                snapshot_root / "drift",
                drift_roots,
            )
            normal = SkillsLoader(
                workspace,
                plugin_roots=frozen_normal,
                ignored_workspace_symlink_roots=ignored_normal_roots,
            ).build_index()
            drift = SkillsLoader(
                workspace,
                builtin_skills_dir=None,
                workspace_skills_dir=workspace / "drift" / "skills",
                plugin_roots=frozen_drift,
                ignored_workspace_symlink_roots=ignored_drift_roots,
            ).build_index()
        except BaseException:
            shutil.rmtree(snapshot_root)
            raise
        catalog = PreparedSkillCatalog(
            generation_id=generation_id,
            snapshot_root=snapshot_root,
            normal=normal,
            drift=drift,
        )
        self._catalogs[generation_id] = catalog
        return catalog

    def get(self, generation_id: str) -> PreparedSkillCatalog | None:
        return self._catalogs.get(generation_id)

    def close(self, generation_id: str) -> None:
        catalog = self._catalogs.pop(generation_id, None)
        if catalog is not None:
            shutil.rmtree(catalog.snapshot_root)

    @staticmethod
    def roots_for(
        generations: list[PluginGeneration],
        *,
        drift: bool,
    ) -> dict[str, tuple[Path, ...]]:
        return {
            generation.plugin_id: (
                generation.contributions.drift_skill_roots
                if drift
                else generation.contributions.skill_roots
            )
            for generation in generations
        }

    @staticmethod
    def _validate_unique_names(
        plugin_roots: dict[str, tuple[Path, ...]],
    ) -> None:
        owners: dict[str, str] = {}
        for plugin_id, roots in sorted(plugin_roots.items()):
            for root in roots:
                for skill_dir in sorted(root.iterdir()):
                    if not skill_dir.is_dir() or not (skill_dir / "SKILL.md").is_file():
                        continue
                    owner = owners.get(skill_dir.name)
                    if owner is not None:
                        raise RuntimeError(
                            f"插件 Skill 名称重复: {skill_dir.name} ({owner}, {plugin_id})"
                        )
                    owners[skill_dir.name] = plugin_id

    @staticmethod
    def _snapshot_roots(
        snapshot_dir: Path,
        plugin_roots: dict[str, tuple[Path, ...]],
    ) -> dict[str, tuple[Path, ...]]:
        frozen: dict[str, tuple[Path, ...]] = {}
        for plugin_id, roots in sorted(plugin_roots.items()):
            owner = hashlib.sha256(plugin_id.encode()).hexdigest()[:12]
            copies: list[Path] = []
            for index, root in enumerate(roots):
                target = snapshot_dir / owner / str(index)
                _ = shutil.copytree(root, target)
                copies.append(target)
            frozen[plugin_id] = tuple(copies)
        return frozen
