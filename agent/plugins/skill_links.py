from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from agent.plugins.manager import ActivePluginInfo

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class PluginSkillSyncResult:
    expected: int = 0
    created: int = 0
    repaired: int = 0
    removed: int = 0
    skipped: int = 0


class PluginSkillLinker:
    def __init__(
        self,
        *,
        workspace: Path,
        plugin_roots: Sequence[Path],
        memory_engine: object | None,
    ) -> None:
        self._workspace_skills = workspace / "skills"
        self._workspace_drift_skills = workspace / "drift" / "skills"
        self._plugin_roots = [root.resolve(strict=False) for root in plugin_roots]
        self._memory_engine = memory_engine

    # 将已生效插件的普通 skill 和 drift skill 同步成 workspace 下的软链接。
    def sync(
        self,
        active_plugins: Sequence[ActivePluginInfo],
    ) -> PluginSkillSyncResult:
        normal = self._sync_links(
            workspace_skills=self._workspace_skills,
            expected=self._build_expected_links(
                active_plugins,
                plugin_subpath=("skills",),
                manifest_key="skills",
            ),
            managed_subpath=("skills",),
        )
        drift = self._sync_links(
            workspace_skills=self._workspace_drift_skills,
            expected=self._build_expected_links(
                active_plugins,
                plugin_subpath=("drift", "skills"),
                manifest_key="drift_skills",
            ),
            managed_subpath=("drift", "skills"),
        )
        return PluginSkillSyncResult(
            expected=normal.expected + drift.expected,
            created=normal.created + drift.created,
            repaired=normal.repaired + drift.repaired,
            removed=normal.removed + drift.removed,
            skipped=normal.skipped + drift.skipped,
        )

    def _sync_links(
        self,
        *,
        workspace_skills: Path,
        expected: Mapping[str, Path],
        managed_subpath: Sequence[str],
    ) -> PluginSkillSyncResult:
        created = 0
        repaired = 0
        skipped = 0

        if expected:
            workspace_skills.mkdir(parents=True, exist_ok=True)

        for link_name, target in expected.items():
            link = workspace_skills / link_name
            action = self._ensure_link(link, target)
            if action == "created":
                created += 1
            elif action == "repaired":
                repaired += 1
            elif action == "skipped":
                skipped += 1

        removed = self._cleanup_stale_links(
            workspace_skills,
            expected,
            managed_subpath,
        )
        return PluginSkillSyncResult(
            expected=len(expected),
            created=created,
            repaired=repaired,
            removed=removed,
            skipped=skipped,
        )

    def _build_expected_links(
        self,
        active_plugins: Sequence[ActivePluginInfo],
        *,
        plugin_subpath: Sequence[str],
        manifest_key: str,
    ) -> dict[str, Path]:
        expected: dict[str, Path] = {}
        for plugin in active_plugins:
            if not _is_safe_name(plugin.plugin_id):
                logger.warning("插件 skill 跳过非法 plugin_id: %s", plugin.plugin_id)
                continue
            for skill_dir in _iter_plugin_skill_dirs(plugin, plugin_subpath):
                if not _is_safe_name(skill_dir.name):
                    logger.warning(
                        "插件 skill 跳过非法 skill 名称: %s/%s",
                        plugin.plugin_id,
                        skill_dir.name,
                    )
                    continue
                link_name = skill_dir.name
                target = skill_dir.resolve(strict=False)
                existing = expected.get(link_name)
                if existing is not None and existing != target:
                    logger.warning("插件 skill 名称重复，保留第一项: %s", link_name)
                    continue
                expected[link_name] = target
        return expected

    def _ensure_link(
        self,
        link: Path,
        target: Path,
    ) -> str:
        if link.is_symlink():
            current = _readlink_target(link)
            if current is not None and _same_path(current, target):
                return "unchanged"
            try:
                link.unlink()
            except OSError as e:
                logger.warning("插件 skill 软链接删除失败 (%s): %s", link, e)
                return "skipped"
            return "repaired" if self._create_link(link, target) else "skipped"

        if link.exists():
            if not _remove_existing_path(link):
                return "skipped"
            return "repaired" if self._create_link(link, target) else "skipped"

        return "created" if self._create_link(link, target) else "skipped"

    def _create_link(
        self,
        link: Path,
        target: Path,
    ) -> bool:
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as e:
            logger.warning("插件 skill 软链接创建失败 (%s -> %s): %s", link, target, e)
            return False
        return True

    def _cleanup_stale_links(
        self,
        workspace_skills: Path,
        expected: Mapping[str, Path],
        managed_subpath: Sequence[str],
    ) -> int:
        if not workspace_skills.exists():
            return 0
        removed = 0
        for item in list(workspace_skills.iterdir()):
            if item.name in expected:
                continue
            if not self._is_managed_link(item, managed_subpath):
                continue
            try:
                item.unlink()
            except OSError as e:
                logger.warning("插件 skill stale 软链接删除失败 (%s): %s", item, e)
                continue
            removed += 1
        return removed

    def _is_managed_link(
        self,
        path: Path,
        managed_subpath: Sequence[str],
    ) -> bool:
        if not path.is_symlink():
            return False
        target = _readlink_target(path)
        if target is None:
            return False
        return any(
            _is_under_plugin_skills(target, root, managed_subpath)
            for root in self._plugin_roots
        )


def _iter_plugin_skill_dirs(
    plugin: ActivePluginInfo,
    plugin_subpath: Sequence[str],
) -> list[Path]:
    result: list[Path] = []
    roots = _resolve_skill_roots(plugin, plugin_subpath)
    for skills_dir in roots:
        if not skills_dir.is_dir():
            continue
        for child in sorted(skills_dir.iterdir(), key=lambda item: item.name):
            if not child.is_dir():
                continue
            if not (child / "SKILL.md").exists():
                continue
            result.append(child)
    return result


def _resolve_skill_roots(
    plugin: ActivePluginInfo,
    plugin_subpath: Sequence[str],
) -> tuple[Path, ...]:
    if tuple(plugin_subpath) == ("skills",):
        return plugin.skill_roots
    if tuple(plugin_subpath) == ("drift", "skills"):
        return plugin.drift_skill_roots
    return ()


def _is_safe_name(name: str) -> bool:
    value = name.strip()
    return bool(value) and "/" not in value and "\\" not in value and ".." not in value


def _readlink_target(link: Path) -> Path | None:
    try:
        raw = link.readlink()
    except OSError as e:
        logger.warning("读取软链接失败 (%s): %s", link, e)
        return None
    if raw.is_absolute():
        return raw.resolve(strict=False)
    return (link.parent / raw).resolve(strict=False)


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def _remove_existing_path(path: Path) -> bool:
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError as e:
        logger.warning("插件 skill 覆盖旧路径失败 (%s): %s", path, e)
        return False
    return True


def _is_under_plugin_skills(
    target: Path,
    plugin_root: Path,
    managed_subpath: Sequence[str],
) -> bool:
    normalized_target = target.resolve(strict=False)
    normalized_root = plugin_root.resolve(strict=False)
    try:
        relative = normalized_target.relative_to(normalized_root)
    except ValueError:
        return False
    parts = relative.parts
    expected = tuple(managed_subpath)
    return len(parts) >= len(expected) + 1 and parts[-(len(expected) + 1):-1] == expected
