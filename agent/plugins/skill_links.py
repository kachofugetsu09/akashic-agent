from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from agent.plugins.manager import ActivePluginInfo
from infra.persistence.json_store import atomic_save_json, load_json

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
        self._workspace = workspace.resolve(strict=False)
        self._workspace_skills = self._workspace / "skills"
        self._workspace_drift_skills = self._workspace / "drift" / "skills"
        self._ownership_path = self._workspace / "runtime" / "plugin-skill-links.json"
        self._owned_links = self._load_owned_links()

    def validate(self, active_plugins: Sequence[ActivePluginInfo]) -> None:
        """Fail before promotion when projected names collide with user-owned paths."""

        self._validate_links(
            self._workspace_skills,
            self._build_expected_links(active_plugins, plugin_subpath=("skills",)),
        )
        self._validate_links(
            self._workspace_drift_skills,
            self._build_expected_links(
                active_plugins,
                plugin_subpath=("drift", "skills"),
            ),
        )

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
            ),
        )
        drift = self._sync_links(
            workspace_skills=self._workspace_drift_skills,
            expected=self._build_expected_links(
                active_plugins,
                plugin_subpath=("drift", "skills"),
            ),
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
            if current is None or not self._is_managed_link(link, current):
                raise RuntimeError(f"插件 skill 投影与用户软链接冲突: {link}")
            if _same_path(current, target):
                return "unchanged"
            self._record_owned_link(link, target)
            try:
                link.unlink()
            except OSError as e:
                raise RuntimeError(f"插件 skill 软链接删除失败: {link}") from e
            self._create_link(link, target)
            return "repaired"

        if link.exists():
            raise RuntimeError(f"插件 skill 投影与用户文件或目录冲突: {link}")

        self._record_owned_link(link, target)
        self._create_link(link, target)
        return "created"

    def _create_link(
        self,
        link: Path,
        target: Path,
    ) -> None:
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as e:
            raise RuntimeError(f"插件 skill 软链接创建失败: {link} -> {target}") from e

    def _cleanup_stale_links(
        self,
        workspace_skills: Path,
        expected: Mapping[str, Path],
    ) -> int:
        if not workspace_skills.exists():
            return 0
        removed = 0
        for item in list(workspace_skills.iterdir()):
            if item.name in expected:
                continue
            target = _readlink_target(item) if item.is_symlink() else None
            if target is None or not self._is_managed_link(item, target):
                continue
            try:
                item.unlink()
            except OSError as e:
                raise RuntimeError(f"插件 skill stale 软链接删除失败: {item}") from e
            self._forget_owned_link(item)
            removed += 1
        return removed

    def _is_managed_link(
        self,
        path: Path,
        target: Path,
    ) -> bool:
        recorded = self._owned_links.get(self._ownership_key(path))
        return recorded is not None and _same_path(Path(recorded), target)

    def _validate_links(
        self,
        workspace_skills: Path,
        expected: Mapping[str, Path],
    ) -> None:
        for link_name in expected:
            link = workspace_skills / link_name
            if link.is_symlink():
                target = _readlink_target(link)
                if target is None or not self._is_managed_link(link, target):
                    raise RuntimeError(f"插件 skill 投影与用户软链接冲突: {link}")
            elif link.exists():
                raise RuntimeError(f"插件 skill 投影与用户文件或目录冲突: {link}")

    def _ownership_key(self, link: Path) -> str:
        absolute = link.parent.resolve(strict=False) / link.name
        return str(absolute.relative_to(self._workspace))

    def _record_owned_link(self, link: Path, target: Path) -> None:
        self._owned_links[self._ownership_key(link)] = str(target.resolve(strict=False))
        self._save_owned_links()

    def _forget_owned_link(self, link: Path) -> None:
        _ = self._owned_links.pop(self._ownership_key(link), None)
        self._save_owned_links()

    def _load_owned_links(self) -> dict[str, str]:
        raw = load_json(
            self._ownership_path,
            default={"version": 1, "links": {}},
            domain="plugin_skill_links",
        )
        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise RuntimeError(f"插件 skill ownership 结构无效: {self._ownership_path}")
        links = raw.get("links")
        if not isinstance(links, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in links.items()
        ):
            raise RuntimeError(f"插件 skill ownership links 无效: {self._ownership_path}")
        return dict(links)

    def _save_owned_links(self) -> None:
        atomic_save_json(
            self._ownership_path,
            {"version": 1, "links": self._owned_links},
            ensure_ascii=False,
            domain="plugin_skill_links",
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
