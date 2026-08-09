from __future__ import annotations

import logging
import secrets
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
        self._owned_links, self._pending_links = self._load_ownership()
        self._recover_pending_links()

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
            self._transition_link(link, old=current, new=target)
            return "repaired"

        if link.exists():
            raise RuntimeError(f"插件 skill 投影与用户文件或目录冲突: {link}")

        self._transition_link(link, old=None, new=target)
        return "created"

    def _replace_link(
        self,
        link: Path,
        target: Path,
    ) -> None:
        temporary = link.with_name(f".{link.name}.akashic-{secrets.token_hex(8)}")
        try:
            temporary.symlink_to(target, target_is_directory=True)
            temporary.replace(link)
        except OSError as e:
            raise RuntimeError(f"插件 skill 软链接创建失败: {link} -> {target}") from e
        finally:
            if temporary.is_symlink():
                temporary.unlink()

    def _create_link(self, link: Path, target: Path) -> None:
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as error:
            raise RuntimeError(f"插件 skill 软链接创建失败: {link} -> {target}") from error

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
            self._transition_link(item, old=target, new=None)
            removed += 1
        return removed

    def _is_managed_link(
        self,
        path: Path,
        target: Path,
    ) -> bool:
        key = self._ownership_key(path)
        recorded = self._owned_links.get(key)
        if recorded is not None and _same_path(Path(recorded), target):
            return True
        pending = self._pending_links.get(key)
        if pending is None:
            return False
        return any(
            value is not None and _same_path(Path(value), target)
            for value in (pending["old"], pending["new"])
        )

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

    def _transition_link(
        self,
        link: Path,
        *,
        old: Path | None,
        new: Path | None,
    ) -> None:
        """Journal, atomically mutate, and commit one managed symlink transition."""

        # 1. Persist both valid endpoints before changing the filesystem.
        key = self._ownership_key(link)
        pending_links = dict(self._pending_links)
        pending_links[key] = {
            "old": _path_text(old),
            "new": _path_text(new),
        }
        self._write_ownership(self._owned_links, pending_links)
        self._pending_links = pending_links

        # 2. Replace the directory entry, then commit ownership to the observed target.
        if new is None:
            try:
                link.unlink()
            except OSError as error:
                raise RuntimeError(f"插件 skill stale 软链接删除失败: {link}") from error
        elif old is None:
            self._create_link(link, new)
        else:
            self._replace_link(link, new)
        self._commit_transition(key, new)

    def _commit_transition(self, key: str, target: Path | None) -> None:
        owned_links = dict(self._owned_links)
        pending_links = dict(self._pending_links)
        if target is None:
            _ = owned_links.pop(key, None)
        else:
            owned_links[key] = str(target.resolve(strict=False))
        _ = pending_links.pop(key, None)
        self._write_ownership(owned_links, pending_links)
        self._owned_links = owned_links
        self._pending_links = pending_links

    def _load_ownership(self) -> tuple[dict[str, str], dict[str, dict[str, str | None]]]:
        raw = load_json(
            self._ownership_path,
            default={"version": 1, "links": {}, "pending": {}},
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
        pending = raw.get("pending", {})
        if not isinstance(pending, dict):
            raise RuntimeError(f"插件 skill ownership pending 无效: {self._ownership_path}")
        normalized_pending: dict[str, dict[str, str | None]] = {}
        for key, value in pending.items():
            if (
                not isinstance(key, str)
                or not isinstance(value, dict)
                or set(value) != {"old", "new"}
                or (
                    value.get("old") is not None
                    and not isinstance(value.get("old"), str)
                )
                or (
                    value.get("new") is not None
                    and not isinstance(value.get("new"), str)
                )
            ):
                raise RuntimeError(
                    f"插件 skill ownership pending item 无效: {self._ownership_path}"
                )
            normalized_pending[key] = {
                "old": value.get("old"),
                "new": value.get("new"),
            }
        return dict(links), normalized_pending

    def _recover_pending_links(self) -> None:
        if not self._pending_links:
            return
        for key, transition in tuple(self._pending_links.items()):
            link = self._link_for_key(key)
            actual = _readlink_target(link) if link.is_symlink() else None
            if link.exists() and not link.is_symlink():
                raise RuntimeError(f"插件 skill pending 路径被用户文件占用: {link}")
            old = _optional_path(transition["old"])
            new = _optional_path(transition["new"])
            if _optional_same_path(actual, new):
                self._commit_transition(key, new)
            elif _optional_same_path(actual, old):
                pending_links = dict(self._pending_links)
                _ = pending_links.pop(key, None)
                self._write_ownership(self._owned_links, pending_links)
                self._pending_links = pending_links
            else:
                raise RuntimeError(f"插件 skill pending 状态无法恢复: {link}")

    def _link_for_key(self, key: str) -> Path:
        relative = Path(key)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"插件 skill ownership key 越界: {key}")
        link = self._workspace / relative
        if link.parent.resolve(strict=False) not in {
            self._workspace_skills.resolve(strict=False),
            self._workspace_drift_skills.resolve(strict=False),
        }:
            raise RuntimeError(f"插件 skill ownership key 非法: {key}")
        return link

    def _write_ownership(
        self,
        owned_links: Mapping[str, str],
        pending_links: Mapping[str, Mapping[str, str | None]],
    ) -> None:
        atomic_save_json(
            self._ownership_path,
            {
                "version": 1,
                "links": dict(owned_links),
                "pending": {
                    key: dict(value) for key, value in pending_links.items()
                },
            },
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


def _path_text(path: Path | None) -> str | None:
    return str(path.resolve(strict=False)) if path is not None else None


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value is not None else None


def _optional_same_path(left: Path | None, right: Path | None) -> bool:
    if left is None or right is None:
        return left is right
    return _same_path(left, right)
