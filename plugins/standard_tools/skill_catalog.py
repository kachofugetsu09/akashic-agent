"""standard_tools 拥有的 Skill 解析与可用性判断。"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import yaml

from agent.host_bridge.factory import build_requirements_checker
from agent.plugin_composition.assets import InstalledAsset
from agent.plugin_composition.shell_runtime import resolve_shell
from agent.plugin_contracts.inspection import (
    SKILL_INSPECTION as SKILL_INSPECTION,
)

SkillSource = Literal["plugin", "workspace", "user"]


class RequirementsChecker(Protocol):
    def check_requirements(
        self,
        bins: list[str],
        env: list[str],
    ) -> "RequirementsAvailability": ...


class RequirementsAvailability(Protocol):
    @property
    def missing_bins(self) -> tuple[str, ...]: ...

    @property
    def missing_env(self) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class SkillRecord:
    """保存一次读取的 Skill 正文与来源；资源使用时读取当前文件。"""

    name: str
    display_name: str
    source: SkillSource
    source_id: str
    root_dir: Path
    skill_file: Path
    content: str
    description: str
    when_to_use: str
    config: dict[str, Any]
    always: bool
    available: bool
    missing: str


def skill_body(content: str) -> str:
    """去掉 frontmatter，保留 Skill 正文。"""

    if content.startswith("---"):
        match = re.match(r"^---\n.*?\n---\n", content, re.DOTALL)
        if match:
            return content[match.end() :].strip()
    return content


def _by_name(path: Path) -> str:
    return path.name


def _default_user_skill_dir() -> Path:
    """用户级 Skill 目录；与插件随包资产并列的独立来源。"""

    return Path.home() / ".akashic" / "skills"


class SkillCatalogParser:
    """按来源分层解析 Skill，并计算本 generation 的可用性。

    来源优先级从高到低是工作区目录、用户目录、插件随包资产；高层同名 Skill 静默
    覆盖低层，插件资产同一层内重名仍是错误。本地目录不需要打包成插件。
    """

    def __init__(self, capability_checker: RequirementsChecker | None = None):
        if capability_checker is None:
            capability_checker = build_requirements_checker()
        self._capability_checker = capability_checker
        self._shell_path: str | None = None

    @staticmethod
    def local_roots(
        *,
        workspace_dir: Path | None,
        user_dir: Path | None,
    ) -> tuple[tuple[SkillSource, Path], ...]:
        """本地目录来源，按优先级从低到高排列。"""

        roots: list[tuple[SkillSource, Path]] = [
            ("user", user_dir if user_dir is not None else _default_user_skill_dir()),
        ]
        if workspace_dir is not None:
            roots.append(("workspace", workspace_dir / "skills"))
        return tuple(roots)

    def parse(
        self,
        assets: tuple[InstalledAsset, ...],
        *,
        workspace_dir: Path | None = None,
        user_dir: Path | None = None,
        category: str = "skills",
    ) -> tuple[SkillRecord, ...]:
        records: dict[str, SkillRecord] = {}

        # 1. 插件随包资产（最低优先级，同层重名报错）
        plugin_records: dict[str, SkillRecord] = {}
        for asset in assets:
            if asset.category != category:
                continue
            if not asset.root_dir.is_dir():
                raise FileNotFoundError(f"声明资产目录不存在: {asset.root_dir}")
            for skill_dir in sorted(asset.root_dir.iterdir(), key=lambda item: item.name):
                if not skill_dir.is_dir():
                    continue
                skill_file = skill_dir / "SKILL.md"
                if not skill_file.is_file():
                    continue
                name = skill_dir.name
                if name in plugin_records:
                    previous = plugin_records[name].source_id
                    raise RuntimeError(
                        f"插件 Skill 名称重复: {name} ({previous}, {asset.owner_id})"
                    )
                plugin_records[name] = self._build_record(
                    name=name,
                    root_dir=skill_dir,
                    skill_file=skill_file,
                    source="plugin",
                    source_id=asset.owner_id,
                )

        # 2. 本地来源按 user、workspace 顺序覆盖；普通目录链接可直接使用。
        records.update(plugin_records)
        plugin_roots = {item.root_dir.resolve() for item in plugin_records.values()}
        for source, root in self.local_roots(workspace_dir=workspace_dir, user_dir=user_dir):
            if not root.is_dir():
                continue
            for skill_dir in sorted(root.iterdir(), key=_by_name):
                if not skill_dir.is_dir():
                    continue
                resolved = skill_dir.resolve(strict=True)
                if source == "workspace" and skill_dir.is_symlink() and resolved in plugin_roots:
                    continue
                skill_file = resolved / "SKILL.md"
                if skill_file.is_file():
                    records[skill_dir.name] = self._build_record(
                        name=skill_dir.name, root_dir=resolved, skill_file=skill_file,
                        source=source, source_id=str(root),
                    )

        return tuple(records[name] for name in sorted(records))

    def _build_record(
        self,
        *,
        name: str,
        root_dir: Path,
        skill_file: Path,
        source: SkillSource,
        source_id: str,
    ) -> SkillRecord:
        content = skill_file.read_text(encoding="utf-8")
        meta = self._parse_frontmatter(content) or {}
        config = self._parse_skill_config(meta.get("metadata", ""), skill_file=skill_file)
        missing = self._get_missing_requirements(config)
        return SkillRecord(
            name=name,
            display_name=meta.get("name") or name,
            source=source,
            source_id=source_id,
            root_dir=root_dir,
            skill_file=skill_file,
            content=content,
            description=meta.get("description") or name,
            when_to_use=meta.get("when_to_use", ""),
            config=config,
            always=self._as_bool(config.get("always"))
            or self._as_bool(meta.get("always")),
            available=not missing,
            missing=missing,
        )

    def _parse_frontmatter(self, content: str) -> dict[str, Any]:
        if not content.startswith("---"):
            return {}
        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}
        loaded = cast(object, yaml.safe_load(parts[1]) or {})
        if not isinstance(loaded, dict):
            return {}
        data = cast(dict[object, Any], loaded)
        return {str(key): value for key, value in data.items()}

    def _parse_skill_config(
        self,
        raw: str | object,
        *,
        skill_file: Path,
    ) -> dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            data = cast(dict[str, Any], raw)
        else:
            text = str(raw).strip()
            if not text:
                return {}
            try:
                parsed: Any = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Skill metadata 不是有效 JSON: {skill_file}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"Skill metadata 必须是对象: {skill_file}")
            data = cast(dict[str, Any], parsed)
        for key in ("akashic", "skill"):
            value = data.get(key)
            if isinstance(value, dict):
                return cast(dict[str, Any], value)
        return data

    def _get_missing_requirements(self, config: dict[str, Any]) -> str:
        requires = config.get("requires", {})
        if not isinstance(requires, dict):
            return ""
        requires_dict = cast(dict[str, object], requires)
        bins = self._string_list(requires_dict.get("bins"))
        env_names = self._string_list(requires_dict.get("env"))
        if self._capability_checker is not None and (bins or env_names):
            availability = self._capability_checker.check_requirements(
                bins,
                env_names,
            )
            return ", ".join(
                [
                    *(f"CLI: {name}" for name in availability.missing_bins),
                    *(f"ENV: {name}" for name in availability.missing_env),
                ]
            )
        missing: list[str] = []
        for binary in bins:
            if not shutil.which(binary, path=self._binary_search_path()):
                missing.append(f"CLI: {binary}")
        for env in env_names:
            if not os.environ.get(env):
                missing.append(f"ENV: {env}")
        return ", ".join(missing)

    def _binary_search_path(self) -> str:
        if self._shell_path is None:
            if os.name == "nt":
                self._shell_path = os.environ.get("PATH", "")
            else:
                shell = resolve_shell()
                result = subprocess.run(
                    shell.derive_argv("command env -0", login=True),
                    check=True,
                    capture_output=True,
                    timeout=10,
                )
                paths = [
                    item[5:]
                    for item in result.stdout.split(b"\0")
                    if item.startswith(b"PATH=")
                ]
                if not paths:
                    raise RuntimeError(f"用户 login shell 未导出 PATH: {shell.path}")
                self._shell_path = os.fsdecode(paths[-1])
        return self._shell_path

    @staticmethod
    def _string_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        items = cast(list[object], value)
        return [item for item in items if isinstance(item, str)]

    @staticmethod
    def _as_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return False


class SkillInspectionProvider:
    """向宿主发布当前固定 generation 的技能只读投影。"""

    def __init__(self, read_catalog: Callable[[], Awaitable[tuple[SkillRecord, ...]]]) -> None:
        self._read_catalog = read_catalog

    async def list_skills(self) -> tuple[Mapping[str, object], ...]:
        return tuple(
            {
                "name": record.name,
                "display_name": record.display_name,
                "description": record.description,
                "source": record.source,
                "source_id": record.source_id,
                "available": record.available,
                "missing": record.missing,
            }
            for record in await self._read_catalog()
        )


__all__ = [
    "SKILL_INSPECTION",
    "SkillCatalogParser",
    "SkillInspectionProvider",
    "SkillRecord",
    "skill_body",
]
