from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from agent.plugin_composition.context import Context
from agent.plugin_composition.model import CompositionError, PluginRuntime, ServiceKey


@dataclass(frozen=True, slots=True)
class PluginSkillContribution:
    skill_roots: tuple[Path, ...] = ()
    drift_skill_roots: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class _SkillRegistration:
    token: int
    plugin_id: str
    kind: Literal["skill", "drift_skill"]
    path: Path


SKILLS = ServiceKey["PluginSkills"]("core.skills")


class PluginSkills:
    """Collect plugin-owned Skill roots for one composition Root."""

    def __init__(self) -> None:
        self._next_token = 1
        self._registrations: dict[int, _SkillRegistration] = {}
        self._frozen: Mapping[str, PluginSkillContribution] | None = None

    async def register(
        self,
        ctx: Context,
        relative_path: str,
        *,
        drift: bool = False,
    ) -> None:
        """Register one plugin-owned Skill root as a Fiber Effect."""

        runtime = ctx.runtime
        path = _resolve_skill_root(runtime, relative_path)
        kind: Literal["skill", "drift_skill"] = "drift_skill" if drift else "skill"
        await ctx.effect(
            lambda: self._register(runtime.plugin_id, kind, path),
            label=f"skill:{kind}:{relative_path}",
        )

    def freeze(self) -> Mapping[str, PluginSkillContribution]:
        """Freeze the roots that Core may compile into generation catalogs."""

        if self._frozen is not None:
            return self._frozen

        # 1. Preserve registration order within each plugin and catalog lane.
        skills: dict[str, list[Path]] = {}
        drift_skills: dict[str, list[Path]] = {}
        for registration in sorted(
            self._registrations.values(),
            key=lambda item: item.token,
        ):
            target = drift_skills if registration.kind == "drift_skill" else skills
            target.setdefault(registration.plugin_id, []).append(registration.path)

        # 2. Publish one immutable generation input.
        plugin_ids = {*skills, *drift_skills}
        self._frozen = MappingProxyType(
            {
                plugin_id: PluginSkillContribution(
                    skill_roots=tuple(skills.get(plugin_id, ())),
                    drift_skill_roots=tuple(drift_skills.get(plugin_id, ())),
                )
                for plugin_id in sorted(plugin_ids)
            }
        )
        return self._frozen

    def _register(
        self,
        plugin_id: str,
        kind: Literal["skill", "drift_skill"],
        path: Path,
    ) -> Callable[[], None]:
        """Add one declaration and return its exact inverse."""

        # 1. Registration closes before snapshot compilation.
        if self._frozen is not None:
            raise CompositionError(
                "PLUGIN_SKILLS_FROZEN",
                "插件 Skill 声明已冻结，不能在 snapshot 发布后新增",
            )
        if any(
            item.plugin_id == plugin_id and item.kind == kind and item.path == path
            for item in self._registrations.values()
        ):
            raise CompositionError(
                "DUPLICATE_PLUGIN_SKILL_ROOT",
                f"插件 Skill root 重复: {plugin_id} {kind} {path}",
            )

        # 2. The disposer owns only this registration token.
        token = self._next_token
        self._next_token += 1
        self._registrations[token] = _SkillRegistration(
            token=token,
            plugin_id=plugin_id,
            kind=kind,
            path=path,
        )

        def cleanup() -> None:
            _ = self._registrations.pop(token, None)

        return cleanup


def _resolve_skill_root(runtime: PluginRuntime, relative_path: str) -> Path:
    """Resolve one Skill root without allowing plugin-source escape."""

    if (
        not isinstance(relative_path, str)
        or not relative_path
        or relative_path != relative_path.strip()
        or Path(relative_path).is_absolute()
    ):
        raise ValueError("插件 Skill root 必须是非空相对路径")
    plugin_root = runtime.plugin_dir.resolve(strict=True)
    try:
        path = (plugin_root / relative_path).resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError(f"插件 Skill root 不存在: {relative_path}") from error
    if not path.is_relative_to(plugin_root):
        raise RuntimeError(f"插件 Skill root 路径越界: {relative_path}")
    if not path.is_dir():
        raise RuntimeError(f"插件 Skill root 不是目录: {relative_path}")
    return path
