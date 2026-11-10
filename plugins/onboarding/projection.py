"""步骤随贡献者 Effect 存在，顺序由真实功能 Fiber 计算。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from agent.plugin_composition import Context, Effect
from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG
from plugins.onboarding.contract import Ability, Step


@dataclass(frozen=True)
class Registration:
    context: Context
    step: Step


@dataclass(frozen=True)
class Group:
    title: str
    ability: Ability | None


class Registry:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.steps: dict[str, Registration] = {}
        self.groups: dict[str, Group] = {}

    async def group(self, ctx: Context, key: str, title: str, ability: Ability | None = None) -> Effect:
        if ctx.root_instance_token is not self.ctx.root_instance_token:
            raise ValueError("配置分组不能跨运行图")
        def start():
            if key in self.groups:
                raise ValueError(f"配置分组重复: {key}")
            self.groups[key] = Group(title, ability)
            return lambda: self.groups.pop(key)
        return await ctx.effect(start, label=f"onboarding-group:{key}")

    async def register(self, ctx: Context, step: Step) -> Effect:
        """身份取自贡献 Context，回收只移除自己的步骤。"""
        if ctx.root_instance_token is not self.ctx.root_instance_token:
            raise ValueError("配置贡献不能跨运行图")
        key = f"{ctx.runtime.plugin_id}/{step.key}"
        def start():
            if key in self.steps:
                raise ValueError(f"配置步骤重复: {key}")
            self.steps[key] = Registration(ctx, step)
            return lambda: self.steps.pop(key)
        return await ctx.effect(start, label=f"onboarding:{step.key}")

    async def catalog(self) -> dict[str, object]:
        """只返回目录；单项状态由独立请求读取，故障不会吞掉其他项。"""
        async with self.ctx.runtime_scope():
            graph = self.ctx.require(RUNTIME_CATALOG)(self.ctx)
        items = cast(list[dict[str, Any]], graph["plugins"])
        targets: dict[str, set[int]] = {}
        for registration in self.steps.values():
            targets.setdefault(registration.context.runtime.plugin_id, set()).update(registration.step.function_fibers)
        edges: dict[str, set[str]] = {}
        unresolved: set[str] = set()
        for item in items:
            owner = item["id"]
            fibers = item["composition"]["fibers"]
            selected = targets.get(owner, set())
            # 显式功能分支或无步骤插件的必需分支；不把 UI 注册当业务依赖。
            functions = [fiber for fiber in fibers if fiber["fiber_id"] in selected or (not selected and fiber["required"])]
            edges[owner] = {provider for fiber in functions for provider in fiber["dependency_providers"].values()
                            if provider is not None and provider != owner}
            if any(fiber["missing_services"] for fiber in functions):
                unresolved.add(owner)
        levels: dict[str, int | None] = {}
        visiting: set[str] = set()
        def level(owner: str) -> int | None:
            if owner in levels:
                return levels[owner]
            if owner in visiting:
                raise ValueError(f"功能依赖存在环: {owner}")
            visiting.add(owner)
            parents = [level(parent) for parent in edges.get(owner, set())]
            result = None if owner in unresolved or any(parent is None for parent in parents) else (max(cast(list[int], parents)) + 1 if parents else 0)
            visiting.remove(owner)
            levels[owner] = result
            return result
        rows = []
        for key, registration in tuple(self.steps.items()):
            owner = registration.context.runtime.plugin_id
            step = registration.step
            rows.append({"id": key, "owner": owner, "key": step.key, "title": step.title,
                "group": step.group, "group_title": self._group_title(step.group), "route": step.route,
                "level": level(owner), "dependency_count": len(edges.get(owner, set()))})
        rows.sort(key=lambda row: (float("inf") if row["level"] is None else row["level"], row["dependency_count"], row["owner"], row["key"]))
        return {"steps": rows, "groups": self._group_rows(rows)}

    def _group_title(self, key: str) -> str:
        group = self.groups.get(key)
        return key if group is None else group.title

    # 分组按其首个步骤的依赖顺序排列；没有步骤的分组不出现，引导里没有空卡。
    def _group_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        seen: set[str] = set()
        for row in rows:
            key = cast(str, row["group"])
            if key in seen:
                continue
            seen.add(key)
            group = self.groups.get(key)
            ability = None if group is None else group.ability
            result.append({
                "key": key, "title": self._group_title(key),
                "required": ability is not None and ability.required,
                "pitch": "" if ability is None else ability.pitch,
                "benefit": "" if ability is None else ability.benefit,
                "preview": [] if ability is None else [{"speaker": line.speaker, "text": line.text} for line in ability.preview],
            })
        return result

    async def status(self, key: str) -> dict[str, object]:
        registration = self.steps[key]
        async with registration.context.runtime_scope():
            return await registration.step.status()
