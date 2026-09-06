from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from graphlib import TopologicalSorter

from agent.plugin_composition import Context, Effect, ServiceKey
from agent.plugin_composition.models import BoundChatModel, ModelRequest
from plugins.content.api import Reference
from session.message import ContentPart, Message

from .api import ContextModel, Materials, Summary, SummaryReducer

Prepare = Callable[[tuple[Message, ...], str], Awaitable[Materials]]


@dataclass(frozen=True, slots=True)
class _Source:
    prepare: Prepare
    after: tuple[str, ...]
    prompt: bool
    summary: bool
    plugin_id: str
    reduce: SummaryReducer | None


class MaterialView:
    """固定本次请求的贡献者；只收集材料，不调用模型或修改消息。"""

    def __init__(self, sources: tuple[_Source, ...]):
        self._sources = sources
        self._active = True

    def close(self) -> None:
        self._active = False

    def _check_active(self) -> None:
        if not self._active:
            raise RuntimeError("材料 view 已关闭")

    async def prepare(self, snapshot: tuple[Message, ...], source: str) -> Materials:
        """按显式依赖收集；摘要只能有一个 owner，冲突引用不能静默覆盖。"""
        self._check_active()
        prompts: list[str] = []
        context: list[ContentPart] = []
        summary: Summary | None = None
        references: dict[str, Reference] = {}
        for owner in self._sources:
            material = await owner.prepare(snapshot, source)
            self._check_active()
            if not isinstance(material, Materials):
                raise TypeError("材料 owner 必须返回 Materials")
            if material.system_prompt:
                if not owner.prompt:
                    raise PermissionError("此材料 owner 没有 Prompt 贡献权")
                prompts.append(material.system_prompt)
            context.extend(material.context)
            if material.summary is not None:
                if not owner.summary:
                    raise PermissionError("此材料 owner 没有摘要发布权")
                if summary is not None:
                    raise ValueError("摘要必须只有一个 owner")
                summary = material.summary
            for ref in material.references:
                previous = references.get(ref.ref)
                if previous is not None and previous != ref:
                    raise ValueError("同一引用的材料证据冲突")
                references[ref.ref] = ref
        return Materials("\n\n".join(prompts), tuple(context), summary, tuple(references.values()))

    async def reduce(
        self, snapshot: tuple[Message, ...], materials: Materials,
        request: ModelRequest, model: BoundChatModel, projection: ContextModel,
        *, source: str, force: bool,
    ) -> Summary | None:
        """只有同一个摘要 owner 能缩减；其余已取得材料保持原样。"""
        self._check_active()
        for owner in self._sources:
            if owner.reduce is not None:
                summary = await owner.reduce(snapshot, materials, request, model, projection, source=source, force=force)
                self._check_active()
                if summary is not None and not isinstance(summary, Summary):
                    raise TypeError("缩减必须返回已发布的 Summary")
                if summary is None:
                    return materials.summary
                previous = materials.summary
                if previous is not None:
                    if summary.reference == previous.reference and summary != previous:
                        raise ValueError("同一持久摘要引用的内容不能改变")
                    if summary.source_message_ids[:len(previous.source_message_ids)] != previous.source_message_ids:
                        raise ValueError("缩减不能撤回已覆盖的摘要来源")
                    if (summary.source_message_ids, summary.content) == (previous.source_message_ids, previous.content):
                        return previous
                return summary
        return materials.summary


class ContextMaterials:
    """普通材料注册和生命周期；Prompt 权由组合配置授予，检索默认低信任。"""

    def __init__(
        self, ctx: Context, *, prompt_sources: Mapping[str, str],
        summary_source: tuple[str, str] | None = None,
    ):
        self._ctx = ctx
        self._prompt_sources = dict(prompt_sources)
        self._summary_source = summary_source
        self._sources: dict[str, _Source] = {}

    async def register(
        self, ctx: Context, *, name: str, prepare: Prepare,
        after: tuple[str, ...] = (), prompt: bool = False,
        reduce: SummaryReducer | None = None,
    ) -> Effect:
        """同一名称只有一个真实注册 owner；不按安装顺序或数字 priority 合并。"""
        if ctx.root_instance_token is not self._ctx.root_instance_token:
            raise ValueError("材料注册不能跨 composition Root")
        if not isinstance(name, str) or not name or not callable(prepare):
            raise ValueError("材料必须有名称和 prepare 函数")
        if not isinstance(after, tuple) or any(not isinstance(key, str) or not key for key in after):
            raise ValueError("材料依赖必须是非空名称的 tuple")
        if type(prompt) is not bool:
            raise TypeError("Prompt 声明必须是 bool")
        plugin_id = ctx.runtime.plugin_id
        expected = self._prompt_sources.get(name)
        if expected is not None and plugin_id != expected:
            raise PermissionError(f"材料 {name} 只授予实际插件 {expected}")
        if prompt and expected != plugin_id:
            raise PermissionError(f"组合配置没有授予 {name} Prompt 贡献权")
        summary = self._summary_source == (name, plugin_id)
        if self._summary_source is not None and self._summary_source[0] == name and not summary:
            raise PermissionError(f"摘要材料 {name} 的实际插件不匹配")
        if reduce is not None:
            if not summary:
                raise PermissionError("只有获授的摘要材料 owner 能注册缩减")
            if not callable(reduce):
                raise TypeError("摘要缩减必须可调用")

        def setup():
            if name in self._sources:
                raise ValueError(f"材料 owner 重复: {name}")
            self._sources[name] = _Source(prepare, after, prompt, summary, plugin_id, reduce)
            return lambda: self._sources.pop(name)

        return await ctx.effect(setup, label=f"materials:{name}")

    @asynccontextmanager
    async def bind(self, *, exclude: frozenset[str] = frozenset()) -> AsyncIterator[MaterialView]:
        """调用程序明确选择材料；持有原 Root 到请求提交，排除者不会执行。"""
        async with self._ctx.runtime_scope():
            sources = {name: source for name, source in self._sources.items() if name not in exclude}
            for name, plugin_id in self._prompt_sources.items():
                if name in exclude:
                    continue
                source = sources.get(name)
                if source is None or not source.prompt or source.plugin_id != plugin_id:
                    raise ValueError(f"获授的 Prompt 材料未就绪: {name}")
            if self._summary_source is not None and self._summary_source[0] not in exclude:
                name, plugin_id = self._summary_source
                source = sources.get(name)
                if source is None or not source.summary or source.plugin_id != plugin_id:
                    raise ValueError(f"获授的摘要材料未就绪: {name}")
            graph = {key: sources[key].after for key in sorted(sources)}
            missing = {dep for deps in graph.values() for dep in deps} - sources.keys()
            if missing:
                raise ValueError(f"材料依赖缺失: {sorted(missing)}")
            order = tuple(TopologicalSorter(graph).static_order())
            view = MaterialView(tuple(sources[key] for key in order))
            try:
                yield view
            finally:
                view.close()


MATERIALS = ServiceKey[ContextMaterials]("context.materials.v1")
