from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass

from agent.plugin_composition import Context, Effect, ServiceKey
from agent.plugin_composition.models import BoundChatModel, ModelRequest
from agent.plugin_contracts import Message

from .api import (
    ContextModel,
    MaterialData,
    Materials,
    Reminder,
    Summary,
    SummaryData,
    SummaryReducer,
    decode_material,
    decode_summary,
    material_data,
)

Prepare = Callable[[tuple[Message, ...], str], Awaitable[MaterialData]]


@dataclass(frozen=True, slots=True)
class _Source:
    prepare: Prepare
    priority: int
    prompt: bool
    summary: bool
    context: Context
    reduce: SummaryReducer | None

    @property
    def plugin_id(self) -> str:
        return self.context.runtime.plugin_id


class MaterialView:
    """固定本次请求的贡献者；只收集材料，不调用模型或修改消息。"""

    def __init__(self, ctx: Context, sources: tuple[tuple[str, _Source], ...]):
        self._ctx = ctx
        self._sources = sources
        self._active = True

    def close(self) -> None:
        self._active = False

    def _check_active(self) -> None:
        if not self._active:
            raise RuntimeError("材料 view 已关闭")

    async def prepare(
        self, snapshot: tuple[Message, ...], source: str, *,
        caller: Context | None = None, reminders: tuple[Mapping[str, object], ...] = (),
    ) -> MaterialData:
        """按固定贡献者收集；同优先级按实际插件 ID 和块名称的 UTF-8 字节排序。"""
        self._check_active()
        prompts: list[tuple[int, bytes, bytes, str]] = []
        blocks: dict[tuple[str, str], Reminder] = {}

        def collect(plugin_id: str, items: tuple[Reminder, ...]) -> None:
            for item in items:
                key = (plugin_id, item.name)
                if key in blocks:
                    raise ValueError(f"提醒身份重复: {key}")
                blocks[key] = item

        if reminders:
            if caller is None or caller.root_instance_token is not self._ctx.root_instance_token:
                raise ValueError("调用程序的提醒需要同一 Root 的实际 Context owner")
            collect(caller.runtime.plugin_id, decode_material({"reminders": reminders}).reminders)
        summary: Summary | None = None
        references: dict[str, Mapping[str, object]] = {}
        for name, owner in self._sources:
            material = decode_material(await owner.prepare(snapshot, source))
            self._check_active()
            if material.system_prompt:
                if not owner.prompt:
                    raise PermissionError("此材料 owner 没有 Prompt 贡献权")
                prompts.append((owner.priority, owner.plugin_id.encode("utf-8"), name.encode("utf-8"), material.system_prompt))
            collect(owner.plugin_id, material.reminders)
            if material.summary is not None:
                if not owner.summary:
                    raise PermissionError("此材料 owner 没有摘要发布权")
                if summary is not None:
                    raise ValueError("摘要必须只有一个 owner")
                summary = material.summary
            for ref in material.references:
                identity = ref.get("ref")
                if not isinstance(identity, str) or not identity:
                    raise ValueError("引用必须包含非空 ref 字符串")
                previous = references.get(identity)
                if previous is not None and dict(previous) != dict(ref):
                    raise ValueError("同一引用的材料证据冲突")
                references[identity] = ref
        ordered = tuple(block for _, block in sorted(
            blocks.items(), key=lambda item: (item[1].priority, item[0][0].encode("utf-8"), item[0][1].encode("utf-8")),
        ))
        return material_data(Materials("\n\n".join(item[3] for item in sorted(prompts)), ordered, summary, tuple(references.values())))

    async def reduce(
        self, snapshot: tuple[Message, ...], materials: MaterialData,
        request: ModelRequest, model: BoundChatModel, projection: ContextModel,
        *, source: str, force: bool,
    ) -> SummaryData | None:
        """只有同一个摘要 owner 能缩减；其余已取得材料保持原样。"""
        self._check_active()
        current = decode_material(materials)
        for _, owner in self._sources:
            if owner.reduce is not None:
                summary_value = await owner.reduce(
                    snapshot, material_data(current), request, model, projection,
                    source=source, force=force,
                )
                self._check_active()
                summary = decode_summary(summary_value)
                if summary is None:
                    return None
                previous = current.summary
                if previous is not None:
                    if summary.reference == previous.reference and summary != previous:
                        raise ValueError("同一持久摘要引用的内容不能改变")
                    if summary.source_message_ids[:len(previous.source_message_ids)] != previous.source_message_ids:
                        raise ValueError("缩减不能撤回已覆盖的摘要来源")
                    if (summary.source_message_ids, summary.content) == (previous.source_message_ids, previous.content):
                        return None
                return {
                    "reference": summary.reference,
                    "source_message_ids": summary.source_message_ids,
                    "content": summary.content,
                }
        return None


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
        priority: int = 0, prompt: bool = False,
        reduce: SummaryReducer | None = None,
    ) -> Effect:
        """同一名称只有一个真实注册 owner；priority 只排序，不表示依赖或权限。"""
        if ctx.root_instance_token is not self._ctx.root_instance_token:
            raise ValueError("材料注册不能跨 composition Root")
        if not isinstance(name, str) or not name or not callable(prepare):
            raise ValueError("材料必须有名称和 prepare 函数")
        if type(priority) is not int:
            raise TypeError("材料 priority 必须是整数")
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
            self._sources[name] = _Source(prepare, priority, prompt, summary, ctx, reduce)
            return lambda: self._sources.pop(name)

        return await ctx.effect(setup, label=f"materials:{name}")

    def binding_contributors(self) -> tuple[Context, ...]:
        """材料目录的实际注册者随同服务归档；来源仍在 bind 时选择排除项。"""
        return tuple(source.context for source in self._sources.values())

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
            order = sorted(sources, key=lambda key: (
                sources[key].plugin_id.encode("utf-8"), key.encode("utf-8"),
            ))
            view = MaterialView(self._ctx, tuple((key, sources[key]) for key in order))
            try:
                yield view
            finally:
                view.close()


MATERIALS = ServiceKey[ContextMaterials]("context.materials.v3")
