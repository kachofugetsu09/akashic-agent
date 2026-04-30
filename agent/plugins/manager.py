from __future__ import annotations

import functools
import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import Any

from agent.lifecycle.types import (
    AfterReasoningCtx,
    AfterStepCtx,
    AfterTurnCtx,
    BeforeReasoningCtx,
    BeforeStepCtx,
    BeforeTurnCtx,
)
from agent.plugins.registry import HandlerType, MetadataKind, PluginEventType, plugin_registry
from bus.event_bus import EventBus

logger = logging.getLogger(__name__)

_EVENT_TYPE_MAP: dict[PluginEventType, type] = {
    PluginEventType.BEFORE_TURN: BeforeTurnCtx,
    PluginEventType.BEFORE_REASONING: BeforeReasoningCtx,
    PluginEventType.BEFORE_STEP: BeforeStepCtx,
    PluginEventType.AFTER_STEP: AfterStepCtx,
    PluginEventType.AFTER_REASONING: AfterReasoningCtx,
    PluginEventType.AFTER_TURN: AfterTurnCtx,
}


class PluginManager:
    def __init__(
        self,
        plugin_dirs: list[Path],
        *,
        event_bus: EventBus,
        tool_registry: Any = None,
    ) -> None:
        self._dirs = plugin_dirs
        self._event_bus = event_bus
        self._tool_registry = tool_registry
        self._loaded: set[str] = set()

    @property
    def loaded_count(self) -> int:
        return len(self._loaded)

    # 扫描所有 plugin_dirs，返回可加载的插件描述列表
    def discover(self) -> list[dict[str, str]]:
        mods: list[dict[str, str]] = []
        seen_names: set[str] = set()
        for d in self._dirs:
            if not d.is_dir():
                continue
            source = d.name
            for child in sorted(d.iterdir()):
                # 1. 跳过非目录和没有 plugin.py 的目录
                if not child.is_dir():
                    continue
                main = child / "plugin.py"
                if not main.exists():
                    continue
                # 2. 同名插件 first-wins，后续同名打 warning 跳过
                if child.name in seen_names:
                    logger.warning("插件名重复，跳过: %s (%s)", child.name, main)
                    continue
                seen_names.add(child.name)
                # 3. import_path 带上 source 避免不同目录同名插件覆盖 sys.modules
                mods.append({
                    "name": child.name,
                    "module_path": str(main),
                    "import_path": f"akasic_plugin_{source}_{child.name}",
                })
        return mods

    async def load_all(self) -> None:
        for mod in self.discover():
            await self._load_one(mod)

    async def _load_one(self, mod: dict[str, str]) -> None:
        mp = mod["import_path"]
        # 1. 幂等：已加载过直接跳过
        if mp in self._loaded:
            return
        # 2. 用 importlib 从文件路径加载，不依赖 sys.path
        try:
            self._import_plugin(mp, Path(mod["module_path"]))
        except Exception as e:
            logger.warning("插件 %s 导入失败: %s", mod["name"], e)
            return
        # 3. 导入触发 __init_subclass__，从 registry 取注册的类
        cls = plugin_registry._classes.get(mp)
        if cls is None:
            logger.warning("插件 %s 未注册类", mod["name"])
            return
        # 4. 实例化并绑定生命周期 handlers 到 event_bus，注册工具到 tool_registry
        instance = cls()
        plugin_registry.register_instance(mp, instance)
        self._bind_handlers(instance, mp)
        self._register_tools(instance, mp)
        # 5. 给插件机会做异步初始化
        if hasattr(instance, "initialize"):
            await instance.initialize()
        self._loaded.add(mp)
        logger.info("插件已加载: %s", mod["name"])

    def _import_plugin(self, module_name: str, path: Path) -> None:
        # 1. 从文件路径构建 ModuleSpec
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载插件文件: {path}")
        # 2. 先注册到 sys.modules 再执行，避免插件内部相对 import 找不到自身
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]

    def _register_tools(self, instance: Any, module_path: str) -> None:
        if self._tool_registry is None:
            return
        from agent.tools.base import Tool as AgentTool
        for md in plugin_registry.get_handlers_by_module_path(module_path):
            # 1. 只处理 TOOL 类型元数据
            if md.kind != MetadataKind.TOOL:
                continue
            bound = functools.partial(md.handler, instance, None)
            tool_name = md.tool_name or md.handler_name
            description = (md.handler.__doc__ or "").strip()
            schema = md.tool_schema or {"type": "object", "properties": {}, "required": []}
            # 2. 动态创建 Tool 子类并绑定 execute
            ToolCls = type(
                f"PluginTool_{tool_name}",
                (AgentTool,),
                {
                    "name": tool_name,
                    "description": description,
                    "parameters": schema,
                    "execute": _make_execute(bound),
                },
            )
            # 3. 注册到 ToolRegistry，标记来源为 plugin
            plugin_name = getattr(instance, "name", None) or module_path
            self._tool_registry.register(
                ToolCls(),
                risk=md.tool_risk or "read-write",
                always_on=bool(md.tool_always_on),
                search_hint=md.tool_search_hint,
                source_type="plugin",
                source_name=plugin_name,
            )
            logger.info("插件工具已注册: %s (来自 %s)", tool_name, plugin_name)

    def _bind_handlers(self, instance: Any, module_path: str) -> None:
        for md in plugin_registry.get_handlers_by_module_path(module_path):
            # 1. Phase 1 只绑定生命周期 handler，TOOL 类型留给后续 phase
            if md.kind != MetadataKind.LIFECYCLE:
                continue
            # 2. 跳过当前 phase 尚未支持的事件类型
            ctx_type = _EVENT_TYPE_MAP.get(md.event_type)  # type: ignore[arg-type]
            if ctx_type is None:
                continue
            # 3. 绑定 instance 为第一个参数，TAP handler 用工厂包装确保返回 None
            bound = functools.partial(md.handler, instance)
            if md.handler_type == HandlerType.TAP:
                self._event_bus.on(ctx_type, _make_tap_wrapper(bound))
            else:
                self._event_bus.on(ctx_type, bound)


def _make_tap_wrapper(bound: Any) -> Any:
    # TAP handler 只观测，不修改事件；包装确保返回 None，不干扰 fanout 语义
    async def wrapper(ctx: Any) -> None:
        await bound(ctx)
    return wrapper


def _make_execute(bound: Any) -> Any:
    # 预先提取插件函数接受的参数名（排除 self/event），用于过滤 Registry 注入的 context 字段
    sig = inspect.signature(bound)
    accepted = frozenset(
        name for name in sig.parameters if name not in ("self", "event")
    )

    # 工厂函数把 bound 和 accepted 锁进闭包，避免动态 type() 时 self 顶掉 bound
    async def execute(self: Any, **kwargs: Any) -> str:
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
        result = bound(**filtered)
        if inspect.isawaitable(result):
            result = await result
        return str(result)
    return execute
