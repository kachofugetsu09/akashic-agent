from __future__ import annotations

import functools
import importlib.util
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
from agent.plugins.registry import MetadataKind, PluginEventType, plugin_registry
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
        # 4. 实例化并绑定生命周期 handlers 到 event_bus
        instance = cls()
        plugin_registry.register_instance(mp, instance)
        self._bind_handlers(instance, mp)
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

    def _bind_handlers(self, instance: Any, module_path: str) -> None:
        for md in plugin_registry.get_handlers_by_module_path(module_path):
            # 1. Phase 1 只绑定生命周期 handler，TOOL 类型留给后续 phase
            if md.kind != MetadataKind.LIFECYCLE:
                continue
            # 2. 跳过当前 phase 尚未支持的事件类型
            ctx_type = _EVENT_TYPE_MAP.get(md.event_type)  # type: ignore[arg-type]
            if ctx_type is None:
                continue
            # 3. 绑定 instance 为第一个参数，注册到 event_bus
            bound = functools.partial(md.handler, instance)
            self._event_bus.on(ctx_type, bound)
