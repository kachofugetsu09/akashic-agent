from __future__ import annotations

import functools
import importlib.util
import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Any, cast

from agent.lifecycle.types import (
    AfterReasoningCtx,
    AfterStepCtx,
    AfterToolResultCtx,
    AfterTurnCtx,
    BeforeReasoningCtx,
    BeforeStepCtx,
    BeforeToolCallCtx,
    BeforeTurnCtx,
    PreToolCtx,
)
from agent.plugins.registry import MetadataKind, PluginEventType, plugin_registry
from agent.tool_hooks.base import ToolHook
from agent.tool_hooks.types import HookContext, HookOutcome
from bus.event_bus import EventBus

logger = logging.getLogger(__name__)

_EVENT_TYPE_MAP: dict[PluginEventType, type] = {
    PluginEventType.BEFORE_TURN: BeforeTurnCtx,
    PluginEventType.BEFORE_REASONING: BeforeReasoningCtx,
    PluginEventType.BEFORE_STEP: BeforeStepCtx,
    PluginEventType.AFTER_STEP: AfterStepCtx,
    PluginEventType.AFTER_REASONING: AfterReasoningCtx,
    PluginEventType.AFTER_TURN: AfterTurnCtx,
    PluginEventType.BEFORE_TOOL_CALL: BeforeToolCallCtx,
    PluginEventType.AFTER_TOOL_RESULT: AfterToolResultCtx,
}


class PluginManager:
    def __init__(
        self,
        plugin_dirs: list[Path],
        *,
        event_bus: EventBus,
        tool_registry: Any = None,
        observe_db_path: Path | None = None,
    ) -> None:
        self._dirs = plugin_dirs
        self._event_bus = event_bus
        self._tool_registry = tool_registry
        self._observe_db_path = observe_db_path
        self._loaded: set[str] = set()
        self._tool_hooks: list[ToolHook] = []
        self._before_turn_modules_early: list[object] = []
        self._before_turn_modules_late: list[object] = []

    @property
    def loaded_count(self) -> int:
        return len(self._loaded)

    @property
    def tool_hooks(self) -> list[object]:
        return list(self._tool_hooks)

    @property
    def before_turn_modules_early(self) -> list[object]:
        return list(self._before_turn_modules_early)

    @property
    def before_turn_modules_late(self) -> list[object]:
        return list(self._before_turn_modules_late)

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
        # 4. 实例化，读 manifest 覆盖元信息，注入 PluginContext
        instance = cls()
        plugin_dir = Path(mod["module_path"]).parent
        _apply_manifest(instance, plugin_dir)
        plugin_id = str(instance.name) if instance.name else mod["name"]
        plugin_config = _load_plugin_config(plugin_dir)
        from agent.plugins.context import PluginContext, PluginKVStore
        instance.context = PluginContext(  # type: ignore[attr-defined]
            event_bus=self._event_bus,
            tool_registry=self._tool_registry,
            plugin_id=plugin_id,
            plugin_dir=plugin_dir,
            kv_store=PluginKVStore(plugin_dir / ".kv.json"),
            config=plugin_config,
            observe_db_path=self._observe_db_path,
        )
        plugin_registry.register_instance(mp, instance)
        self._bind_handlers(instance, mp)
        tool_names = self._register_tools(instance, mp)
        hook_count_before = len(self._tool_hooks)
        self._bind_tool_hooks(instance, mp)
        early_count_before = len(self._before_turn_modules_early)
        late_count_before = len(self._before_turn_modules_late)
        self._collect_before_turn_modules(instance)
        # 5. 给插件机会做异步初始化；失败时回滚所有注册
        try:
            if hasattr(instance, "initialize"):
                await instance.initialize()
        except Exception as e:
            logger.warning("插件 %s 初始化失败，回滚: %s", mod["name"], e)
            plugin_registry.remove_plugin(mp)
            for tn in tool_names:
                if self._tool_registry is not None:
                    self._tool_registry.unregister(tn)
            del self._tool_hooks[hook_count_before:]
            del self._before_turn_modules_early[early_count_before:]
            del self._before_turn_modules_late[late_count_before:]
            return
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

    def _register_tools(self, instance: Any, module_path: str) -> list[str]:
        tool_names: list[str] = []
        if self._tool_registry is None:
            return tool_names
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
            tool_names.append(tool_name)
            logger.info("插件工具已注册: %s (来自 %s)", tool_name, plugin_name)
        return tool_names

    def _bind_handlers(self, instance: Any, module_path: str) -> None:
        for md in plugin_registry.get_handlers_by_module_path(module_path):
            # 1. Phase 1 只绑定生命周期 handler，TOOL 类型留给后续 phase
            if md.kind != MetadataKind.LIFECYCLE:
                continue
            # 2. 跳过当前 phase 尚未支持的事件类型
            ctx_type = _EVENT_TYPE_MAP.get(md.event_type)  # type: ignore[arg-type]
            if ctx_type is None:
                continue
            # 3. 绑定 instance 为第一个参数，EventBus 已处理 sync/async，直接注册
            bound = functools.partial(md.handler, instance)
            self._event_bus.on(ctx_type, bound)

    def _bind_tool_hooks(self, instance: Any, module_path: str) -> None:
        for md in plugin_registry.get_handlers_by_module_path(module_path):
            if md.kind != MetadataKind.TOOL_HOOK:
                continue
            bound = functools.partial(md.handler, instance)
            hook = _PluginToolHook(
                name=f"plugin:{getattr(instance, 'name', module_path)}:{md.handler_name}",
                handler=bound,
                tool_name_filter=md.hook_tool_name,
            )
            self._tool_hooks.append(hook)
            logger.info("插件 tool hook 已注册: %s", hook.name)

    def _collect_before_turn_modules(self, instance: Any) -> None:
        self._before_turn_modules_early.extend(
            _load_module_list(instance, "before_turn_modules_early")
        )
        self._before_turn_modules_late.extend(
            _load_module_list(instance, "before_turn_modules_late")
        )

    async def terminate_all(self) -> None:
        for mp in list(self._loaded):
            instance = plugin_registry.get_instance(mp)
            if instance is not None and hasattr(instance, "terminate"):
                try:
                    await instance.terminate()
                except Exception as e:
                    logger.warning("插件 terminate 失败 (%s): %s", mp, e)
            # 注销工具
            for md in plugin_registry.get_handlers_by_module_path(mp):
                if md.kind == MetadataKind.TOOL and self._tool_registry is not None:
                    self._tool_registry.unregister(md.tool_name or md.handler_name)
            plugin_registry.remove_plugin(mp)
        self._loaded.clear()
        self._tool_hooks.clear()
        self._before_turn_modules_early.clear()
        self._before_turn_modules_late.clear()


def _load_plugin_config(plugin_dir: Path) -> "Any":
    # 1. 读取 _conf_schema.json，提取每个字段的 default 值
    from agent.plugins.config import PluginConfig
    schema_path = plugin_dir / "_conf_schema.json"
    if not schema_path.exists():
        return None
    try:
        loaded = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("_conf_schema.json 读取失败 (%s): %s", plugin_dir, e)
        return None
    if not isinstance(loaded, dict):
        logger.warning("_conf_schema.json 格式错误，期望 dict (%s)", plugin_dir)
        return None
    raw: dict[str, object] = cast("dict[str, object]", loaded)
    values: dict[str, Any] = {}
    for key, spec in raw.items():
        if not isinstance(key, str):
            continue
        if not isinstance(spec, dict):
            continue
        if "default" in spec:
            values[key] = spec["default"]
    # 2. 读取 plugin_config.json，用户级覆盖默认值
    override_path = plugin_dir / "plugin_config.json"
    if override_path.exists():
        try:
            override = json.loads(override_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("plugin_config.json 读取失败 (%s): %s", plugin_dir, e)
        else:
            if isinstance(override, dict):
                raw_override: dict[str, object] = cast("dict[str, object]", override)
                for key, value in raw_override.items():
                    if not isinstance(key, str):
                        continue
                    values[key] = value
            else:
                logger.warning("plugin_config.json 格式错误，期望 dict (%s)", plugin_dir)
    return PluginConfig(values)


def _load_module_list(instance: Any, method_name: str) -> list[object]:
    provider = getattr(instance, method_name, None)
    if provider is None:
        return []
    if not callable(provider):
        logger.warning("插件 %s.%s 不是可调用对象", type(instance).__name__, method_name)
        return []
    try:
        loaded = provider()
    except Exception as e:
        logger.warning("插件 %s.%s 加载失败: %s", type(instance).__name__, method_name, e)
        return []
    if loaded is None:
        return []
    if not isinstance(loaded, list):
        logger.warning("插件 %s.%s 返回值不是 list", type(instance).__name__, method_name)
        return []
    return loaded


_MANIFEST_FIELDS = ("name", "version", "desc", "author")


def _apply_manifest(instance: Any, plugin_dir: Path) -> None:
    manifest_path = plugin_dir / "manifest.yaml"
    if not manifest_path.exists():
        return
    try:
        import yaml
        loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("manifest.yaml 读取失败 (%s): %s", plugin_dir, e)
        return
    if not isinstance(loaded, dict):
        logger.warning("manifest.yaml 格式错误，期望 dict (%s)", plugin_dir)
        return
    raw: dict[str, object] = cast("dict[str, object]", loaded)
    # 逐字段覆盖实例属性，非字符串值转 str，缺失字段跳过
    for field in _MANIFEST_FIELDS:
        val = raw.get(field)
        if val is not None:
            setattr(instance, field, str(val))


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


class _PluginToolHook(ToolHook):
    """将插件的 @on_tool_pre handler 适配为 ToolExecutor 的 ToolHook 接口。"""

    event = "pre_tool_use"

    def __init__(
        self,
        name: str,
        handler: Any,
        tool_name_filter: str | None = None,
    ) -> None:
        self.name = name
        self._handler = handler
        self._tool_name_filter = tool_name_filter

    def matches(self, ctx: HookContext) -> bool:
        if self._tool_name_filter is None:
            return True
        return ctx.request.tool_name == self._tool_name_filter

    async def run(self, ctx: HookContext) -> HookOutcome:
        # 1. 构造 PreToolCtx（复制 arguments，避免插件直接改原对象）
        event = PreToolCtx(
            session_key=ctx.request.session_key,
            channel=ctx.request.channel,
            chat_id=ctx.request.chat_id,
            tool_name=ctx.request.tool_name,
            arguments=dict(ctx.current_arguments),
        )
        # 2. 调插件 handler，返回值决定行为
        result = self._handler(event)
        if inspect.isawaitable(result):
            result = await result
        # 3. None → 不改参；dict → 新 arguments
        if result is None:
            return HookOutcome()
        if isinstance(result, dict):
            return HookOutcome(updated_input=cast("dict[str, Any]", result))
        return HookOutcome()
