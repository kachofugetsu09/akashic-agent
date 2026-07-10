from __future__ import annotations

import functools
import importlib.util
import inspect
import logging
import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from agent.plugins.manifest import load_plugin_manifest, write_plugin_manifest
from agent.plugins.specs import (
    McpServerSpec,
    ProactiveSourceSpec,
    RegisteredProactiveSource,
)
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
    PromptRenderCtx,
)
from agent.plugins.registry import MetadataKind, PluginEventType, plugin_registry
from agent.plugins.source_resolver import resolve_plugin_sources
from agent.plugins.jobs import PluginJobSpec, PluginLlmService, RegisteredPluginJob
from agent.tool_hooks.base import ToolHook
from agent.tool_hooks.types import HookContext, HookOutcome
from bus.event_bus import EventBus
from infra.channels.contract import Channel

logger = logging.getLogger(__name__)

_EVENT_TYPE_MAP: dict[PluginEventType, type] = {
    PluginEventType.BEFORE_TURN: BeforeTurnCtx,
    PluginEventType.BEFORE_REASONING: BeforeReasoningCtx,
    PluginEventType.PROMPT_RENDER: PromptRenderCtx,
    PluginEventType.BEFORE_STEP: BeforeStepCtx,
    PluginEventType.AFTER_STEP: AfterStepCtx,
    PluginEventType.AFTER_REASONING: AfterReasoningCtx,
    PluginEventType.AFTER_TURN: AfterTurnCtx,
    PluginEventType.BEFORE_TOOL_CALL: BeforeToolCallCtx,
    PluginEventType.AFTER_TOOL_RESULT: AfterToolResultCtx,
}


@dataclass(frozen=True)
class ActivePluginInfo:
    plugin_id: str
    plugin_dir: Path
    manifest: dict[str, object]
    module_path: str
    declares_aka_plugin: bool = True
    skill_roots: tuple[Path, ...] = ()
    drift_skill_roots: tuple[Path, ...] = ()
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)


class PluginManager:
    def __init__(
        self,
        plugin_dirs: list[Path],
        *,
        event_bus: EventBus,
        tool_registry: Any = None,
        workspace: Path | None = None,
        session_manager: Any = None,
        memory_engine: Any = None,
        llm: PluginLlmService | None = None,
        installed_cache_root: Path | None = None,
    ) -> None:
        self._dirs = plugin_dirs
        self._event_bus = event_bus
        self._tool_registry = tool_registry
        self._workspace = workspace
        self._session_manager = session_manager
        self._memory_engine = memory_engine
        self._llm = llm
        self._installed_cache_root = installed_cache_root
        self._loaded: set[str] = set()
        self._channels: list[Channel] = []
        self._tool_hooks: list[ToolHook] = []
        self._before_turn_modules: list[object] = []
        self._before_reasoning_modules: list[object] = []
        self._prompt_render_modules: list[object] = []
        self._before_step_modules: list[object] = []
        self._after_step_modules: list[object] = []
        self._after_reasoning_modules: list[object] = []
        self._after_turn_modules: list[object] = []
        self._proactive_modules: list[object] = []
        self._proactive_lifecycles: list[object] = []
        self._proactive_module_factories: list[object] = []
        self._proactive_runtime_factories: list[object] = []
        self._proactive_sources: list[RegisteredProactiveSource] = []
        self._jobs: list[RegisteredPluginJob] = []
        self._active_plugins: dict[str, ActivePluginInfo] = {}

    @property
    def loaded_count(self) -> int:
        return len(self._loaded)

    @property
    def tool_hooks(self) -> list[ToolHook]:
        return list(self._tool_hooks)

    @property
    def channels(self) -> list[Channel]:
        return list(self._channels)

    @property
    def before_turn_modules(self) -> list[object]:
        return list(self._before_turn_modules)

    @property
    def before_reasoning_modules(self) -> list[object]:
        return list(self._before_reasoning_modules)

    @property
    def prompt_render_modules(self) -> list[object]:
        return list(self._prompt_render_modules)

    @property
    def before_step_modules(self) -> list[object]:
        return list(self._before_step_modules)

    @property
    def after_step_modules(self) -> list[object]:
        return list(self._after_step_modules)

    @property
    def after_reasoning_modules(self) -> list[object]:
        return list(self._after_reasoning_modules)

    @property
    def after_turn_modules(self) -> list[object]:
        return list(self._after_turn_modules)

    @property
    def proactive_modules(self) -> list[object]:
        return list(self._proactive_modules)

    @property
    def proactive_lifecycles(self) -> list[object]:
        return list(self._proactive_lifecycles)

    @property
    def proactive_module_factories(self) -> list[object]:
        return list(self._proactive_module_factories)

    @property
    def proactive_runtime_factories(self) -> list[object]:
        return list(self._proactive_runtime_factories)

    @property
    def proactive_sources(self) -> list[RegisteredProactiveSource]:
        return list(self._proactive_sources)

    @property
    def jobs(self) -> list[RegisteredPluginJob]:
        return list(self._jobs)

    @property
    def llm(self) -> PluginLlmService | None:
        return self._llm

    @property
    def plugin_dirs(self) -> list[Path]:
        return list(self._dirs)

    def active_plugins(self) -> list[ActivePluginInfo]:
        return [
            plugin
            for module_path, plugin in self._active_plugins.items()
            if self._registry_active(module_path)
        ]

    def sync_manifest(self, *, plugins_home: Path | None = None) -> Path:
        entries = load_plugin_manifest(plugins_home)
        for mod in self.discover():
            entries.setdefault(_resolve_plugin_id(mod), True)
        return write_plugin_manifest(entries, plugins_home=plugins_home)

    def _registry_active(self, module_path: str) -> bool:
        if module_path not in self._active_plugins:
            return False
        instance = plugin_registry.get_instance(module_path)
        if instance is None:
            return True
        checker = getattr(instance, "is_active", None)
        if not callable(checker):
            return True
        try:
            return bool(checker())
        except Exception as e:
            logger.warning("插件 active 状态检查失败 (%s): %s", module_path, e)
            return True

    @property
    def telegram_bot_commands(self) -> list[tuple[str, str]]:
        commands: list[tuple[str, str]] = []
        for module_path in self._loaded:
            instance = plugin_registry.get_instance(module_path)
            if instance is None:
                continue
            getter = getattr(instance, "telegram_bot_commands", None)
            if getter is None:
                continue
            typed_getter = cast(Callable[[], list[tuple[str, str]]], getter)
            for command, description in typed_getter():
                commands.append((str(command), str(description)))
        return commands

    # 扫描所有 plugin_dirs，返回可加载的插件描述列表
    def discover(self) -> list[dict[str, str]]:
        mods: list[dict[str, str]] = []
        seen_names: set[str] = set()
        for source in resolve_plugin_sources(
            self._dirs,
            installed_cache_root=self._installed_cache_root,
        ):
            name = source.plugin_root.parent.name if source.source_type == "installed" else source.plugin_root.name
            if name in seen_names and source.source_type == "builtin":
                logger.warning("插件名重复，跳过: %s (%s)", name, source.plugin_root)
                continue
            seen_names.add(name)
            import_suffix = name.replace("-", "_").replace("@", "_")
            import_source = source.marketplace or source.plugin_root.parent.name
            module_path = source.plugin_root / "plugin.py"
            mods.append({
                "name": name,
                "plugin_root": str(source.plugin_root),
                "module_path": str(module_path) if module_path is not None else "",
                "import_path": f"akasic_plugin_{import_source}_{import_suffix}",
                "marketplace": source.marketplace,
                "source_type": source.source_type,
            })
        return mods

    async def load_all(self) -> None:
        for mod in self.discover():
            await self._load_one(mod)

    async def _load_one(self, mod: dict[str, str]) -> None:
        mp = mod["import_path"]
        plugin_dir = Path(mod["plugin_root"])
        initial_plugin_id = _resolve_plugin_id(mod)
        # 1. 幂等：已加载过直接跳过
        if mp in self._loaded:
            return
        plugin_manifest = load_plugin_manifest(_plugins_home(self._installed_cache_root))
        if plugin_manifest.get(initial_plugin_id, True) is False:
            logger.info("插件已禁用（manifest.toml）: %s", initial_plugin_id)
            return
        # 1b. 本地禁用标记存在时跳过
        if _is_plugin_disabled(plugin_dir):
            logger.info("插件已禁用（plugin.disabled）: %s", mod["name"])
            return
        instance = None
        tool_names: list[str] = []
        hook_count_before = len(self._tool_hooks)
        before_turn_count_before = len(self._before_turn_modules)
        before_reasoning_count_before = len(self._before_reasoning_modules)
        prompt_render_count_before = len(self._prompt_render_modules)
        before_step_count_before = len(self._before_step_modules)
        after_step_count_before = len(self._after_step_modules)
        after_reasoning_count_before = len(self._after_reasoning_modules)
        after_turn_count_before = len(self._after_turn_modules)
        proactive_module_count_before = len(self._proactive_modules)
        proactive_lifecycle_count_before = len(self._proactive_lifecycles)
        proactive_factory_count_before = len(self._proactive_module_factories)
        proactive_runtime_factory_count_before = len(self._proactive_runtime_factories)
        proactive_source_count_before = len(self._proactive_sources)
        job_count_before = len(self._jobs)
        module_path = mod["module_path"].strip()
        if module_path:
            try:
                self._import_plugin(mp, Path(module_path))
            except Exception as e:
                logger.warning("插件 %s 导入失败: %s", mod["name"], e)
                return
            cls = plugin_registry._classes.get(mp)
            if cls is None:
                logger.warning("插件 %s 未注册类", mod["name"])
                return
            instance = cls()
            name = str(instance.name or mod["name"]).strip()
            if not name:
                logger.warning("插件 %s 缺少 name", mod["name"])
                return
            plugin_id = f"{name}@{mod['marketplace']}" if mod["marketplace"] else name
            if plugin_id != initial_plugin_id:
                raise RuntimeError(
                    f"插件目录身份与声明不一致: directory={initial_plugin_id} declared={plugin_id}"
                )
            data_dir = _resolve_plugin_data_dir(name, mod, self._installed_cache_root)
            try:
                plugin_config = _load_plugin_config(
                    data_dir,
                    getattr(cls, "ConfigModel", None),
                )
            except _PluginConfigError as e:
                logger.warning("插件 %s 配置无效，跳过: %s", mod["name"], e)
                return
            from agent.plugins.context import PluginContext, PluginKVStore
            instance.context = PluginContext(  # type: ignore[attr-defined]
                event_bus=self._event_bus,
                tool_registry=self._tool_registry,
                plugin_id=plugin_id,
                plugin_dir=plugin_dir,
                data_dir=data_dir,
                kv_store=PluginKVStore(data_dir / ".kv.json"),
                config=plugin_config,
                workspace=self._workspace,
                session_manager=self._session_manager,
                memory_engine=self._memory_engine,
                llm=self._llm,
            )
            plugin_registry.register_instance(mp, instance)
            self._bind_handlers(instance, mp)
            tool_names = self._register_tools(instance, mp)
            self._bind_tool_hooks(instance, mp)
            self._collect_before_turn_modules(instance)
            self._collect_before_reasoning_modules(instance)
            self._collect_prompt_render_modules(instance)
            self._collect_before_step_modules(instance)
            self._collect_after_step_modules(instance)
            self._collect_after_reasoning_modules(instance)
            self._collect_after_turn_modules(instance)
            self._collect_proactive_modules(instance)
            self._collect_proactive_lifecycles(instance)
            self._collect_proactive_module_factories(instance)
            self._collect_proactive_runtime_factories(instance)
            self._collect_proactive_sources(instance, plugin_id)
            self._collect_jobs(instance, plugin_id)
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
                del self._before_turn_modules[before_turn_count_before:]
                del self._before_reasoning_modules[before_reasoning_count_before:]
                del self._prompt_render_modules[prompt_render_count_before:]
                del self._before_step_modules[before_step_count_before:]
                del self._after_step_modules[after_step_count_before:]
                del self._after_reasoning_modules[after_reasoning_count_before:]
                del self._after_turn_modules[after_turn_count_before:]
                del self._proactive_modules[proactive_module_count_before:]
                del self._proactive_lifecycles[proactive_lifecycle_count_before:]
                del self._proactive_module_factories[proactive_factory_count_before:]
                del self._proactive_runtime_factories[proactive_runtime_factory_count_before:]
                del self._proactive_sources[proactive_source_count_before:]
                del self._jobs[job_count_before:]
                return
            manifest: dict[str, object] = {
                "name": name,
                "version": str(instance.version or ""),
                "desc": str(instance.desc or ""),
                "author": str(instance.author or ""),
            }
            skill_roots = _resolve_declared_roots(plugin_dir, cls.skill_roots())
            drift_skill_roots = _resolve_declared_roots(plugin_dir, cls.drift_skill_roots())
            mcp_servers = _resolve_mcp_servers(plugin_dir, data_dir, cls.mcp_servers())
        else:
            raise RuntimeError(f"插件缺少 plugin.py: {plugin_dir}")
        self._loaded.add(mp)
        self._active_plugins[mp] = ActivePluginInfo(
            plugin_id=plugin_id,
            plugin_dir=plugin_dir,
            manifest=manifest,
            module_path=mp,
            skill_roots=skill_roots,
            drift_skill_roots=drift_skill_roots,
            mcp_servers=mcp_servers,
        )
        if instance is not None:
            self._collect_channels(instance)
        logger.info("插件已加载: %s", mod["name"])

    def _import_plugin(self, module_name: str, path: Path) -> None:
        # 1. 把 plugin.py 当成包入口加载，允许数字前缀目录里的插件使用相对 import。
        spec = importlib.util.spec_from_file_location(
            module_name,
            path,
            submodule_search_locations=[str(path.parent)],
        )
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
        self._collect_phase_modules(
            instance,
            "before_turn_modules",
            self._before_turn_modules,
        )

    def _collect_before_reasoning_modules(self, instance: Any) -> None:
        self._collect_phase_modules(
            instance,
            "before_reasoning_modules",
            self._before_reasoning_modules,
        )

    def _collect_prompt_render_modules(self, instance: Any) -> None:
        self._collect_phase_modules(
            instance,
            "prompt_render_modules",
            self._prompt_render_modules,
        )

    def _collect_before_step_modules(self, instance: Any) -> None:
        self._collect_phase_modules(
            instance,
            "before_step_modules",
            self._before_step_modules,
        )

    def _collect_after_step_modules(self, instance: Any) -> None:
        self._collect_phase_modules(
            instance,
            "after_step_modules",
            self._after_step_modules,
        )

    def _collect_after_reasoning_modules(self, instance: Any) -> None:
        self._collect_phase_modules(
            instance,
            "after_reasoning_modules",
            self._after_reasoning_modules,
        )

    def _collect_after_turn_modules(self, instance: Any) -> None:
        self._collect_phase_modules(
            instance,
            "after_turn_modules",
            self._after_turn_modules,
        )

    def _collect_proactive_modules(self, instance: Any) -> None:
        self._collect_phase_modules(
            instance,
            "proactive_modules",
            self._proactive_modules,
        )

    def _collect_proactive_lifecycles(self, instance: Any) -> None:
        self._collect_phase_modules(
            instance,
            "proactive_lifecycles",
            self._proactive_lifecycles,
        )

    def _collect_proactive_module_factories(self, instance: Any) -> None:
        self._collect_phase_modules(
            instance,
            "proactive_module_factories",
            self._proactive_module_factories,
        )

    def _collect_proactive_runtime_factories(self, instance: Any) -> None:
        self._collect_phase_modules(
            instance,
            "proactive_runtime_factories",
            self._proactive_runtime_factories,
        )

    def _collect_proactive_sources(self, instance: Any, plugin_id: str) -> None:
        seen = {source.spec.id for source in self._proactive_sources if source.plugin_id == plugin_id}
        for source in _load_module_list(instance, "proactive_sources"):
            if not isinstance(source, ProactiveSourceSpec):
                raise RuntimeError(
                    f"插件 {plugin_id}.proactive_sources 返回值不是 ProactiveSourceSpec"
                )
            if not source.id or source.id in seen:
                raise RuntimeError(f"插件 {plugin_id} proactive source id 重复或为空: {source.id!r}")
            if not source.channels or not source.server or not source.fetch_tool:
                raise RuntimeError(f"插件 {plugin_id} proactive source 声明不完整: {source.id}")
            seen.add(source.id)
            self._proactive_sources.append(
                RegisteredProactiveSource(plugin_id=plugin_id, spec=source)
            )

    def _collect_channels(self, instance: Any) -> None:
        for channel in _load_module_list(instance, "channels"):
            self._channels.append(cast(Channel, channel))

    def _collect_jobs(self, instance: Any, plugin_id: str) -> None:
        for spec in _load_module_list(instance, "jobs"):
            if not isinstance(spec, PluginJobSpec):
                logger.warning("插件 %s.jobs 返回值不是 PluginJobSpec", type(instance).__name__)
                continue
            job_id = str(getattr(spec, "id", "") or "").strip()
            if not job_id:
                logger.warning("插件 %s.jobs 返回了缺少 id 的任务", type(instance).__name__)
                continue
            self._jobs.append(
                RegisteredPluginJob(
                    plugin_id=plugin_id,
                    plugin_context=instance.context,
                    spec=spec,
                )
            )

    def _collect_phase_modules(
        self,
        instance: Any,
        attr_name: str,
        target: list[object],
    ) -> None:
        target.extend(_load_module_list(instance, attr_name))

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
            _ = self._active_plugins.pop(mp, None)
        self._loaded.clear()
        self._active_plugins.clear()
        self._tool_hooks.clear()
        self._before_turn_modules.clear()
        self._before_reasoning_modules.clear()
        self._prompt_render_modules.clear()
        self._before_step_modules.clear()
        self._after_step_modules.clear()
        self._after_reasoning_modules.clear()
        self._after_turn_modules.clear()
        self._proactive_modules.clear()
        self._proactive_lifecycles.clear()
        self._proactive_module_factories.clear()
        self._proactive_runtime_factories.clear()
        self._proactive_sources.clear()
        self._jobs.clear()
        self._channels.clear()


class _PluginConfigError(Exception):
    pass


def _load_plugin_config(
    data_dir: Path,
    config_model: type[BaseModel] | None = None,
) -> Any:
    config_path = data_dir / "config.local.toml"
    raw_config: dict[str, Any] = {}
    if config_path.exists():
        try:
            raw_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as e:
            raise _PluginConfigError(str(e)) from e
    if config_model is not None:
        try:
            return config_model.model_validate(raw_config)
        except ValidationError as e:
            raise _PluginConfigError(_format_validation_error(e)) from e
    from agent.plugins.config import PluginConfig
    return PluginConfig(raw_config) if raw_config else None


def _format_validation_error(error: ValidationError) -> str:
    parts: list[str] = []
    for item in error.errors():
        path = ".".join(str(part) for part in item.get("loc", ())) or "<root>"
        parts.append(f"{path}: {item.get('msg', 'invalid')}")
    return "; ".join(parts)


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


def _resolve_plugin_id(mod: dict[str, str]) -> str:
    name = mod["name"]
    marketplace = mod.get("marketplace", "").strip()
    if not marketplace:
        return name
    return f"{name}@{marketplace}"


def _resolve_plugin_data_dir(
    name: str,
    mod: dict[str, str],
    installed_cache_root: Path | None,
) -> Path:
    marketplace = mod.get("marketplace", "").strip()
    suffix = marketplace or "builtin"
    data_dir = _plugins_home(installed_cache_root) / "data" / f"{name}-{suffix}"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _plugins_home(installed_cache_root: Path | None) -> Path:
    if installed_cache_root is not None:
        return installed_cache_root.parent
    return Path.home() / ".akashic-plugin"


def _resolve_declared_roots(
    plugin_dir: Path,
    declared: tuple[str, ...],
) -> tuple[Path, ...]:
    roots: list[Path] = []
    for raw_path in declared:
        path = (plugin_dir / raw_path).resolve(strict=False)
        if not path.is_dir():
            raise RuntimeError(f"插件能力目录不存在: {path}")
        roots.append(path)
    return tuple(roots)


def _resolve_mcp_servers(
    plugin_dir: Path,
    data_dir: Path,
    declared: list[McpServerSpec],
) -> dict[str, dict[str, Any]]:
    servers: dict[str, dict[str, Any]] = {}
    for spec in declared:
        if not isinstance(spec, McpServerSpec) or not spec.name or not spec.command:
            raise RuntimeError(f"插件 MCP server 声明无效: {spec!r}")
        if spec.name in servers:
            raise RuntimeError(f"插件 MCP server 名称重复: {spec.name}")
        command = [_resolve_command_item(plugin_dir, item) for item in spec.command]
        cwd_path = Path(spec.cwd)
        cwd = (
            str(cwd_path)
            if cwd_path.is_absolute()
            else str((plugin_dir / cwd_path).resolve(strict=False))
        )
        env = {**spec.env, "AKA_PLUGIN_DATA_DIR": str(data_dir)}
        if _is_python_command(command[0]):
            runtime_root = _resolve_mcp_runtime_root(plugin_dir, cwd, command)
            if runtime_root is not None:
                venv_python = _venv_python(runtime_root / ".venv")
                if venv_python.exists():
                    command[0] = str(venv_python)
        servers[spec.name] = {"command": command, "env": env, "cwd": cwd}
    return servers


def _resolve_command_item(plugin_dir: Path, item: str) -> str:
    path = Path(item)
    if path.is_absolute() or ("/" not in item and "\\" not in item and not item.startswith(".")):
        return item
    return str((plugin_dir / path).resolve(strict=False))


def _is_python_command(value: str) -> bool:
    return Path(value).name.lower() in {"python", "python3", "python.exe"}


def _resolve_mcp_runtime_root(
    plugin_dir: Path,
    cwd: str,
    command: list[str],
) -> Path | None:
    candidates: list[Path] = []
    if len(command) >= 2:
        script_path = Path(command[1])
        if script_path.is_absolute():
            candidates.append(script_path.parent)
    candidates.extend([Path(cwd), plugin_dir])
    for candidate in candidates:
        if (candidate / "requirements.txt").exists():
            return candidate
    return None


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _collect_skill_names(skill_roots: tuple[Path, ...]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for root in skill_roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir() or not (child / "SKILL.md").exists():
                continue
            if child.name in seen:
                continue
            seen.add(child.name)
            names.append(child.name)
    return names


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
            call_id=ctx.request.call_id,
            source=ctx.request.source,
            request_text=ctx.request.request_text,
            tool_batch=ctx.request.tool_batch,
            tool_batch_index=ctx.request.tool_batch_index,
        )
        # 2. 调插件 handler，返回值决定行为
        result = self._handler(event)
        if inspect.isawaitable(result):
            result = await result
        # 3. None → 不改参；dict → 新 arguments；HookOutcome → 允许插件直接 deny
        if result is None:
            return HookOutcome()
        if isinstance(result, HookOutcome):
            return result
        if isinstance(result, dict):
            return HookOutcome(updated_input=cast("dict[str, Any]", result))
        return HookOutcome()


def _is_plugin_disabled(plugin_dir: Path) -> bool:
    return (plugin_dir / "plugin.disabled").exists()
