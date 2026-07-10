from __future__ import annotations

import asyncio
import functools
import hashlib
import importlib.util
import inspect
import json
import logging
import os
import secrets
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Awaitable, Callable
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
from agent.plugins.scope import CleanupFailure, PluginScope, ScopedEventBus
from agent.plugins.generation import (
    GateCheckResult,
    GateResult,
    PluginContributions,
    PluginGeneration,
    PluginSemanticCheck,
)
from agent.plugins.importer import FreshPluginImporter
from agent.lifecycle.phase import topo_sort_modules
from proactive_v2.lifecycle import ProactiveLifecycleSpec
from proactive_v2.lifecycle import ProactiveLifecycleBuilder
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
        self._scopes: dict[str, PluginScope] = {}
        self._cleanup_failures: list[CleanupFailure] = []
        self._active_generations: dict[str, PluginGeneration] = {}
        self._prepared_generations: dict[str, PluginGeneration] = {}
        self._gate_results: dict[str, GateResult] = {}
        self._stable_aliases: dict[str, str] = {}
        self._generation_sequence = 0
        self._candidate_prepare_lock = asyncio.Lock()
        self._fresh_importer = FreshPluginImporter()
        self._manager_namespace = secrets.token_hex(4)

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

    @property
    def cleanup_failures(self) -> list[CleanupFailure]:
        return list(self._cleanup_failures)

    def generation(self, plugin_id: str) -> PluginGeneration | None:
        return self._active_generations.get(plugin_id)

    def latest_gate(self, plugin_id: str) -> GateResult | None:
        return self._gate_results.get(plugin_id)

    def prepared_generation(self, plugin_id: str) -> PluginGeneration | None:
        return self._prepared_generations.get(plugin_id)

    def sync_manifest(self, *, plugins_home: Path | None = None) -> Path:
        entries = load_plugin_manifest(plugins_home)
        for mod in self.discover():
            _ = entries.setdefault(_resolve_plugin_id(mod), True)
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
            _ = await self._load_one(mod)

    async def prepare_candidate(self, plugin_id: str) -> PluginGeneration | None:
        await self.discard_prepared(plugin_id)
        for mod in self.discover():
            if _resolve_plugin_id(mod) == plugin_id:
                return await self._load_one(mod, activate=False)
        raise KeyError(f"插件不存在: {plugin_id}")

    async def discard_prepared(self, plugin_id: str) -> None:
        generation = self._prepared_generations.pop(plugin_id, None)
        if generation is None:
            return
        self._cleanup_failures.extend(await generation.scope.aclose())
        self._remove_module_tree(generation.module_path)
        generation.state = "discarded"

    async def prepare_changed(self) -> list[dict[str, object]]:
        async with self._candidate_prepare_lock:
            return await self._prepare_changed()

    async def _prepare_changed(self) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        discovered = {
            _resolve_plugin_id(mod): mod
            for mod in self.discover()
        }
        for plugin_id, active in tuple(self._active_generations.items()):
            mod = discovered.get(plugin_id)
            if mod is None:
                continue
            plugin_dir = Path(mod["plugin_root"])
            try:
                source_revision = _source_revision(plugin_dir)
                config_revision = _file_revision(
                    _resolve_plugin_data_dir(
                        mod["name"],
                        mod,
                        self._installed_cache_root,
                    )
                    / "config.local.toml"
                )
            except Exception:
                source_revision = ""
                config_revision = ""
            current_prepared = self._prepared_generations.get(plugin_id)
            matches_active = (
                source_revision == active.source_revision
                and config_revision == active.config_revision
            )
            if matches_active:
                if current_prepared is None:
                    continue
                await self.discard_prepared(plugin_id)
                result = {
                    "plugin_id": plugin_id,
                    "active_generation": active.generation_id,
                    "prepared_generation": None,
                    "gate_status": "active",
                    "candidate_revision": source_revision,
                }
                results.append(result)
                logger.info(
                    "plugin_candidate_status %s",
                    json.dumps(result, ensure_ascii=False, sort_keys=True),
                )
                continue
            if (
                current_prepared is not None
                and source_revision == current_prepared.source_revision
                and config_revision == current_prepared.config_revision
            ):
                continue
            prepared = await self.prepare_candidate(plugin_id)
            gate = self.latest_gate(plugin_id)
            result: dict[str, object] = {
                "plugin_id": plugin_id,
                "active_generation": active.generation_id,
                "prepared_generation": (
                    prepared.generation_id if prepared is not None else None
                ),
                "gate_status": gate.status if gate is not None else "failed",
                "candidate_revision": (
                    gate.candidate_revision if gate is not None else ""
                ),
            }
            results.append(result)
            logger.info(
                "plugin_candidate_status %s",
                json.dumps(result, ensure_ascii=False, sort_keys=True),
            )
        return results

    async def _load_one(
        self,
        mod: dict[str, str],
        *,
        activate: bool = True,
    ) -> PluginGeneration | None:
        stable_module_path = mod["import_path"]
        plugin_dir = Path(mod["plugin_root"])
        initial_plugin_id = _resolve_plugin_id(mod)
        if activate and initial_plugin_id in self._active_generations:
            return self._active_generations[initial_plugin_id]
        plugin_manifest = load_plugin_manifest(_plugins_home(self._installed_cache_root))
        if plugin_manifest.get(initial_plugin_id, True) is False:
            logger.info("插件已禁用（manifest.toml）: %s", initial_plugin_id)
            return None
        if _is_plugin_disabled(plugin_dir):
            logger.info("插件已禁用（plugin.disabled）: %s", mod["name"])
            return None
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
        channel_count_before = len(self._channels)
        module_path = mod["module_path"].strip()
        if not module_path:
            raise RuntimeError(f"插件缺少 plugin.py: {plugin_dir}")
        try:
            source_revision = _source_revision(plugin_dir)
        except Exception as error:
            revision = hashlib.sha256(
                f"{plugin_dir}:{error}".encode()
            ).hexdigest()
            self._record_failed_gate(
                plugin_id=initial_plugin_id,
                revision=revision,
                check_id="source_boundary",
                reason=str(error),
            )
            return None
        data_dir = _resolve_plugin_data_dir(
            mod["name"],
            mod,
            self._installed_cache_root,
        )
        config_revision = _file_revision(data_dir / "config.local.toml")
        self._generation_sequence += 1
        generation_id = (
            f"{initial_plugin_id}:{source_revision[:12]}:{self._generation_sequence}"
        )
        mp = (
            f"{stable_module_path}__g{self._generation_sequence}_"
            f"{source_revision[:8]}_{self._manager_namespace}"
        )
        try:
            self._import_plugin(mp, Path(module_path))
        except Exception as error:
            logger.warning("插件 %s 导入失败: %s", mod["name"], error)
            self._record_failed_gate(
                plugin_id=initial_plugin_id,
                revision=source_revision,
                check_id="import",
                reason=str(error),
            )
            return None
        cls = plugin_registry.get_class(mp)
        if cls is None:
            logger.warning("插件 %s 未注册类", mod["name"])
            self._remove_module_tree(mp)
            self._record_failed_gate(
                plugin_id=initial_plugin_id,
                revision=source_revision,
                check_id="plugin_class",
                reason="plugin.py 未注册 Plugin 子类",
            )
            return None
        try:
            instance = cls()
            name = str(instance.name or mod["name"]).strip()
            if not name:
                raise RuntimeError("插件缺少 name")
            plugin_id = f"{name}@{mod['marketplace']}" if mod["marketplace"] else name
            if plugin_id != initial_plugin_id:
                raise RuntimeError(
                    f"插件目录身份与声明不一致: directory={initial_plugin_id} declared={plugin_id}"
                )
            plugin_config = _load_plugin_config(
                data_dir,
                getattr(cls, "ConfigModel", None),
            )
        except Exception as error:
            self._remove_module_tree(mp)
            self._record_failed_gate(
                plugin_id=initial_plugin_id,
                revision=source_revision,
                check_id=("config" if isinstance(error, _PluginConfigError) else "identity"),
                reason=str(error),
            )
            return None
        from agent.plugins.context import PluginContext, PluginKVStore
        scope = PluginScope(plugin_id)
        instance.context = PluginContext(  # type: ignore[attr-defined]
            event_bus=None,  # type: ignore[arg-type]
            tool_registry=None,
            plugin_id=plugin_id,
            plugin_dir=plugin_dir,
            data_dir=data_dir,
            kv_store=PluginKVStore(data_dir / ".kv.json", writable=False),
            config=plugin_config,
            workspace=self._workspace,
            session_manager=None,
            memory_engine=None,
            llm=None,
            scope=None,
            generation_id=generation_id,
        )
        plugin_registry.register_instance(mp, instance)
        initialization_started = False

        async def rollback_load() -> None:
            terminator = getattr(instance, "terminate", None)
            if initialization_started and callable(terminator):
                try:
                    typed_terminator = cast(
                        Callable[[], Awaitable[None]],
                        terminator,
                    )
                    await typed_terminator()
                except (asyncio.CancelledError, Exception) as terminate_error:
                    self._cleanup_failures.append(
                        CleanupFailure(
                            resource=f"plugin:{plugin_id}:terminate",
                            error=str(terminate_error) or type(terminate_error).__name__,
                        )
                    )
            self._cleanup_failures.extend(await scope.aclose())
            self._remove_module_tree(mp)
            for tool_name in tool_names:
                if self._tool_registry is not None:
                    self._tool_registry.unregister(tool_name)
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
            del self._channels[channel_count_before:]

        try:
            load_phase = "declarations"
            contributions = self._collect_candidate_contributions(
                instance=instance,
                plugin_id=plugin_id,
                plugin_dir=plugin_dir,
                data_dir=data_dir,
                module_path=mp,
            )
            gate_result = self._validate_candidate(
                instance=instance,
                plugin_id=plugin_id,
                revision=source_revision,
                contributions=contributions,
            )
            self._gate_results[plugin_id] = gate_result
            if gate_result.status == "failed":
                raise _CandidateRejected(gate_result)
            generation = PluginGeneration(
                plugin_id=plugin_id,
                generation_id=generation_id,
                module_path=mp,
                source_revision=source_revision,
                config_revision=config_revision,
                instance=instance,
                scope=scope,
                contributions=contributions,
                gate_result=gate_result,
                state="prepared" if not activate else "activating",
            )
            if not activate:
                self._prepared_generations[plugin_id] = generation
                return generation
            staged_event_bus = ScopedEventBus(self._event_bus, scope, staged=True)
            instance.context.event_bus = staged_event_bus
            instance.context.kv_store = PluginKVStore(data_dir / ".kv.json")
            instance.context.session_manager = self._session_manager
            instance.context.memory_engine = self._memory_engine
            instance.context.llm = self._llm
            instance.context.scope = scope
            load_phase = "initialize"
            initialization_started = True
            await instance.initialize()
            load_phase = "publish"
            self._register_tools(instance, mp, tool_names)
            self._bind_tool_hooks(instance, mp)
            self._publish_contributions(contributions)
            self._channels.extend(contributions.channels)
            self._bind_handlers(instance, mp, scope)
            staged_event_bus.activate()
            instance.context.tool_registry = self._tool_registry
        except asyncio.CancelledError:
            rollback_task = asyncio.create_task(
                rollback_load(),
                name=f"plugin_rollback:{plugin_id}",
            )
            while not rollback_task.done():
                try:
                    await asyncio.shield(rollback_task)
                except asyncio.CancelledError:
                    continue
            await rollback_task
            raise
        except _CandidateRejected as error:
            logger.warning(
                "插件 %s 候选验证失败: %s",
                mod["name"],
                error.gate.failure_reason,
            )
            await rollback_load()
            return None
        except Exception as error:
            logger.warning("插件 %s 加载失败，回滚: %s", mod["name"], error)
            self._record_failed_gate(
                plugin_id=plugin_id,
                revision=source_revision,
                check_id=load_phase,
                reason=str(error),
            )
            await rollback_load()
            return None
        self._scopes[mp] = scope
        self._loaded.add(mp)
        self._active_plugins[mp] = ActivePluginInfo(
            plugin_id=plugin_id,
            plugin_dir=plugin_dir,
            manifest=contributions.manifest,
            module_path=mp,
            skill_roots=contributions.skill_roots,
            drift_skill_roots=contributions.drift_skill_roots,
            mcp_servers=contributions.mcp_servers,
        )
        generation.state = "active"
        self._active_generations[plugin_id] = generation
        self._stable_aliases[mp] = stable_module_path
        self._remove_module_tree(stable_module_path)
        self._fresh_importer.register(stable_module_path, plugin_dir)
        plugin_registry.register_instance(stable_module_path, instance)
        sys.modules[stable_module_path] = sys.modules[mp]
        logger.info("插件已加载: %s", mod["name"])
        return generation

    def _collect_candidate_contributions(
        self,
        *,
        instance: Any,
        plugin_id: str,
        plugin_dir: Path,
        data_dir: Path,
        module_path: str,
    ) -> PluginContributions:
        cls = type(instance)
        sources: list[RegisteredProactiveSource] = []
        for source in _load_module_list(instance, "proactive_sources"):
            if not isinstance(source, ProactiveSourceSpec):
                raise RuntimeError(
                    f"插件 {plugin_id}.proactive_sources 返回值不是 ProactiveSourceSpec"
                )
            sources.append(RegisteredProactiveSource(plugin_id=plugin_id, spec=source))
        jobs: list[RegisteredPluginJob] = []
        for spec in _load_module_list(instance, "jobs"):
            if not isinstance(spec, PluginJobSpec):
                raise RuntimeError(
                    f"插件 {plugin_id}.jobs 返回值不是 PluginJobSpec"
                )
            jobs.append(
                RegisteredPluginJob(
                    plugin_id=plugin_id,
                    plugin_context=instance.context,
                    spec=spec,
                )
            )
        return PluginContributions(
            manifest={
                "name": str(instance.name or ""),
                "version": str(instance.version or ""),
                "desc": str(instance.desc or ""),
                "author": str(instance.author or ""),
            },
            skill_roots=_resolve_declared_roots(plugin_dir, cls.skill_roots()),
            drift_skill_roots=_resolve_declared_roots(
                plugin_dir,
                cls.drift_skill_roots(),
            ),
            mcp_servers=_resolve_mcp_servers(
                plugin_dir,
                data_dir,
                cls.mcp_servers(),
            ),
            before_turn_modules=tuple(
                _load_module_list(instance, "before_turn_modules")
            ),
            before_reasoning_modules=tuple(
                _load_module_list(instance, "before_reasoning_modules")
            ),
            prompt_render_modules=tuple(
                _load_module_list(instance, "prompt_render_modules")
            ),
            before_step_modules=tuple(
                _load_module_list(instance, "before_step_modules")
            ),
            after_step_modules=tuple(
                _load_module_list(instance, "after_step_modules")
            ),
            after_reasoning_modules=tuple(
                _load_module_list(instance, "after_reasoning_modules")
            ),
            after_turn_modules=tuple(
                _load_module_list(instance, "after_turn_modules")
            ),
            proactive_modules=tuple(
                _load_module_list(instance, "proactive_modules")
            ),
            proactive_lifecycles=tuple(
                _load_module_list(instance, "proactive_lifecycles")
            ),
            proactive_module_factories=tuple(
                _load_module_list(instance, "proactive_module_factories")
            ),
            proactive_runtime_factories=tuple(
                _load_module_list(instance, "proactive_runtime_factories")
            ),
            proactive_sources=tuple(sources),
            jobs=tuple(jobs),
            channels=cast(
                tuple[Channel, ...],
                tuple(_load_module_list(instance, "channels")),
            ),
        )

    def _validate_candidate(
        self,
        *,
        instance: Any,
        plugin_id: str,
        revision: str,
        contributions: PluginContributions,
    ) -> GateResult:
        checks: list[GateCheckResult] = []
        current = self._active_generations.get(plugin_id)
        other_generations = [
            generation
            for generation in self._active_generations.values()
            if generation.plugin_id != plugin_id
        ]

        def check(check_id: str, passed: bool, evidence: object = "") -> None:
            checks.append(
                GateCheckResult(
                    check_id=check_id,
                    status="passed" if passed else "failed",
                    evidence=evidence,
                )
            )

        check(
            "api_version",
            getattr(instance, "api_version", None) == 1,
            getattr(instance, "api_version", None),
        )
        metadata = plugin_registry.get_handlers_by_module_path(type(instance).__module__)
        tool_names = [
            md.tool_name or md.handler_name
            for md in metadata
            if md.kind == MetadataKind.TOOL
        ]
        duplicate_tools = _duplicates(tool_names)
        current_tool_names = (
            {
                metadata.tool_name or metadata.handler_name
                for metadata in plugin_registry.get_handlers_by_module_path(
                    current.module_path
                )
                if metadata.kind == MetadataKind.TOOL
            }
            if current is not None
            else set()
        )
        occupied_tools = (
            sorted(
                name
                for name in tool_names
                if self._tool_registry.has_tool(name) and name not in current_tool_names
            )
            if self._tool_registry is not None
            else []
        )
        check(
            "tool_names",
            not duplicate_tools and not occupied_tools,
            {"duplicates": duplicate_tools, "occupied": occupied_tools},
        )
        source_ids = [source.spec.id for source in contributions.proactive_sources]
        source_errors = [
            source.spec.id
            for source in contributions.proactive_sources
            if not source.spec.id
            or not source.spec.channels
            or not set(source.spec.channels).issubset({"alert", "content", "context"})
            or not source.spec.server
            or not source.spec.fetch_tool
            or source.spec.poll_interval_seconds < 0
            or source.spec.server not in contributions.mcp_servers
        ]
        check(
            "proactive_sources",
            not _duplicates(source_ids) and not source_errors,
            {"duplicates": _duplicates(source_ids), "invalid": source_errors},
        )
        occupied_servers = {
            server_name
            for generation in other_generations
            for server_name in generation.contributions.mcp_servers
        }
        check(
            "mcp_servers",
            not occupied_servers.intersection(contributions.mcp_servers),
            sorted(occupied_servers.intersection(contributions.mcp_servers)),
        )
        job_ids = [job.spec.id for job in contributions.jobs]
        check(
            "job_ids",
            all(job_ids) and not _duplicates(job_ids) if job_ids else True,
            _duplicates(job_ids),
        )
        channel_names = [
            str(getattr(channel, "name", "")).strip()
            for channel in contributions.channels
        ]
        occupied_channels = {
            str(getattr(channel, "name", "")).strip()
            for generation in other_generations
            for channel in generation.contributions.channels
        }
        check(
            "channel_names",
            (
                all(channel_names)
                and not _duplicates(channel_names)
                and not occupied_channels.intersection(channel_names)
            )
            if channel_names
            else True,
            {
                "duplicates": _duplicates(channel_names),
                "occupied": sorted(occupied_channels.intersection(channel_names)),
            },
        )
        phase_groups = (
            ("before_turn_modules", contributions.before_turn_modules),
            ("before_reasoning_modules", contributions.before_reasoning_modules),
            ("prompt_render_modules", contributions.prompt_render_modules),
            ("before_step_modules", contributions.before_step_modules),
            ("after_step_modules", contributions.after_step_modules),
            ("after_reasoning_modules", contributions.after_reasoning_modules),
            ("after_turn_modules", contributions.after_turn_modules),
        )
        try:
            for field_name, candidate_modules in phase_groups:
                active_modules = [
                    module
                    for generation in other_generations
                    for module in getattr(generation.contributions, field_name)
                ]
                _ = topo_sort_modules([*active_modules, *candidate_modules])
        except RuntimeError as error:
            check("phase_graph", False, str(error))
        else:
            check("phase_graph", True)
        lifecycle_ids = [
            lifecycle.id
            for lifecycle in contributions.proactive_lifecycles
            if isinstance(lifecycle, ProactiveLifecycleSpec)
        ]
        check(
            "proactive_lifecycles",
            len(lifecycle_ids) == len(contributions.proactive_lifecycles)
            and not _duplicates(lifecycle_ids)
            and not {
                lifecycle.id
                for generation in other_generations
                for lifecycle in generation.contributions.proactive_lifecycles
                if isinstance(lifecycle, ProactiveLifecycleSpec)
            }.intersection(lifecycle_ids),
            {
                "duplicates": _duplicates(lifecycle_ids),
                "occupied": sorted(
                    {
                        lifecycle.id
                        for generation in other_generations
                        for lifecycle in generation.contributions.proactive_lifecycles
                        if isinstance(lifecycle, ProactiveLifecycleSpec)
                    }.intersection(lifecycle_ids)
                ),
            },
        )
        lifecycle_structure_errors: list[str] = []
        for lifecycle in contributions.proactive_lifecycles:
            if not isinstance(lifecycle, ProactiveLifecycleSpec):
                continue
            if (
                not lifecycle.id
                or any(not value for value in lifecycle.initial_slots)
                or any(not value for value in lifecycle.terminal_slots)
                or len(set(lifecycle.initial_slots)) != len(lifecycle.initial_slots)
                or len(set(lifecycle.terminal_slots)) != len(lifecycle.terminal_slots)
            ):
                lifecycle_structure_errors.append(f"{lifecycle.id}: slots")
                continue
            try:
                _ = ProactiveLifecycleBuilder().build(
                    ProactiveLifecycleSpec(
                        id=lifecycle.id,
                        modules=lifecycle.modules,
                        initial_slots=lifecycle.initial_slots,
                    )
                )
            except RuntimeError as error:
                lifecycle_structure_errors.append(f"{lifecycle.id}: {error}")
        check(
            "proactive_lifecycle_structure",
            not lifecycle_structure_errors,
            lifecycle_structure_errors,
        )
        try:
            semantic_checks = instance.static_semantic_checks()
        except Exception as error:
            check("semantic_checks", False, str(error))
        else:
            invalid_semantic = [
                semantic
                for semantic in semantic_checks
                if not isinstance(semantic, PluginSemanticCheck) or not semantic.passed
            ]
            check(
                "semantic_checks",
                not invalid_semantic,
                [
                    getattr(semantic, "evidence", repr(semantic))
                    for semantic in invalid_semantic
                ],
            )
        failed = [item for item in checks if item.status == "failed"]
        return GateResult(
            gate_id="G1/G3-static",
            plugin_id=plugin_id,
            candidate_revision=revision,
            status="failed" if failed else "passed",
            checks=tuple(checks),
            failure_reason="; ".join(item.check_id for item in failed),
        )

    def _publish_contributions(self, contributions: PluginContributions) -> None:
        self._before_turn_modules.extend(contributions.before_turn_modules)
        self._before_reasoning_modules.extend(contributions.before_reasoning_modules)
        self._prompt_render_modules.extend(contributions.prompt_render_modules)
        self._before_step_modules.extend(contributions.before_step_modules)
        self._after_step_modules.extend(contributions.after_step_modules)
        self._after_reasoning_modules.extend(contributions.after_reasoning_modules)
        self._after_turn_modules.extend(contributions.after_turn_modules)
        self._proactive_modules.extend(contributions.proactive_modules)
        self._proactive_lifecycles.extend(contributions.proactive_lifecycles)
        self._proactive_module_factories.extend(
            contributions.proactive_module_factories
        )
        self._proactive_runtime_factories.extend(
            contributions.proactive_runtime_factories
        )
        self._proactive_sources.extend(contributions.proactive_sources)
        self._jobs.extend(contributions.jobs)

    def _record_failed_gate(
        self,
        *,
        plugin_id: str,
        revision: str,
        check_id: str,
        reason: str,
    ) -> None:
        self._gate_results[plugin_id] = GateResult(
            gate_id="G1/G3-static",
            plugin_id=plugin_id,
            candidate_revision=revision,
            status="failed",
            checks=(
                GateCheckResult(
                    check_id=check_id,
                    status="failed",
                    evidence=reason,
                ),
            ),
            failure_reason=reason,
        )

    def _import_plugin(self, module_name: str, path: Path) -> None:
        self._fresh_importer.register(module_name, path.parent)
        spec = self._fresh_importer.root_spec(module_name, path)
        if spec is None or spec.loader is None:
            self._fresh_importer.unregister(module_name)
            raise ImportError(f"无法加载插件文件: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except BaseException:
            self._remove_module_tree(module_name)
            raise

    def _remove_module_tree(self, module_name: str) -> None:
        self._fresh_importer.unregister(module_name)
        plugin_registry.remove_module_tree(module_name)
        for imported_name in tuple(sys.modules):
            if imported_name == module_name or imported_name.startswith(f"{module_name}."):
                _ = sys.modules.pop(imported_name, None)

    def _register_tools(
        self,
        instance: Any,
        module_path: str,
        tool_names: list[str],
    ) -> None:
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
            if self._tool_registry.has_tool(tool_name):
                raise RuntimeError(f"插件工具名称重复: {tool_name}")
            tool_names.append(tool_name)
            self._tool_registry.register(
                ToolCls(),
                risk=md.tool_risk or "read-write",
                always_on=bool(md.tool_always_on),
                search_hint=md.tool_search_hint,
                source_type="plugin",
                source_name=plugin_name,
            )
            logger.info("插件工具已注册: %s (来自 %s)", tool_name, plugin_name)

    def _bind_handlers(
        self,
        instance: Any,
        module_path: str,
        scope: PluginScope,
    ) -> None:
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
            _ = scope.subscribe(self._event_bus, ctx_type, bound)

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

    async def terminate_all(self) -> None:
        for plugin_id in tuple(self._prepared_generations):
            await self.discard_prepared(plugin_id)
        for mp in list(self._loaded):
            active_info = self._active_plugins.get(mp)
            instance = plugin_registry.get_instance(mp)
            terminator = getattr(instance, "terminate", None)
            if callable(terminator):
                try:
                    typed_terminator = cast(
                        Callable[[], Awaitable[None]],
                        terminator,
                    )
                    await typed_terminator()
                except Exception as e:
                    logger.warning("插件 terminate 失败 (%s): %s", mp, e)
                    self._cleanup_failures.append(
                        CleanupFailure(
                            resource=f"plugin:{mp}:terminate",
                            error=str(e),
                        )
                    )
            scope = self._scopes.pop(mp, None)
            if scope is not None:
                self._cleanup_failures.extend(await scope.aclose())
            # 注销工具
            for md in plugin_registry.get_handlers_by_module_path(mp):
                if md.kind == MetadataKind.TOOL and self._tool_registry is not None:
                    self._tool_registry.unregister(md.tool_name or md.handler_name)
            self._remove_module_tree(mp)
            stable_alias = self._stable_aliases.pop(mp, None)
            if stable_alias is not None:
                active_alias = plugin_registry.get_instance(stable_alias)
                if active_alias is instance:
                    self._remove_module_tree(stable_alias)
                else:
                    self._fresh_importer.unregister(stable_alias)
            if active_info is not None:
                generation = self._active_generations.pop(active_info.plugin_id, None)
                if generation is not None:
                    generation.state = "retired"
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
        self._scopes.clear()
        self._active_generations.clear()
        self._prepared_generations.clear()
        self._stable_aliases.clear()


class _PluginConfigError(Exception):
    pass


class _CandidateRejected(Exception):
    def __init__(self, gate: GateResult) -> None:
        super().__init__(gate.failure_reason)
        self.gate = gate


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
        if not isinstance(config_model, type) or not issubclass(config_model, BaseModel):
            raise _PluginConfigError("ConfigModel 必须继承 pydantic.BaseModel")
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
        raise RuntimeError(
            f"插件 {type(instance).__name__}.{method_name} 不是可调用对象"
        )
    try:
        loaded = provider()
    except Exception as e:
        raise RuntimeError(
            f"插件 {type(instance).__name__}.{method_name} 声明失败: {e}"
        ) from e
    if loaded is None:
        raise RuntimeError(
            f"插件 {type(instance).__name__}.{method_name} 返回值不能为 None"
        )
    if not isinstance(loaded, list):
        raise RuntimeError(
            f"插件 {type(instance).__name__}.{method_name} 返回值不是 list"
        )
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
    return _plugins_home(installed_cache_root) / "data" / f"{name}-{suffix}"


def _plugins_home(installed_cache_root: Path | None) -> Path:
    if installed_cache_root is not None:
        return installed_cache_root.parent
    return Path.home() / ".akashic-plugin"


def _resolve_declared_roots(
    plugin_dir: Path,
    declared: tuple[str, ...],
) -> tuple[Path, ...]:
    plugin_root = plugin_dir.resolve(strict=False)
    roots: list[Path] = []
    for raw_path in declared:
        path = (plugin_dir / raw_path).resolve(strict=False)
        _require_plugin_path(plugin_root, path, "能力目录")
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
    plugin_root = plugin_dir.resolve(strict=False)
    for spec in declared:
        if not isinstance(spec, McpServerSpec) or not spec.name or not spec.command:
            raise RuntimeError(f"插件 MCP server 声明无效: {spec!r}")
        if not all(isinstance(item, str) and item for item in spec.command):
            raise RuntimeError(f"插件 MCP command 声明无效: {spec.name}")
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in spec.env.items()
        ):
            raise RuntimeError(f"插件 MCP env 声明无效: {spec.name}")
        if spec.name in servers:
            raise RuntimeError(f"插件 MCP server 名称重复: {spec.name}")
        command = [
            _resolve_command_item(plugin_root, item, executable=index == 0)
            for index, item in enumerate(spec.command)
        ]
        cwd_path = Path(spec.cwd)
        resolved_cwd = (
            cwd_path.resolve(strict=False)
            if cwd_path.is_absolute()
            else (plugin_root / cwd_path).resolve(strict=False)
        )
        _require_plugin_path(plugin_root, resolved_cwd, "MCP cwd")
        cwd = str(resolved_cwd)
        env = {**spec.env, "AKA_PLUGIN_DATA_DIR": str(data_dir)}
        if _is_python_command(command[0]):
            runtime_root = _resolve_mcp_runtime_root(plugin_dir, cwd, command)
            if runtime_root is not None:
                venv_python = _venv_python(runtime_root / ".venv")
                if venv_python.exists():
                    command[0] = str(venv_python)
        servers[spec.name] = {"command": command, "env": env, "cwd": cwd}
    return servers


def _resolve_command_item(
    plugin_dir: Path,
    item: str,
    *,
    executable: bool,
) -> str:
    path = Path(item)
    if executable and path.is_absolute():
        return item
    if "/" not in item and "\\" not in item and not item.startswith("."):
        return item
    resolved = (
        path.resolve(strict=False)
        if path.is_absolute()
        else (plugin_dir / path).resolve(strict=False)
    )
    _require_plugin_path(plugin_dir, resolved, "MCP command")
    return str(resolved)


def _require_plugin_path(plugin_dir: Path, path: Path, label: str) -> None:
    try:
        _ = path.relative_to(plugin_dir)
    except ValueError as error:
        raise RuntimeError(f"插件 {label} 越界: {path}") from error


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


def _file_revision(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(str(path.resolve(strict=False)).encode())
    if path.is_file():
        digest.update(path.read_bytes())
    else:
        digest.update(b"<missing>")
    return digest.hexdigest()


def _source_revision(plugin_dir: Path) -> str:
    digest = hashlib.sha256()
    root = plugin_dir.resolve(strict=False)
    excluded = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
    }
    for path in sorted(plugin_dir.rglob("*")):
        relative = path.relative_to(plugin_dir)
        if any(part in excluded for part in relative.parts):
            continue
        if path.is_symlink():
            resolved = path.resolve(strict=False)
            _require_plugin_path(root, resolved, "源码符号链接")
            digest.update(str(relative).encode())
            digest.update(os.readlink(path).encode())
            if resolved.is_file():
                digest.update(resolved.read_bytes())
            continue
        if not path.is_file():
            continue
        resolved = path.resolve(strict=False)
        _require_plugin_path(root, resolved, "源码文件")
        digest.update(str(relative).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _is_plugin_disabled(plugin_dir: Path) -> bool:
    return (plugin_dir / "plugin.disabled").exists()
