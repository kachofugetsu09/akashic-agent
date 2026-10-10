from __future__ import annotations


import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from agent.restart import RestartGate

if TYPE_CHECKING:
    from agent.plugins.manager import PluginManager

logger = logging.getLogger(__name__)

from agent.config_models import Config
from agent.plugins.manifest import plugins_root
from agent.plugins.source_resolver import PluginSourceFailure
from agent.plugins.distribution_sources import distribution_sources
from bootstrap.cleanup import run_cleanup_steps
from bootstrap.workspace_lock import PluginPublicationLock
from core.net.http import SharedHttpResources


@dataclass
class CoreRuntime:
    """装配宿主资源与插件；业务资源由各 provider 拥有。"""

    config: Config
    workspace: Path
    http_resources: SharedHttpResources
    plugin_manager: PluginManager
    plugin_publication_lock: PluginPublicationLock
    restart_gate: "RestartGate"
    _plugin_publication_locked: bool = False

    def _lock_plugin_publication(self) -> None:
        if not self._plugin_publication_locked:
            self.plugin_publication_lock.acquire()
            self._plugin_publication_locked = True

    async def start(self) -> None:
        """取得插件目录独占权，再发布插件；业务资源由插件生命周期拥有。"""
        self._lock_plugin_publication()
        await self.plugin_manager.load_all()

    async def inspect_modules(self) -> str:
        """展示实际发布的组合图，不再构造旧回复 Pipeline。"""
        self._lock_plugin_publication()
        root = self.plugin_manager.live_root
        if root is None:
            await self.plugin_manager.load_all()
            root = self.plugin_manager.live_root
            if root is None:
                raise RuntimeError("插件初始化成功但没有发布 live Root")
        topology = root.topology_view()
        parts = [f"identity: {topology.identity}", f"revision: {topology.composition_revision}"]
        parts.extend(f"fiber: {fiber.parent or '<root>'} -> {fiber.name}" for fiber in topology.fibers)
        parts.extend(f"listener: {listener}" for listener in topology.listeners)
        return "\n".join(parts)

    async def stop(self) -> None:
        """排空插件资源后释放发布锁。"""
        # 插件仍持有资源时不能释放 provider、发布锁或数据库。
        await self.plugin_manager.terminate_all()
        await run_cleanup_steps(
            ("plugin_publication_lock.release", self._release_plugin_publication),
        )

    async def _release_plugin_publication(self) -> None:
        if self._plugin_publication_locked:
            self.plugin_publication_lock.release()
            self._plugin_publication_locked = False


def build_core_runtime(
    config: Config,
    workspace: Path,
    http_resources: SharedHttpResources,
    restart_gate: RestartGate | None = None,
    *,
    plugin_dirs: Iterable[Path] | None = None,
    host_ready: Callable[[], bool] | None = None,
) -> CoreRuntime:
    """装配插件宿主，不创建业务数据库。"""
    from agent.plugins.manager import PluginManager

    # 插件子进程只能使用宿主明确绑定的 Core；不能让普通插件从自身路径猜测。
    os.environ["AKASHIC_CORE_ROOT"] = str(Path(__file__).resolve().parents[1])

    if restart_gate is None:
        restart_gate = RestartGate(boot_id=uuid4().hex, supervised=False)
    resolved_plugin_dirs = _resolve_plugin_dirs(workspace, plugin_dirs=plugin_dirs or ())
    disabled, source_failures = _disabled_builtin_plugins_for_runtime(config, resolved_plugin_dirs)
    distribution = distribution_sources(workspace, plugins_root())
    manager = PluginManager(
        plugin_dirs=resolved_plugin_dirs, workspace=workspace,
        installed_cache_root=plugins_root() / "cache",
        disabled_builtin_plugins=disabled, source_failures=source_failures,
        distribution_sources=distribution.sources,
        ignored_installed_roots=distribution.ignored_installed_roots,
        restart_gate=restart_gate, host_ready=host_ready,
    )
    return CoreRuntime(
        config=config, workspace=workspace, http_resources=http_resources,
        plugin_manager=manager, plugin_publication_lock=PluginPublicationLock(plugins_root()),
        restart_gate=restart_gate,
    )


def _resolve_plugin_dirs(
    workspace: Path,
    *,
    plugin_dirs: Iterable[Path] = (),
) -> list[Path]:
    """Return only explicitly requested development plugin roots."""

    _ = workspace
    roots = [Path(item).expanduser() for item in plugin_dirs]
    extra = os.environ.get("AKASHIC_EXTRA_PLUGIN_DIRS", "")
    roots.extend(
        Path(item).expanduser() for item in extra.split(os.pathsep) if item.strip()
    )
    result: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        normalized = root.resolve(strict=False)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(root)
    return result


def _disabled_builtin_plugins_for_runtime(
    config: Config,
    plugin_dirs: Iterable[Path] = (),
) -> tuple[frozenset[str], tuple[PluginSourceFailure, ...]]:
    """校验显式禁用的插件，不根据运行能力改写用户选择。"""

    disabled = set(config.disabled_builtin_plugins)
    roots = tuple(plugin_dirs)
    if not roots:
        return frozenset(disabled), ()

    from agent.plugins.source_resolver import scan_plugin_sources

    scan = scan_plugin_sources(list(roots))
    known = {
        source.plugin_name
        for source in scan.sources
    }
    unknown = sorted(disabled - known)
    if unknown:
        raise ValueError(
            "agent.plugins.disabled_builtin 包含未知内置插件: " + ", ".join(unknown)
        )
    return frozenset(disabled), scan.failures
