from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from agent.restart import RestartGate
from agent.control.frame_book import FrameBook

if TYPE_CHECKING:
    from agent.plugins.manager import PluginManager
    from infra.channels.artifacts import ChannelAttachmentArtifactStore

logger = logging.getLogger(__name__)

from agent.config_models import Config
from agent.plugins.manifest import plugins_root
from bootstrap.cleanup import run_cleanup_steps
from bootstrap.workspace_lock import PluginPublicationLock
from bus.event_bus import EventBus
from bus.queue import MessageBus
from core.net.http import SharedHttpResources
from session.artifact_store import ArtifactStore
from session.log import MessageLog
from session.admissions import SessionAdmissions
from session.identities import ChannelIdentities
from session.inbound_store import InboundHandoffStore


@dataclass
class CoreRuntime:
    """只装配消息、资源与插件；回复和来源由普通插件运行。"""

    config: Config
    workspace: Path
    http_resources: SharedHttpResources
    bus: MessageBus
    event_bus: EventBus
    message_log: MessageLog
    admissions: SessionAdmissions
    identities: ChannelIdentities
    inbound_store: InboundHandoffStore
    artifact_metadata: ArtifactStore
    channel_attachment_store: ChannelAttachmentArtifactStore
    plugin_manager: PluginManager
    plugin_publication_lock: PluginPublicationLock
    restart_gate: "RestartGate"
    control_frames: FrameBook
    _plugin_publication_locked: bool = False

    def _lock_plugin_publication(self) -> None:
        if not self._plugin_publication_locked:
            self.plugin_publication_lock.acquire()
            self._plugin_publication_locked = True

    async def start(self) -> None:
        """取得插件目录独占权，再发布插件；业务资源由插件生命周期拥有。"""
        self._lock_plugin_publication()
        await self.plugin_manager.load_all()
        self.plugin_manager.sync_manifest()

    async def inspect_modules(self) -> str:
        """展示实际发布的组合图，不再构造旧回复 Pipeline。"""
        self._lock_plugin_publication()
        await self.plugin_manager.load_all()
        snapshot = self.plugin_manager.current_snapshot
        assert snapshot is not None and snapshot.composition_topology is not None
        topology = snapshot.composition_topology
        parts = [f"identity: {topology.identity}", f"revision: {topology.composition_revision}"]
        parts.extend(f"fiber: {fiber.parent or '<root>'} -> {fiber.name}" for fiber in topology.fibers)
        parts.extend(f"listener: {listener}" for listener in topology.listeners)
        return "\n".join(parts)

    async def stop(self) -> None:
        """先排空插件资源，再关闭各自拥有的数据库连接。"""
        async def close_control_frames() -> None:
            self.control_frames.close()

        async def close_storage() -> None:
            # 每个连接都尝试关闭；前一项失败不能泄漏后续 owner。
            errors: list[Exception] = []
            for store in (self.inbound_store, self.identities, self.admissions,
                          self.artifact_metadata, self.message_log):
                try:
                    store.close()
                except Exception as error:
                    errors.append(error)
            if errors:
                raise ExceptionGroup("Core storage close 失败", errors)

        await run_cleanup_steps(
            ("plugin_manager.terminate_all", self.plugin_manager.terminate_all),
            ("control_frames.close", close_control_frames),
            ("event_bus.aclose", self.event_bus.aclose),
            ("plugin_publication_lock.release", self._release_plugin_publication),
            ("storage.close", close_storage),
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
    clear_stale_session_admissions: bool = False,
    plugin_dirs: Iterable[Path] | None = None,
) -> CoreRuntime:
    """从已迁移消息库装配窄 owner；构造失败关闭此前取得的连接。"""
    from contextlib import ExitStack
    from agent.plugins.manager import PluginManager
    from infra.channels.artifacts import ChannelAttachmentArtifactStore

    # 插件子进程只能使用宿主明确绑定的 Core；不能让普通插件从自身路径猜测。
    os.environ["AKASHIC_CORE_ROOT"] = str(Path(__file__).resolve().parents[1])

    # 1. MessageLog 先核对 schema，旧库不能借普通启动绕过 yoyo。
    bus = MessageBus()
    event_bus = EventBus()
    with ExitStack() as cleanup:
        message_log = MessageLog(workspace / "sessions.db")
        _ = cleanup.callback(message_log.close)
        artifact_metadata = ArtifactStore(workspace / "sessions.db")
        _ = cleanup.callback(artifact_metadata.close)
        admissions = SessionAdmissions(workspace / "sessions.db")
        _ = cleanup.callback(admissions.close)
        identities = ChannelIdentities(workspace / "sessions.db")
        _ = cleanup.callback(identities.close)
        inbound_store = InboundHandoffStore(workspace / "sessions.db")
        _ = cleanup.callback(inbound_store.close)
        if clear_stale_session_admissions:
            admissions.clear_stale()
        bus.bind_session_admission_owner(admissions)
        bus.bind_durable_inbound_store(inbound_store)
        attachments = ChannelAttachmentArtifactStore(
            workspace=workspace, metadata_store=artifact_metadata,
        )
        # 2. PluginManager 分配日志、归档和资源能力，不持有旧 SessionManager。
        if restart_gate is None:
            # 每次真实 Core host 启动都必须有新的 transport identity；不能用
            # 固定字符串，否则相邻 unmanaged 进程会被客户端误认为同一次启动。
            restart_gate = RestartGate(boot_id=uuid4().hex, supervised=False)
        control_frames = FrameBook()
        resolved_plugin_dirs = (
            _resolve_plugin_dirs(workspace)
            if plugin_dirs is None
            else _resolve_plugin_dirs(workspace, plugin_dirs=plugin_dirs)
        )
        manager = PluginManager(
            plugin_dirs=resolved_plugin_dirs, event_bus=event_bus,
            workspace=workspace, message_log=message_log, channel_identities=identities,
            installed_cache_root=plugins_root() / "cache",
            channel_attachment_store=attachments,
            disabled_builtin_plugins=_disabled_builtin_plugins_for_runtime(
                config, resolved_plugin_dirs
            ),
            restart_gate=restart_gate,
            control_frames=control_frames,
        )
        manager.channel_generation_host.bind_input_custody(bus)
        bus.bind_channel_outbound_dispatcher(manager.channel_generation_host.dispatch_outbound)
        runtime = CoreRuntime(
            config=config, workspace=workspace, http_resources=http_resources,
            bus=bus, event_bus=event_bus, message_log=message_log,
            admissions=admissions, identities=identities, inbound_store=inbound_store,
            artifact_metadata=artifact_metadata, channel_attachment_store=attachments,
            plugin_manager=manager, plugin_publication_lock=PluginPublicationLock(plugins_root()),
            restart_gate=restart_gate,
            control_frames=control_frames,
        )
        _ = cleanup.pop_all()
        return runtime


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
) -> frozenset[str]:
    """Apply generic disabled/Workload rules to explicit development roots."""

    disabled = set(config.disabled_builtin_plugins)
    roots = tuple(plugin_dirs)
    if not roots:
        return frozenset(disabled)

    from agent.plugins.source_resolver import resolve_plugin_sources

    known = {
        source.plugin_name or source.plugin_root.name
        for source in resolve_plugin_sources(list(roots))
    }
    unknown = sorted(disabled - known)
    if unknown:
        raise ValueError(
            "agent.plugins.disabled_builtin 包含未知内置插件: " + ", ".join(unknown)
        )
    if os.environ.get("AKASHIC_WORKLOAD_SOCKET", "").strip():
        return frozenset(disabled)

    from agent.plugins.static_manifest import load_static_plugin_manifest

    unavailable = {
        manifest.name
        for root in roots
        for path in root.glob("*/akashic.plugin.toml")
        if (manifest := load_static_plugin_manifest(path.parent)).workloads
    }
    if unavailable:
        logger.warning(
            "当前部署没有 Workload Controller，未启用内置 Workload 插件: %s",
            ", ".join(sorted(unavailable)),
        )
    disabled.update(unavailable)
    return frozenset(disabled)
