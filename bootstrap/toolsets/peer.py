from __future__ import annotations

from agent.config_models import Config
from agent.peer_agent.process_manager import PeerProcessConfig, PeerProcessManager
from agent.peer_agent.poller import PeerAgentPoller
from bus.queue import MessageBus
from core.net.http import SharedHttpResources


def build_peer_agent_resources(
    config: Config,
    bus: MessageBus,
    http_resources: SharedHttpResources,
) -> tuple[PeerProcessManager | None, PeerAgentPoller | None]:
    """构建 PeerProcessManager 和 PeerAgentPoller（同步部分），工具发现在异步启动时完成。"""
    if not config.peer_agents:
        return None, None

    proc_configs = [
        PeerProcessConfig(
            name=pa.name,
            base_url=pa.base_url,
            launcher=pa.launcher,
            cwd=pa.cwd,
            health_path=pa.health_path,
            startup_timeout_s=pa.startup_timeout_s,
            shutdown_timeout_s=pa.shutdown_timeout_s,
        )
        for pa in config.peer_agents
    ]
    pm = PeerProcessManager(configs=proc_configs, requester=http_resources.local_service)
    poller = PeerAgentPoller(bus=bus, process_manager=pm, requester=http_resources.local_service)
    return pm, poller
