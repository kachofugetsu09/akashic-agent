"""
PeerAgentRegistry：启动时自发现所有配置的 peer agents，生成工具并注册。
"""
from __future__ import annotations

import logging

from agent.config_models import PeerAgentConfig
from agent.peer_agent.card_resolver import (
    AgentCard,
    AgentCardUnavailableError,
    fetch_agent_card,
)
from agent.peer_agent.process_manager import PeerProcessManager
from agent.peer_agent.poller import PeerAgentPoller
from agent.peer_agent.tool import PeerAgentTool
from core.net.http import HttpRequester

logger = logging.getLogger(__name__)


class PeerAgentRegistry:
    def __init__(
        self,
        process_manager: PeerProcessManager,
        poller: PeerAgentPoller,
        requester: HttpRequester,
    ) -> None:
        self._pm = process_manager
        self._poller = poller
        self._requester = requester

    async def discover_all(
        self, peer_configs: list[PeerAgentConfig]
    ) -> list[PeerAgentTool]:
        """
        从 config 直接创建工具（不要求服务器在线）。
        如果服务器恰好在线，用 AgentCard 信息补充 description。
        """
        tools: list[PeerAgentTool] = []
        for cfg in peer_configs:
            # 先用 config 静态信息构建 AgentCard
            card = AgentCard(
                name=cfg.name,
                url=cfg.base_url,
                description=cfg.description,
            )
            # 尝试从运行中的服务器获取更完整的 card（服务器未启动时跳过）
            try:
                live_card = await fetch_agent_card(cfg.base_url, self._requester)
                # 配置拥有进程身份和路由，live card 只补充展示能力
                card = AgentCard(
                    name=cfg.name,
                    url=cfg.base_url,
                    description=live_card.description or cfg.description,
                    skills=live_card.skills,
                )
                logger.info(
                    "[PeerAgentRegistry] 已从服务器获取 AgentCard 展示信息: %s @ %s",
                    cfg.name, cfg.base_url,
                )
            except AgentCardUnavailableError:
                logger.info(
                    "[PeerAgentRegistry] 服务器未在线，使用 config 静态信息: %s @ %s",
                    cfg.name, cfg.base_url,
                )

            tool = PeerAgentTool(
                card=card,
                process_manager=self._pm,
                poller=self._poller,
                requester=self._requester,
            )
            tools.append(tool)
            logger.info(
                "[PeerAgentRegistry] 工具已注册: %s  描述: %.60s...",
                tool.name, tool.description,
            )
        return tools
