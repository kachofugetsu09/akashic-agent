from __future__ import annotations

from typing import TYPE_CHECKING

from agent.plugins import Plugin
from plugins.feishu.channel import FeishuChannel
from plugins.feishu.config import load_feishu_config

if TYPE_CHECKING:
    from infra.channels.contract import Channel


class FeishuPlugin(Plugin):
    name = "feishu"
    desc = "飞书私聊渠道"

    def channels(self) -> list["Channel"]:
        config = load_feishu_config(plugin_dir=self.context.plugin_dir)
        if not config.app_id or not config.app_secret:
            return []
        return [
            FeishuChannel(
                app_id=config.app_id,
                app_secret=config.app_secret,
                allow_from=config.allow_from,
                domain=config.domain,
            )
        ]
