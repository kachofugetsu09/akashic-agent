"""Independent composition keys consumed by the Akashic client plugin.

The client plugin names the capabilities it uses by their stable ServiceKey
identity.  It does not import another plugin's implementation.  The channel
host resolves these keys inside each request scope.
"""

from __future__ import annotations

from plugins.ledger.contract import MESSAGE_CATALOG, SESSION_ADMIN
from plugins.ui.contract import (
    WEB_UI as WEB_UI,
)
from plugins.reply.contract import (
    REPLY_STATUS as REPLY_STATUS,
    ReplyStatus as ReplyStatusReader,
)
from plugins.channels.contract import (
    CHANNEL_STEER as CHANNEL_STEER,
    ChannelSteer as ChannelSteer,
)
from plugins.ui.contract import (
    MESSAGE_DISPLAY as MESSAGE_DISPLAY,
    PLUGIN_UI as PLUGIN_UI,
)

# 聊天启动只等待消息、展示与插件界面这些通用能力；模型、检视等由各自插件经插件界面自带。
CLIENT_CAPABILITIES = (
    MESSAGE_CATALOG,
    MESSAGE_DISPLAY,
    PLUGIN_UI,
    WEB_UI,
    SESSION_ADMIN,
)


__all__ = [
    "CLIENT_CAPABILITIES",
    "CHANNEL_STEER",
    "ChannelSteer",
    "MESSAGE_DISPLAY",
    "PLUGIN_UI",
    "SESSION_ADMIN",
    "WEB_UI",
    "ReplyStatusReader",
    "REPLY_STATUS",
]
