"""Akashic 聊天目录的公开领域合同：界面 HTTP 入口与其他插件共用同一组领域操作。

本合同只描述 Web 聊天通道自己拥有的事实（会话目录、消息页、查找、标题与软删、
导航置顶、附件登记与读取）；消费者只 import 本模块，不触碰路由或通道实现。
定义放在 services.py：UI 宿主自身也不得 import 本插件 contract（R12）。
"""
from __future__ import annotations

from .services import CHAT_SESSIONS, ArtifactContent, ChatRequestError, ChatSessions, ChatSessionsHost

__all__ = ["CHAT_SESSIONS", "ArtifactContent", "ChatRequestError", "ChatSessions", "ChatSessionsHost"]
