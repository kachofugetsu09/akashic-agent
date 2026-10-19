"""Akasha 设置页的配置路由；检索与学习记录在聊天里随消息展示，不另设查看页。"""

from __future__ import annotations

from fastapi import FastAPI

from agent.plugin_composition import DashboardContext
from plugins.ui.contract import UI
from .settings import SETTINGS

inject = (UI, SETTINGS)


def register(app: FastAPI, context: DashboardContext) -> None:
    context.require(UI).register_configuration(app, context, SETTINGS, "/api/dashboard/akasha/config")
