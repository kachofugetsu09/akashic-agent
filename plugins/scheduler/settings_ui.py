"""设置页路由；dashboard.py 已是任务展示投影，被 inspection 在加载期导入，不能再反向导入 plugin。"""
from plugins.ui.contract import UI
from .plugin import SETTINGS

inject = (UI, SETTINGS,)


def register(app, context):
    context.require(UI).register_configuration(app, context, SETTINGS, "/api/dashboard/scheduler/config")
