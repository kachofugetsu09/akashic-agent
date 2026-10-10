from plugins.ui.contract import UI
from .settings import SETTINGS

inject = (UI, SETTINGS,)


def register(app, context):
    context.require(UI).register_configuration(app, context, SETTINGS, "/api/dashboard/telegram_channel/config")
