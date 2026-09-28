from agent.plugin_contracts.configuration import register_routes
from .settings import SETTINGS

inject = (SETTINGS,)


def register(app, context):
    register_routes(app, context, SETTINGS, "/api/dashboard/telegram_channel/config")
