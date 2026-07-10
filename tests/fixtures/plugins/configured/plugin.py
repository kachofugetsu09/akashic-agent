from pydantic import BaseModel

from agent.plugins import Plugin


class ConfiguredModel(BaseModel):
    api_key: str = "test-key"
    max_results: int = 10
    enabled: bool = True


class Configured(Plugin):
    name = "configured"
    ConfigModel = ConfiguredModel
