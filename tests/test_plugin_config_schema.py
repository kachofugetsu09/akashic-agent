from __future__ import annotations

from pathlib import Path

import pytest

from agent.config import Config
from agent.plugins.manager import PluginManager
from agent.plugins.registry import plugin_registry
from bus.event_bus import EventBus


@pytest.fixture(autouse=True)
def _clean_registry():
    plugin_registry._handlers._handlers.clear()
    plugin_registry._classes.clear()
    plugin_registry._instances.clear()
    yield
    plugin_registry._handlers._handlers.clear()
    plugin_registry._classes.clear()
    plugin_registry._instances.clear()


def _write_typed_plugin(root: Path) -> None:
    plugin_dir = root / "typed"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.py").write_text(
        """
from pydantic import BaseModel
from agent.plugins import Plugin


class TypedConfig(BaseModel):
    api_key: str
    max_results: int = 5


class TypedPlugin(Plugin):
    name = "typed"
    ConfigModel = TypedConfig
""".strip(),
        encoding="utf-8",
    )


def _get_instance(name: str):
    for instance in plugin_registry._instances.values():
        if getattr(instance, "name", None) == name:
            return instance
    raise KeyError(name)


@pytest.mark.asyncio
async def test_plugin_config_model_validates_and_injects_config(tmp_path: Path):
    _write_typed_plugin(tmp_path)
    manager = PluginManager(
        plugin_dirs=[tmp_path],
        event_bus=EventBus(),
        plugin_configs={"typed": {"api_key": "secret", "max_results": 9}},
    )

    await manager.load_all()

    config = _get_instance("typed").context.config
    assert config.api_key == "secret"
    assert config.max_results == 9


@pytest.mark.asyncio
async def test_plugin_config_model_failure_skips_plugin(tmp_path: Path):
    _write_typed_plugin(tmp_path)
    manager = PluginManager(
        plugin_dirs=[tmp_path],
        event_bus=EventBus(),
        plugin_configs={"typed": {"max_results": "bad"}},
    )

    await manager.load_all()

    assert manager.loaded_count == 0


def test_config_load_resolves_plugin_env_and_legacy_qqbot(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[llm]
provider = "openai"

[llm.main]
model = "m"
api_key = "k"

[agent]
system_prompt = "s"

[plugins.typed]
api_key = "${PLUGIN_TOKEN}"

[channels.qqbot]
app_id = "app"
client_secret = "${QQBOT_SECRET}"
allow_from = ["user-openid"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PLUGIN_TOKEN", "plugin-secret")
    monkeypatch.setenv("QQBOT_SECRET", "qq-secret")

    config = Config.load(config_path)

    assert config.plugins["typed"]["api_key"] == "plugin-secret"
    assert config.plugins["qqbot"]["client_secret"] == "qq-secret"
    assert config.plugins["qqbot"]["allow_from"] == ["user-openid"]
