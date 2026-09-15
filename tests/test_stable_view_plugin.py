"""stable_view 插件：/stable 命令渲染当前组合的 ASCII 依赖树。"""
import pytest

from agent.plugin_composition import CompositionRoot, PluginRuntime
from agent.plugin_composition.commands import COMMANDS
from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG
from plugins.commands import plugin as commands_plugin
from plugins.stable_view import plugin as stable_view
from plugins.stable_view.plugin import format_stable_catalog


def _catalog() -> dict[str, object]:
    return {
        "snapshot_id": "0123456789abcdef",
        "plugins": [
            {
                "id": "commands",
                "revision": "revcommands01",
                "generation_id": "gen-commands",
                "api_version": 3,
                "composition": {
                    "ready": True,
                    "fibers": [
                        {"name": "commands", "parent": None, "state": "active",
                         "required": True, "dependencies": [],
                         "missing_services": [], "error": None},
                    ],
                    "incident_count": 0,
                },
            },
            {
                "id": "stable_view",
                "revision": "revstable0001",
                "generation_id": "gen-stable",
                "api_version": 3,
                "composition": {
                    "ready": True,
                    "fibers": [
                        {"name": "stable_view", "parent": None, "state": "active",
                         "required": True,
                         "dependencies": ["core.commands", "core.runtime_catalog.v1"],
                         "missing_services": [], "error": None},
                        {"name": "stable_view.child", "parent": "stable_view",
                         "state": "active", "required": False,
                         "dependencies": [], "missing_services": [], "error": None},
                    ],
                    "incident_count": 0,
                },
            },
        ],
        "mcp_servers": [{"name": "feed"}],
    }


def test_format_stable_catalog_renders_tree() -> None:
    text = format_stable_catalog(_catalog())
    assert "stable snapshot 0123456789abcdef" in text
    assert "stable_view" in text
    assert "└─ stable_view.child" in text
    assert "core.commands" in text
    assert "feed" in text


def test_format_stable_catalog_unavailable() -> None:
    text = format_stable_catalog({"unavailable": {"code": "x", "message": "nope"}})
    assert "暂不可用" in text and "nope" in text


@pytest.mark.asyncio
async def test_stable_command_returns_tree(tmp_path) -> None:
    root = CompositionRoot("selected")
    try:
        await root.context.provide(RUNTIME_CATALOG, _catalog)
        await root.mount(commands_plugin.apply, name="commands")
        await root.mount(
            stable_view.apply,
            name="stable_view",
            inject=(COMMANDS, RUNTIME_CATALOG),
            runtime=PluginRuntime(
                plugin_id="stable_view", generation_id="gen-stable",
                plugin_dir=tmp_path, data_dir=tmp_path / "data",
                workspace=tmp_path, config={},
            ),
        )
        commands = root.context.require(COMMANDS).freeze()
        result = await commands.execute(
            "/stable", session_key="chat", channel="test",
            chat_id="room", sender="user",
        )
        assert result is not None
        assert result.result.kind == "success"
        assert "stable_view.child" in result.result.text
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_stable_command_reports_reader_failure(tmp_path) -> None:
    def broken() -> dict[str, object]:
        raise RuntimeError("no scope")

    root = CompositionRoot("selected")
    try:
        await root.context.provide(RUNTIME_CATALOG, broken)
        await root.mount(commands_plugin.apply, name="commands")
        await root.mount(
            stable_view.apply,
            name="stable_view",
            inject=(COMMANDS, RUNTIME_CATALOG),
            runtime=PluginRuntime(
                plugin_id="stable_view", generation_id="gen-stable",
                plugin_dir=tmp_path, data_dir=tmp_path / "data",
                workspace=tmp_path, config={},
            ),
        )
        commands = root.context.require(COMMANDS).freeze()
        result = await commands.execute(
            "/stable", session_key="chat", channel="test",
            chat_id="room", sender="user",
        )
        assert result is not None
        assert result.result.kind == "error"
        assert "no scope" in result.result.text
    finally:
        await root.dispose()
