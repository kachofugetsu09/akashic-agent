"""stable_view 插件：/stable 命令渲染当前组合的 ASCII 依赖树。"""
import pytest

from agent.plugin_composition import CompositionRoot, PluginRuntime
from agent.plugin_composition.commands import COMMANDS
from agent.plugins.snapshot import RuntimeSnapshotCompiler, RuntimeSnapshotStore
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
                         "dependencies": ["core.commands"],
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


def test_format_stable_catalog_mcp_unavailable() -> None:
    catalog = _catalog()
    catalog.pop("mcp_servers")
    catalog["mcp_unavailable"] = {
        "code": "mcp_catalog_unavailable", "message": "no session",
    }
    text = format_stable_catalog(catalog)
    assert "mcp_catalog_unavailable" in text
    assert "stable_view" in text


async def _mount_stable_view(tmp_path, monkeypatch, catalog) -> CompositionRoot:
    store = RuntimeSnapshotStore()
    snapshot = RuntimeSnapshotCompiler().compile({})
    store.install(snapshot)
    root = CompositionRoot("selected")
    root._bind_runtime_scope_acquirer(store.acquire)
    if catalog is not None:
        monkeypatch.setattr(
            stable_view, "build_stable_plugin_catalog", lambda _snapshot: catalog,
        )
    await root.mount(commands_plugin.apply, name="commands")
    await root.mount(
        stable_view.apply,
        name="stable_view",
        inject=(COMMANDS,),
        runtime=PluginRuntime(
            plugin_id="stable_view", generation_id="gen-stable",
            plugin_dir=tmp_path, data_dir=tmp_path / "data",
            workspace=tmp_path, config={},
        ),
    )
    return root


@pytest.mark.asyncio
async def test_stable_command_returns_tree(tmp_path, monkeypatch) -> None:
    root = await _mount_stable_view(tmp_path, monkeypatch, _catalog())
    try:
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
async def test_stable_command_reports_reader_failure(tmp_path, monkeypatch) -> None:
    def broken(_snapshot: object) -> dict[str, object]:
        raise RuntimeError("no scope")

    monkeypatch.setattr(stable_view, "build_stable_plugin_catalog", broken)
    root = await _mount_stable_view(tmp_path, monkeypatch, None)
    try:
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
