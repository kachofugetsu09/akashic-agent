from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.core.response_parser import ResponseMetadata
from agent.lifecycle.types import AfterReasoningCtx, PromptRenderCtx
from agent.plugins.context import PluginContext, PluginKVStore
from agent.plugins.manager import PluginManager
from agent.plugins.registry import plugin_registry
from bus.event_bus import EventBus
from plugins.meme.plugin import MemePlugin, MemePromptModule


@pytest.fixture(autouse=True)
def _clean_registry():
    plugin_registry._handlers._handlers.clear()
    plugin_registry._classes.clear()
    plugin_registry._instances.clear()
    yield
    plugin_registry._handlers._handlers.clear()
    plugin_registry._classes.clear()
    plugin_registry._instances.clear()


def _write_meme_workspace(workspace: Path) -> Path:
    memes = workspace / "memes"
    (memes / "shy").mkdir(parents=True)
    image = memes / "shy" / "001.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    (memes / "manifest.json").write_text(
        json.dumps(
            {"categories": {"shy": {"desc": "害羞", "enabled": True}}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return image


async def _make_plugin(tmp_path: Path) -> MemePlugin:
    plugin_dir = tmp_path / "plugins" / "meme"
    plugin_dir.mkdir(parents=True)
    plugin = MemePlugin()
    plugin.context = PluginContext(
        event_bus=None,
        tool_registry=None,
        plugin_id="meme",
        plugin_dir=plugin_dir,
        kv_store=PluginKVStore(plugin_dir / ".kv.json"),
        workspace=tmp_path,
    )
    await plugin.initialize()
    return plugin


@pytest.mark.asyncio
async def test_meme_prompt_module_injects_bottom_section(tmp_path: Path) -> None:
    _write_meme_workspace(tmp_path)
    plugin = await _make_plugin(tmp_path)
    module = plugin.prompt_render_modules_bottom()[0]
    assert isinstance(module, MemePromptModule)

    ctx = PromptRenderCtx(
        session_key="telegram:1",
        channel="telegram",
        chat_id="1",
        content="你好",
        media=None,
        timestamp=datetime.now(timezone.utc),
        history=[],
        skill_names=[],
        retrieved_memory_block="",
        disabled_sections=set(),
        turn_injection_prompt="",
    )
    frame = SimpleNamespace(slots={"prompt:ctx": ctx})

    await module.run(frame)

    assert ctx.system_sections_bottom[0].name == "memes"
    assert "<meme:shy>" in ctx.system_sections_bottom[0].content


@pytest.mark.asyncio
async def test_plugin_manager_collects_meme_prompt_module_before_initialize(tmp_path: Path) -> None:
    _write_meme_workspace(tmp_path)
    plugin_dir = tmp_path / "plugin_src" / "meme"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(
        "from plugins.meme.plugin import MemePlugin\n"
        "class TestMemePlugin(MemePlugin):\n"
        "    name = 'meme'\n",
        encoding="utf-8",
    )
    manager = PluginManager(
        [plugin_dir.parent],
        event_bus=EventBus(),
        workspace=tmp_path,
    )

    await manager.load_all()

    assert manager.loaded_count == 1
    assert len(manager.prompt_render_modules_bottom) == 1


@pytest.mark.asyncio
async def test_meme_plugin_decorates_after_reasoning(tmp_path: Path) -> None:
    image = _write_meme_workspace(tmp_path)
    plugin = await _make_plugin(tmp_path)
    ctx = AfterReasoningCtx(
        session_key="telegram:1",
        channel="telegram",
        chat_id="1",
        tools_used=(),
        thinking=None,
        response_metadata=ResponseMetadata(raw_text="好的 <meme:shy>"),
        streamed=False,
        tool_chain=(),
        context_retry={},
        reply="好的 <meme:shy>",
    )

    out = await plugin.decorate_meme(ctx)

    assert out.reply == "好的"
    assert out.media == [str(image)]
    assert out.meme_tag == "shy"
