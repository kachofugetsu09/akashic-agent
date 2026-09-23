import asyncio
import json
from collections.abc import Mapping
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
import shutil
from typing import cast

import pytest

from tests.fixtures.plugin_workspace import initialize_plugin_workspace

from agent.plugin_composition.config_input import save_config
from pydantic import ValidationError

from agent.plugin_composition import CompositionError, FiberState, ServiceKey
from agent.plugin_composition.assets import INSTALLED_ASSETS
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.channels import CHANNEL_INPUT, ChannelInboundMessage
from agent.plugins.manager import PluginManager
from agent.plugin_composition.assets import InstalledAsset
from bus.event_bus import EventBus
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from plugins.content.plugin import check_text
from plugins.context.materials import MATERIALS
from plugins.context.plugin import Config
from plugins.conversation.plugin import check_origin
from plugins.tools.plugin import ALL_TOOLS, TOOLS
from plugins.sources.plugin import SOURCES
from session.log import MessageLog
from session.artifact_store import ArtifactStore
from session.message import ContentPart, Input, Output, ToolResult


def _system_prompt(material: Mapping[str, object]) -> str:
    """读取并窄化 prompt 材料中的系统提示。"""
    value = material["system_prompt"]
    if not isinstance(value, str):
        raise AssertionError("system_prompt 必须是字符串")
    return value


def _environment_reminder(material: Mapping[str, object]) -> str:
    """读取并窄化 prompt 材料中的 environment 提醒。"""
    reminders = material["reminders"]
    if not isinstance(reminders, (list, tuple)):
        raise AssertionError("reminders 必须是列表")
    for reminder in reminders:
        if not isinstance(reminder, Mapping) or reminder.get("name") != "environment":
            continue
        value = reminder.get("text")
        if not isinstance(value, str):
            raise AssertionError("environment reminder 文本必须是字符串")
        return value
    raise AssertionError("缺少 environment reminder")


def prompt_sources(sources):
    for name in ("assets", "prompt", "standard_tools"):
        shutil.copytree(
            Path(__file__).parents[1] / "plugins" / name,
            sources / name,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    settings = (
        sources.parent / "workspace/plugin-data/context-builtin"
    )
    settings.parent.mkdir(parents=True, exist_ok=True)
    save_config(settings, {"summary_source": [], "prompt_sources": {"default_prompt": "prompt", "skills": "standard_tools"}})
    veda = sources.parent / "workspace/memory/VEDA.md"
    veda.parent.mkdir(parents=True, exist_ok=True)
    veda.write_text("唯一人格甲")
    skill = sources / "fixture_skills/skills/example"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\ndescription: fixture task\nalways: true\n---\n读取 resource.txt，保留原内容。")
    (skill / "resource.txt").write_text("resource-a")
    (sources / "fixture_skills/plugin.py").write_text('''api_version = 3
name = "fixture_skills"
version = "1.0.0"
from agent.plugin_composition.assets import INSTALLED_ASSETS
inject = (INSTALLED_ASSETS,)
async def apply(ctx):
    await ctx.require(INSTALLED_ASSETS).register(ctx, "skills", "skills")
''')
    personal = sources.parent / "workspace/skills/unmanaged"
    personal.mkdir(parents=True, exist_ok=True)
    (personal / "SKILL.md").write_text("非插件技能不得进入新目录")


@asynccontextmanager
async def application(tmp_path):
    sources = tmp_path / "plugins"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    initialize_plugin_workspace(workspace)
    log = MessageLog(workspace / "sessions.db")
    store = ArtifactStore(workspace / "sessions.db")
    for name in ("content", "context", "tools"):
        shutil.copytree(Path(__file__).parents[1] / "plugins" / name, sources / name,
                        ignore=shutil.ignore_patterns("__pycache__"))
    prompt_sources(sources)
    artifacts = ChannelAttachmentArtifactStore(
        workspace=tmp_path / "workspace", metadata_store=store
    )
    host = PluginManager(
        [sources],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home/cache",
        message_log=log,
        channel_attachment_store=artifacts,
    )
    try:
        await host.load_all()
        yield log, host
    finally:
        await host.terminate_all()
        log.close()
        store.close()


async def _installed_assets(host: PluginManager) -> tuple[InstalledAsset, ...]:
    """Read assets through the actual active fixture consumer."""
    generation = host.generation("fixture_skills")
    assert generation is not None and generation.fiber is not None
    context = generation.fiber.context
    async with context.runtime_scope():
        return context.require(INSTALLED_ASSETS)(context)


@pytest.mark.asyncio
@pytest.mark.parametrize("source,channel", [("conversation", "akashic"), ("programmatic", "programmatic"),
                                           ("wake", None), ("conversation", "telegram_bot")])
async def test_prompt_reads_veda_and_fixed_input_time_without_rewriting_messages(tmp_path, source, channel):
    async with application(tmp_path) as (log, host):
        writer = log.writer("s", author="user", source=source, body_types=(Input,),
                            content={"text": check_text, "channel.origin": check_origin})
        parts = (ContentPart("text", "今天做什么"),)
        if channel:
            parts += (ContentPart("channel.origin", {"channel": channel, "chat_id": "room", "sender": "u"}),)
        accepted = writer.append("input", Input(parts))
        log.writer("s", author="user", source="unrelated", body_types=(Input,), content={"text": check_text}).append(
            "other", Input((ContentPart("text", "另一个来源"),)))
        original = log.reader("s").snapshot()
        root = host.live_root
        assert root is not None
        async with root.context.require(MATERIALS).bind() as view:
                first = await view.prepare(original, source)
                second = await view.prepare(original, source)
                assert first == second
                first_prompt = _system_prompt(first)
                assert "唯一人格甲" in first_prompt
                assert "load_skill" not in first_prompt
                assert ("Telegram 渲染限制" in first_prompt) == (channel == "telegram_bot")
                environment = _environment_reminder(first)
                assert accepted.recorded_at.astimezone().isoformat() in environment
                assert "input_id: input" in environment
                assert "time_basis" in environment
                assert ("channel_origin" in environment) == (channel is not None)
                assert "Client Surface" not in str(first)
                assert "example" in first_prompt
                assert "非插件技能" not in str(first)
                base_directory = next(line.removeprefix("资源目录：") for line in first_prompt.splitlines()
                                      if line.startswith("资源目录："))
                assert (Path(base_directory) / "resource.txt").read_text() == "resource-a"
                assert "读取 resource.txt" in first_prompt
                (tmp_path / "workspace/memory/VEDA.md").write_text("唯一人格乙")
                third = await view.prepare(original, source)
                third_prompt = _system_prompt(third)
                assert "唯一人格乙" in third_prompt and "唯一人格甲" in first_prompt
                assert log.reader("s").snapshot() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [None, b" \n", b"\xff"])
async def test_prompt_fails_on_missing_or_corrupt_veda_without_reset(tmp_path, payload):
    async with application(tmp_path) as (log, host):
        veda = tmp_path / "workspace/memory/VEDA.md"
        if payload is None:
            veda.unlink()
        else:
            veda.write_bytes(payload)
        root = host.live_root
        assert root is not None
        async with root.context.require(MATERIALS).bind() as view:
                with pytest.raises(RuntimeError, match=r"persona\.py --workspace"):
                    await view.prepare((), "conversation")
        assert not veda.exists() if payload is None else veda.read_bytes() == payload
        assert not (tmp_path / "workspace/memory/veda-backups").exists()


@pytest.mark.asyncio
async def test_skill_catalog_cache_still_requires_the_calling_task_lease(tmp_path):
    async with application(tmp_path) as (_, host):
        root = host.live_root
        assert root is not None
        generation = host.generation("standard_tools")
        assert generation is not None and generation.fiber is not None
        context = generation.fiber.context
        async with context.runtime_scope():
            service = context.require(ServiceKey("standard_tools.skill_inspection.v1"))
            assert [item["name"] for item in service.list_skills()] == ["example"]

            async def inherited_task():
                return service.list_skills()

            with pytest.raises(CompositionError, match="授权需要当前 Context"):
                await asyncio.create_task(inherited_task())
        with pytest.raises(CompositionError, match="授权需要当前 Context"):
            service.list_skills()


@pytest.mark.asyncio
async def test_load_skill_uses_new_stable_tree_after_restart(tmp_path):
    async with application(tmp_path) as (log, host):
        root = host.live_root
        assert root is not None
        ctx = root.context
        reference = await ctx.require(TOOLS).bind_scoped(
                ctx.require(ALL_TOOLS)().select("load_skill"), ctx.require(BINDINGS)
            )
        metadata = ctx.require(BINDINGS).describe(reference, TOOLS)
        state = cast(Mapping[str, object], metadata["state"])
        assert set(cast(tuple[str, ...], state["skills"])) == {"example"}
        assets = await _installed_assets(host)
        asset = next(
                item
                for item in assets
                if item.owner_id == "fixture_skills" and item.category == "skills"
            )
        original_root = asset.root_dir / "example"
        # 安装改变后，下一次 stable 只使用新生成的资源树。
        (tmp_path / "plugins/fixture_skills/skills/example/resource.txt").write_text("resource-b")
        (tmp_path / "plugins/fixture_skills/skills/example/SKILL.md").write_text("---\ndescription: updated\n---\n新版指令")
        await host.reconcile_changed()
    assert (original_root / "resource.txt").read_text() == "resource-a"
    log = MessageLog(tmp_path / "workspace/sessions.db")
    store = ArtifactStore(tmp_path / "workspace/sessions.db")
    artifacts = ChannelAttachmentArtifactStore(
        workspace=tmp_path / "workspace", metadata_store=store
    )
    host = PluginManager(
        [tmp_path / "plugins"],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home/cache",
        message_log=log,
        channel_attachment_store=artifacts,
    )
    try:
        await host.load_all()
        root = host.live_root
        assert root is not None
        ctx = root.context
        bindings = ctx.require(BINDINGS)
        replacement = await ctx.require(TOOLS).bind_scoped(
                ctx.require(ALL_TOOLS)().select("load_skill"), bindings
            )
        assert replacement != reference
        async with bindings.open(replacement, TOOLS) as (tools, metadata):
            async with tools.open(metadata) as tool:
                newer_arguments = await tool.prepare({"skill": "example"})
                assert isinstance(newer_arguments, Mapping)
                newer = await tool.invoke("new", newer_arguments)
                current = cast(Mapping[str, object], json.loads(cast(str, newer.parts[0].value)))
                assert current["instructions"] == "新版指令"
                assert (Path(cast(str, current["base_directory"])) / "resource.txt").read_text() == "resource-b"
    finally:
        await host.terminate_all()
        log.close()
        store.close()


def test_context_grants_can_be_disabled_in_toml_and_reject_bad_owners():
    import tomllib
    assert Config.model_validate(tomllib.loads("summary_source = []\nprompt_sources = {}\n")).summary_source == ()
    for value in ({"summary_source": ["only-one"]}, {"summary_source": ["", "owner"]},
                  {"prompt_sources": {"prompt": " "}}, {"summary_source": None}):
        with pytest.raises(ValidationError):
            Config.model_validate(value)


@pytest.mark.asyncio
async def test_default_reply_uses_prompt_and_real_skill_tool_with_provider_view(
    tmp_path,
):
    from datetime import UTC, datetime
    from tests.test_default_reply import application as reply_application

    def sources(root):
        prompt_sources(root)
        path = root / "test_provider/plugin.py"
        path.write_text(
                path.read_text().replace(
                    '"write_evidence", {})', '"load_skill", {"skill": "example"})'
                ).replace(
                    '    await ctx.provide(ServiceKey("tools.cleanup.v1"), partial(\n'
                    '        shell_cleanup, ctx, ShellOwners(ctx), ctx.require(TASKS).open(ctx),\n'
                    '    ))\n',
                    '',
                )
        )

    async with reply_application(tmp_path, replying=True, extra_sources=sources) as (log, host):
        root = host.live_root
        assert root is not None
        ctx = root.context
        conversation = host.generation("conversation")
        assert conversation is not None and conversation.fiber is not None
        assert conversation.fiber.state is FiberState.ACTIVE
        source_service = ctx.require(SOURCES)
        async with aclosing(source_service.changes()) as changes:
            async with asyncio.timeout(5):
                async for entries in changes:
                    if any(item.name == "conversation" for item in entries):
                        break
        await ctx.require(CHANNEL_INPUT)("test:room", "input", ChannelInboundMessage(
                "test", "user", "room", "读取 example 技能", datetime(2026, 9, 7, tzinfo=UTC), {}))
        async def completed():
            async for _ in log.catalog().follow():
                rows = log.reader("test:room").snapshot()
                if any(isinstance(row.body, Output) and row.body.finish == "complete" for row in rows):
                    return rows
        rows = await asyncio.wait_for(completed(), 10)
        assert rows is not None
        assert [type(row.body) for row in rows] == [Input, Output, ToolResult, Output]
        root = host.live_root
        assert root is not None
        calls = root.context.require(ServiceKey("fixture.calls"))
        assert "唯一人格甲" in str(calls[0].messages)
        environment = calls[0].messages[-1]["content"]
        assert "input_id: input" in environment
        assert rows[0].recorded_at.astimezone().isoformat() in environment
        assert "fixture task" in str(calls[0].messages)
        assert "load_skill" in str(calls[0].tools)
        result = cast(
                Mapping[str, object], json.loads(cast(str, rows[2].body.parts[0].value))
            )
        assert (
                Path(cast(str, result["base_directory"])) / "resource.txt"
            ).read_text() == "resource-a"
        assert result["instructions"] == "读取 resource.txt，保留原内容。"
