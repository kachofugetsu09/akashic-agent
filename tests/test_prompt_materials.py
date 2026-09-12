import asyncio
import json
from collections.abc import Mapping
from contextlib import asynccontextmanager
from pathlib import Path
import shutil
from typing import cast

import pytest
from pydantic import ValidationError

from agent.plugin_composition import ServiceKey
from agent.plugin_composition.bindings import BINDINGS, Bindings
from agent.plugin_composition.channels import CHANNEL_INPUT, ChannelInboundMessage
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import get_current_runtime_snapshot, lease_runtime_snapshot
from bus.event_bus import EventBus
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from plugins.content.plugin import check_text
from plugins.context.materials import MATERIALS
from plugins.context.plugin import Config
from plugins.conversation.plugin import check_origin
from plugins.tools.plugin import ALL_TOOLS, TOOLS
from session.log import MessageLog
from session.artifact_store import ArtifactStore
from session.message import ContentPart, Input, Output, ToolResult
from tests.test_message_push_plugin import storage


def prompt_sources(sources):
    for name in ("prompt", "standard_tools"):
        shutil.copytree(
            Path(__file__).parents[1] / "plugins" / name,
            sources / name,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    settings = (
        sources.parent / "workspace/plugin-data/context-builtin/config.local.toml"
    )
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(
        'summary_source = []\nprompt_sources = {default_prompt = "prompt", skills = "standard_tools"}\n'
    )
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
skill_roots = ("skills",)
async def apply(ctx, config):
    pass
''')
    personal = sources.parent / "workspace/skills/unmanaged"
    personal.mkdir(parents=True, exist_ok=True)
    (personal / "SKILL.md").write_text("非插件技能不得进入新目录")


@asynccontextmanager
async def application(tmp_path):
    sources = tmp_path / "plugins"
    store, log = storage(tmp_path / "workspace")
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
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            async with snapshot.composition_root.context.require(MATERIALS).bind() as view:
                first = await view.prepare(original, source)
                second = await view.prepare(original, source)
                assert first == second
                assert "唯一人格甲" in first["system_prompt"]
                assert "load_skill" not in first["system_prompt"]
                assert ("Telegram 渲染限制" in first["system_prompt"]) == (channel == "telegram_bot")
                environment = next(part["text"] for part in first["reminders"] if part["name"] == "environment")
                assert accepted.recorded_at.astimezone().isoformat() in environment
                assert "input_id: input" in environment
                assert "time_basis" in environment
                assert ("channel_origin" in environment) == (channel is not None)
                assert "Client Surface" not in str(first)
                assert "example" in first["system_prompt"]
                assert "非插件技能" not in str(first)
                base_directory = next(line.removeprefix("资源目录：") for line in first["system_prompt"].splitlines()
                                      if line.startswith("资源目录："))
                assert (Path(base_directory) / "resource.txt").read_text() == "resource-a"
                assert "读取 resource.txt" in first["system_prompt"]
                (tmp_path / "workspace/memory/VEDA.md").write_text("唯一人格乙")
                third = await view.prepare(original, source)
                assert "唯一人格乙" in third["system_prompt"] and "唯一人格甲" in first["system_prompt"]
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
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            async with snapshot.composition_root.context.require(MATERIALS).bind() as view:
                with pytest.raises(RuntimeError, match=r"persona\.py --workspace"):
                    await view.prepare((), "conversation")
        assert not veda.exists() if payload is None else veda.read_bytes() == payload
        assert not (tmp_path / "workspace/memory/veda-backups").exists()


@pytest.mark.asyncio
async def test_load_skill_reopens_original_tree_after_source_removal_and_restart(tmp_path):
    async with application(tmp_path) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            reference = ctx.require(TOOLS).bind(
                ctx.require(ALL_TOOLS)().select("load_skill"), ctx.require(BINDINGS)
            )
            metadata = ctx.require(BINDINGS).describe(reference, TOOLS)
            state = cast(Mapping[str, object], metadata["state"])
            assert set(cast(tuple[str, ...], state["skills"])) == {"example"}
            catalog_id = snapshot.asset_catalog_generation_id
            assert catalog_id is not None
            catalog = host._asset_host.get(catalog_id)
            assert catalog is not None
            asset = next(
                item
                for item in catalog.assets
                if item.owner_id == "fixture_skills" and item.category == "skills"
            )
            original_root = asset.root_dir / "example"
        # 原安装改变后，工具打开的是 capture 已归档的完整资源。
        (tmp_path / "plugins/fixture_skills/skills/example/resource.txt").write_text("resource-b")
        (tmp_path / "plugins/fixture_skills/skills/example/SKILL.md").write_text("---\ndescription: updated\n---\n新版指令")
    assert not original_root.exists()
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
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            replacement = ctx.require(TOOLS).bind(
                ctx.require(ALL_TOOLS)().select("load_skill"), ctx.require(BINDINGS)
            )
            assert replacement != reference
    finally:
        await host.terminate_all()
        log.close()
        store.close()
    shutil.rmtree(tmp_path / "plugins")
    log = MessageLog(tmp_path / "workspace/sessions.db")
    store = ArtifactStore(tmp_path / "workspace/sessions.db")
    artifacts = ChannelAttachmentArtifactStore(
        workspace=tmp_path / "workspace", metadata_store=store
    )
    host = PluginManager(
        [],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home/cache",
        message_log=log,
        channel_attachment_store=artifacts,
    )
    try:
        bindings = Bindings(log, host._archive, host.open_binding)
        async with bindings.open(replacement, TOOLS) as (tools, metadata):
            async with tools.open(metadata) as tool:
                newer = await tool.invoke("new", await tool.prepare({"skill": "example"}))
                current = cast(Mapping[str, object], json.loads(cast(str, newer.parts[0].value)))
                assert current["instructions"] == "新版指令"
                assert (Path(cast(str, current["base_directory"])) / "resource.txt").read_text() == "resource-b"
        async with bindings.open(reference, TOOLS) as (tools, metadata):
            assert "fixture_skills" not in get_current_runtime_snapshot().generations
            async with tools.open(metadata) as tool:
                arguments = await tool.prepare({"skill": "example"})
                result = await tool.invoke("original", arguments)
                assert result.outcome == "success"
                value = cast(Mapping[str, object], json.loads(cast(str, result.parts[0].value)))
                root = Path(cast(str, value["base_directory"]))
                assert (root / "resource.txt").read_text() == "resource-a"
                assert value["source_id"] == "fixture_skills"
                assert (await tool.invoke("unknown", await tool.prepare({"skill": "unmanaged"}))).outcome == "error"
                # 损坏已发布树必须报错；不能从安装路径补齐或伪造成功。
                (root / "resource.txt").chmod(0o600)
                (root / "resource.txt").write_text("tampered")
                with pytest.raises(RuntimeError, match="文件树损坏"):
                    await tool.invoke("retry", arguments)
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
            ).replace('    await ctx.provide(ServiceKey("tools.cleanup.v1"), shell_cleanup)\n', '')
        )

    async with reply_application(tmp_path, replying=True, extra_sources=sources) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
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
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            calls = snapshot.composition_root.context.require(ServiceKey("fixture.calls"))
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
