import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
import shutil

import pytest
from pydantic import ValidationError

from agent.persona import VedaLoadError
from agent.plugin_composition import ServiceKey
from agent.plugin_composition.bindings import BINDINGS, Bindings
from agent.plugin_composition.channels import CHANNEL_INPUT, ChannelInboundMessage
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import get_current_runtime_snapshot, lease_runtime_snapshot
from bus.event_bus import EventBus
from plugins.content.plugin import check_text
from plugins.context.materials import MATERIALS
from plugins.context.plugin import Config
from plugins.conversation.plugin import check_origin
from plugins.tools.plugin import TOOLS
from session.log import MessageLog
from session.message import ContentPart, Input, Output, ToolResult


def prompt_sources(sources):
    for name in ("prompt", "skills"):
        shutil.copytree(Path(__file__).parents[1] / "plugins" / name, sources / name,
                        ignore=shutil.ignore_patterns("__pycache__"))
    settings = sources.parent / "workspace/plugin-data/context-builtin/config.local.toml"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text('summary_source = []\nprompt_sources = {default_prompt = "prompt"}\n')
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
    for name in ("context", "tools"):
        shutil.copytree(Path(__file__).parents[1] / "plugins" / name, sources / name,
                        ignore=shutil.ignore_patterns("__pycache__"))
    prompt_sources(sources)
    log = MessageLog(tmp_path / "sessions.db")
    host = PluginManager([sources], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home/cache", message_log=log)
    try:
        await host.load_all()
        yield log, host
    finally:
        await host.terminate_all()
        log.close()


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
                assert "唯一人格甲" in first.system_prompt
                assert "load_skill" not in first.system_prompt
                assert ("Telegram 渲染限制" in first.system_prompt) == (channel == "telegram_bot")
                environment = next(part.value for part in first.context if part.kind == "environment")
                assert environment["request_time"] == accepted.recorded_at.astimezone().isoformat()
                assert environment["input_id"] == "input"
                assert "time_basis" in environment
                assert ("channel_origin" in environment) == (channel is not None)
                assert "Client Surface" not in str(first)
                catalog = next(part.value for part in first.context if part.kind == "skills")
                assert [entry["name"] for entry in catalog["skills"]] == ["example"]
                assert "非插件技能" not in str(first)
                active = catalog["active_skills"][0]
                assert (Path(active["base_directory"]) / "resource.txt").read_text() == "resource-a"
                assert "读取 resource.txt" not in first.system_prompt
                (tmp_path / "workspace/memory/VEDA.md").write_text("唯一人格乙")
                third = await view.prepare(original, source)
                assert "唯一人格乙" in third.system_prompt and "唯一人格甲" in first.system_prompt
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
                with pytest.raises(VedaLoadError, match="veda-reset"):
                    await view.prepare((), "conversation")
        assert not veda.exists() if payload is None else veda.read_bytes() == payload
        assert not (tmp_path / "workspace/memory/veda-backups").exists()


@pytest.mark.asyncio
async def test_load_skill_reopens_original_tree_after_source_removal_and_restart(tmp_path):
    async with application(tmp_path) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            reference = ctx.require(TOOLS).bind("load_skill", ctx.require(BINDINGS))
            metadata = ctx.require(BINDINGS).describe(reference, TOOLS)
            assert set(metadata["state"]["skills"]) == {"example"}
            original_root = snapshot.plugin_skill_index.records["example"].root_dir
        # 原安装改变后，工具打开的是 capture 已归档的完整资源。
        (tmp_path / "plugins/fixture_skills/skills/example/resource.txt").write_text("resource-b")
        (tmp_path / "plugins/fixture_skills/skills/example/SKILL.md").write_text("---\ndescription: updated\n---\n新版指令")
    assert not original_root.exists()
    log = MessageLog(tmp_path / "sessions.db")
    host = PluginManager([tmp_path / "plugins"], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home/cache", message_log=log)
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            replacement = ctx.require(TOOLS).bind("load_skill", ctx.require(BINDINGS))
            assert replacement != reference
    finally:
        await host.terminate_all()
        log.close()
    shutil.rmtree(tmp_path / "plugins")
    log = MessageLog(tmp_path / "sessions.db")
    host = PluginManager([], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home/cache", message_log=log)
    try:
        bindings = Bindings(log, host._archive, host.open_binding)
        async with bindings.open(replacement, TOOLS) as (tools, metadata):
            async with tools.open(metadata) as tool:
                newer = await tool.invoke("new", await tool.prepare({"skill": "example"}))
                current = json.loads(newer.parts[0].value)
                assert current["instructions"] == "新版指令"
                assert (Path(current["base_directory"]) / "resource.txt").read_text() == "resource-b"
        async with bindings.open(reference, TOOLS) as (tools, metadata):
            assert "fixture_skills" not in get_current_runtime_snapshot().generations
            async with tools.open(metadata) as tool:
                arguments = await tool.prepare({"skill": "example"})
                result = await tool.invoke("original", arguments)
                assert result.outcome == "success"
                value = json.loads(result.parts[0].value)
                root = Path(value["base_directory"])
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


def test_context_grants_can_be_disabled_in_toml_and_reject_bad_owners():
    import tomllib
    assert Config.model_validate(tomllib.loads("summary_source = []\nprompt_sources = {}\n")).summary_source == ()
    for value in ({"summary_source": ["only-one"]}, {"summary_source": ["", "owner"]},
                  {"prompt_sources": {"prompt": " "}}, {"summary_source": None}):
        with pytest.raises(ValidationError):
            Config.model_validate(value)


@pytest.mark.asyncio
@pytest.mark.parametrize("restricted", [False, True])
async def test_default_reply_uses_prompt_and_real_skill_tool_with_menu_authority(tmp_path, restricted):
    from datetime import UTC, datetime
    from tests.test_default_reply import application as reply_application

    def sources(root):
        prompt_sources(root)
        if restricted:
            path = root / "reply/plugin.py"
            path.write_text(path.read_text().replace("tools: tuple[str, ...] | None = None",
                                                   'tools: tuple[str, ...] | None = ("write_evidence",)'))
        else:
            path = root / "test_provider/plugin.py"
            path.write_text(path.read_text().replace('"write_evidence", {})', '"load_skill", {"skill": "example"})'))

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
        assert [type(row.body) for row in rows] == [Input, Output, ToolResult, Output]
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            calls = snapshot.composition_root.context.require(ServiceKey("fixture.calls"))
            assert "唯一人格甲" in str(calls[0].messages)
            context = json.loads(calls[0].messages[-1]["content"])["context"]
            environment = next(part["value"] for part in context if part["kind"] == "environment")
            assert environment["input_id"] == "input"
            assert environment["request_time"] == rows[0].recorded_at.astimezone().isoformat()
            assert "fixture task" in str(calls[0].messages)
            assert ("load_skill" in str(calls[0].tools)) != restricted
            if not restricted:
                result = json.loads(rows[2].body.parts[0].value)
                assert (Path(result["base_directory"]) / "resource.txt").read_text() == "resource-a"
                assert result["instructions"] == "读取 resource.txt，保留原内容。"
