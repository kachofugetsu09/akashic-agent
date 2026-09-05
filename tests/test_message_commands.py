import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from agent.plugin_composition import ServiceKey
from agent.plugin_composition.channels import CHANNEL_INPUT
from agent.plugins.snapshot import lease_runtime_snapshot
from plugins.conversation.commands import CONVERSATION_COMMANDS
from plugins.conversation.plugin import CONVERSATION
from session.log import MessageWriter
from session.message import Control, Output
from tests.test_channel_input import raw, runtime


def command_plugin(tmp_path, *, read_only=False, recover=True, quiet=False, blocked=False):
    plugin = tmp_path / "plugins/probe_command/plugin.py"
    plugin.parent.mkdir(parents=True)
    plugin.write_text('''
import asyncio
import json
from pathlib import Path
from agent.plugin_composition import ServiceKey
from agent.plugin_composition.commands import COMMANDS, CommandDefinition, CommandResult, CommandRecoveryRequired
api_version = 3
name = "probe_command"
version = "1.0.0"
inject = (COMMANDS,)
async def apply(ctx, config):
    calls = []
    entered, gate = asyncio.Event(), asyncio.Event()
    if not BLOCKED:
        gate.set()
    path = Path(RECEIPT)
    def identity(invocation):
        return [invocation.message_id, invocation.name, invocation.raw_input,
                invocation.session_key, invocation.channel, invocation.chat_id, invocation.sender]
    async def invoke(invocation):
        calls.append(identity(invocation))
        path.write_text(json.dumps(identity(invocation)))
        entered.set()
        await gate.wait()
        return CommandResult("success", RESULT)
    def recover(invocation):
        if not path.exists() or json.loads(path.read_text()) != identity(invocation):
            raise CommandRecoveryRequired("missing or conflicting receipt")
        return CommandResult("success", RESULT)
    await ctx.require(COMMANDS).register(ctx, CommandDefinition(
        name="probe", aliases=("p",), description="read verified identity", handler=invoke,
        read_only=READ_ONLY, recover=RECOVER))
    await ctx.provide(ServiceKey("fixture.command_calls"), calls)
    await ctx.provide(ServiceKey("fixture.command_events"), (entered, gate))
'''.replace('RECEIPT', repr(str(tmp_path / 'receipt.json')))
        .replace('RESULT', repr('' if quiet else 'original result'))
        .replace('READ_ONLY', repr(read_only)).replace('RECOVER', 'recover' if recover else 'None')
        .replace('BLOCKED', repr(blocked)))
    return plugin


async def run(host, *, input_id=None, text="/p hello", resume=None):
    async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
        root = snapshot.composition_root.context
        if input_id is not None:
            await root.require(CHANNEL_INPUT)("probe:room", input_id, replace(raw().message, content=text))
        if resume is not None:
            await root.require(CONVERSATION)("probe:room").resume("retry", resume)
        command = root.require(CONVERSATION_COMMANDS)
        task = await root.require(CONVERSATION)("probe:room").start(command)
    return None if task is None else await task.join()


@pytest.mark.asyncio
@pytest.mark.parametrize("quiet", [False, True])
async def test_command_runs_without_model_and_preserves_identity_and_one_result(tmp_path, quiet):
    command_plugin(tmp_path, read_only=True, quiet=quiet)
    async with runtime(tmp_path) as (log, host, *_):
        result = await run(host, input_id="first")
        assert result.body.finish == ("quiet" if quiet else "complete")
        assert result.author == "app" and result.source == "conversation"
        fact = result.body.parts[0]
        assert fact.kind == "command.result" and fact.value["input_id"] == "first"
        assert fact.value["name"] == "probe"
        assert all(part.kind not in {"model.facts", "context.summary"} for part in result.body.parts)
        assert await run(host, input_id="first") is None
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            calls = snapshot.composition_root.context.require(ServiceKey("fixture.command_calls"))
            assert calls == [["first", "probe", " hello", "probe:room", "probe", "room", "user"]]
        assert len(log.reader("probe:room").snapshot()) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("recoverable", [False, True])
async def test_effect_before_output_failure_recovers_original_handler_or_refuses_without_reexecution(
    tmp_path, monkeypatch, recoverable,
):
    plugin = command_plugin(tmp_path, recover=recoverable)
    append = MessageWriter.append
    def fail_result(writer, message_id, body, **kwargs):
        if isinstance(body, Output):
            raise OSError("injected before command Message commit")
        return append(writer, message_id, body, **kwargs)
    async with runtime(tmp_path) as (log, host, *_):
        with monkeypatch.context() as patch:
            patch.setattr(MessageWriter, "append", fail_result)
            with pytest.raises(OSError, match="before command Message"):
                await run(host, input_id="first")
        assert (tmp_path / 'receipt.json').exists()
        assert isinstance(log.reader("probe:room").snapshot()[-1].body, Control)
        plugin.write_text('raise AssertionError("must open original archive")\n')
        if recoverable:
            result = await run(host, resume="first")
            assert result.body.parts[1].value == "original result"
            assert result.body.finish == "complete"
            assert await run(host) is None
        else:
            from agent.plugin_composition.commands import CommandRecoveryRequired
            with pytest.raises(CommandRecoveryRequired, match="禁止自动重跑"):
                await run(host, resume="first")
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            calls = snapshot.composition_root.context.require(ServiceKey("fixture.command_calls"))
            assert len(calls) == 1
        outputs = [m for m in log.reader("probe:room").snapshot() if isinstance(m.body, Output)]
        assert len(outputs) == int(recoverable)


@pytest.mark.asyncio
async def test_unknown_command_leaves_input_for_default_program(tmp_path):
    async with runtime(tmp_path) as (log, host, *_):
        assert await run(host, input_id="unknown", text="/unknown hello") is None
        assert len(log.reader("probe:room").snapshot()) == 1
        assert log.owner("plugin:conversation").list() == ()


@pytest.mark.asyncio
async def test_new_input_cancels_command_and_recovery_cannot_close_newer_input(tmp_path):
    command_plugin(tmp_path, blocked=True)
    async with runtime(tmp_path) as (log, host, *_):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            entered, gate = snapshot.composition_root.context.require(ServiceKey("fixture.command_events"))
        first = asyncio.create_task(run(host, input_id="first"))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
                await snapshot.composition_root.context.require(CHANNEL_INPUT)(
                    "probe:room", "new", replace(raw().message, content="new question"))
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(first, 2)
            assert await run(host) is None
            rows = log.reader("probe:room").snapshot()
            assert len(rows) == 3
            assert rows[-1].body.finish == "continue"
            assert rows[-1].body.parts[0].value["input_id"] == "first"
            from plugins.conversation.source import needs_reply
            assert needs_reply(rows, "conversation")
            async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
                assert len(snapshot.composition_root.context.require(ServiceKey("fixture.command_calls"))) == 1
        finally:
            gate.set()
            await asyncio.gather(first, return_exceptions=True)


@pytest.mark.asyncio
async def test_default_reply_short_circuits_command_before_model_or_tool(tmp_path):
    from tests.test_default_reply import application
    command_plugin(tmp_path, read_only=True)
    async with application(tmp_path, replying=True) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            await snapshot.composition_root.context.require(CHANNEL_INPUT)(
                "probe:room", "first", replace(raw().message, content="/probe"))
        async def finished():
            async for _ in log.catalog().follow():
                rows = log.reader("probe:room").snapshot()
                if any(isinstance(row.body, Output) for row in rows):
                    return rows
        rows = await asyncio.wait_for(finished(), 3)
        assert len(rows) == 2 and rows[-1].body.finish == "complete"
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            assert snapshot.composition_root.context.require(ServiceKey("fixture.calls")) == []
        assert not (tmp_path / "effect.txt").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [False, True])
async def test_abandoned_command_never_starts_or_replays_after_later_input(tmp_path, monkeypatch, started):
    from agent.plugin_composition.commands import CommandRegistry
    command_plugin(tmp_path, blocked=started, read_only=not started)
    async with runtime(tmp_path) as (log, host, *_):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            entered, gate = snapshot.composition_root.context.require(ServiceKey("fixture.command_events"))
        if not started:
            entered, gate = asyncio.Event(), asyncio.Event()
            async def before_handler(*args, **kwargs):
                entered.set()
                await gate.wait()
                raise AssertionError("abandoned handler must not run")
            monkeypatch.setattr(CommandRegistry, "execute", before_handler)
        first = asyncio.create_task(run(host, input_id="first"))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
                root = snapshot.composition_root.context
                conversation = root.require(CONVERSATION)("probe:room")
                task = await conversation.start(root.require(CONVERSATION_COMMANDS))
                reader = log.reader("probe:room")
                await conversation.control("abandon", Control("abandon", reader.get("first").seq),
                                           expected_head=reader.head(source="conversation"), handle=task.handle)
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(first, 2)
            assert await run(host, input_id="new", text="new question") is None
            assert not any(isinstance(m.body, Output) for m in log.reader("probe:room").snapshot())
            async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
                assert len(snapshot.composition_root.context.require(ServiceKey("fixture.command_calls"))) == int(started)
            assert (tmp_path / "receipt.json").exists() == started
            assert len(log.owner("plugin:conversation").list()) == 1
        finally:
            gate.set()
            await asyncio.gather(first, return_exceptions=True)


@pytest.mark.asyncio
async def test_multiple_pending_commands_publish_in_input_order_not_receipt_key_order(tmp_path, monkeypatch):
    command_plugin(tmp_path, read_only=True)
    async with runtime(tmp_path) as (log, host, *_):
        append = MessageWriter.append
        def fail(writer, message_id, body, **kwargs):
            if isinstance(body, Output):
                raise OSError("leave immutable command intent pending")
            return append(writer, message_id, body, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(MessageWriter, "append", fail)
            for identity in ("z_first", "a_second"):
                with pytest.raises(OSError, match="intent pending"):
                    await run(host, input_id=identity)
        intents = log.owner("plugin:conversation").list()
        assert [row.value["input_id"] for _, row in intents] == ["a_second", "z_first"]
        result = await run(host, resume="a_second")
        outputs = [m for m in log.reader("probe:room").snapshot() if isinstance(m.body, Output)]
        assert [m.body.parts[0].value["input_id"] for m in outputs] == ["z_first", "a_second"]
        assert [m.body.finish for m in outputs] == ["continue", "complete"]
        assert result == outputs[-1]
