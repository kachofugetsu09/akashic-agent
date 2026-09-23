"""Read the recursive contract from the current owners in isolated workspaces."""

from __future__ import annotations

import asyncio
import ast
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

from akashic_sdk import AsyncAkashic, RemoteError
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.tasks import Tasks
from agent.plugins.install import install_git_plugin
from agent.plugins.manager import PluginManager
from agent.plugins.selection import PluginSelection
from bus.event_bus import EventBus
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from plugins.content.plugin import check_text
from plugins.delivery.records import DeliveryRecords
from plugins.message_push.tool import message_id
from plugins.tools.execution import ToolExecution
from plugins.tools.plugin import ALL_TOOLS, TOOLS, open_tool
from plugins.turn_projection.plugin import TurnProjection
from session.artifact_store import ArtifactStore
from session.log import MessageLog, SessionAttributes
from session.message import ContentPart, Input, Output, ToolResult
from tests.fixtures.plugin_workspace import initialize_plugin_workspace
from tests.test_delivery_bindings import sources
from tests.test_plugin_hot_reload import _v3_source, _write_checked_plugin
from tests.test_plugin_install import _commit
from tests.test_programmatic_control import endpoint
from tests.test_subagent_messages import CONTROLS, application


def _new_dir(parent: Path, name: str) -> Path:
    directory = parent / name
    directory.mkdir()
    return directory


def _save_plugin(path: Path, source: str) -> None:
    """Check a dynamic plugin before writing it to an isolated source tree."""

    tree = ast.parse(source, filename=str(path))
    compile(tree, str(path), "exec")
    path.write_text(source, encoding="utf-8")


async def read_selection(parent: Path) -> dict[str, object]:
    """Observe failed compilation and valid local replacement on one Root."""

    path = _new_dir(parent, "selection")
    plugin = _write_checked_plugin(path / "plugins", "selection", _v3_source("selection", version="a"))
    workspace = path / "workspace"
    initialize_plugin_workspace(workspace)
    bus = EventBus()
    manager = PluginManager([path / "plugins"], event_bus=bus, workspace=workspace, installed_cache_root=path / "home/cache")
    try:
        # 1. Read the committed input and its running owner.
        await manager.load_all()
        root = manager.live_root
        old = manager.generation("selection")
        before = PluginSelection(workspace).read()
        assert root is not None and old is not None and old.fiber is not None
        (plugin / "plugin.py").write_text("def invalid(:\n", encoding="utf-8")
        # 2. Compile failure leaves both owners untouched; valid source replaces only the Fiber.
        await manager.reconcile_changed()
        after_compile = PluginSelection(workspace).read()
        old_state = old.fiber.state.value
        _save_plugin(plugin / "plugin.py", _v3_source("selection", version="b"))
        await manager.reconcile_changed()
        current = manager.generation("selection")
        assert current is not None and current.fiber is not None
        return {
            "before_ref": before,
            "after_compile_ref": after_compile,
            "committed_ref": PluginSelection(workspace).read(),
            "old_fiber_after_compile": old_state,
            "root_before": root.generation_id,
            "root_after": manager.live_root.generation_id if manager.live_root is root else None,
            "new_fiber": current.fiber.state.value,
            "old_scope_closed": old.scope.closed,
        }
    finally:
        await manager.terminate_all()
        await bus.aclose()


async def read_failed_selection(parent: Path, monkeypatch) -> dict[str, object]:
    """Read a selected import failure, retained cleanup owner, and explicit retry."""

    path = _new_dir(parent, "failed-selection")
    source = _new_dir(path, "source")
    entry = source / "plugin.py"
    _save_plugin(entry, _v3_source("failed_target", version="a"))
    _commit(source)
    workspace = path / "workspace"
    initialize_plugin_workspace(workspace)
    home = path / "home"
    install_git_plugin(workspace=workspace, source=str(source), marketplace="lab", plugins_home=home)
    bus = EventBus()
    manager = PluginManager([], event_bus=bus, workspace=workspace, installed_cache_root=home / "cache")
    cleanup_fail = True
    original_load = manager._load_live_generation
    failed = None
    operation = None

    async def observe_load(generation):
        nonlocal failed
        if generation.plugin_id == "failed_target@lab" and generation.archive_ref != old.archive_ref and failed is None:
            failed = generation

            async def close():
                if cleanup_fail:
                    raise OSError("fixture cleanup blocked")

            generation.scope.defer("fixture-cleanup", close)
        await original_load(generation)

    try:
        # 1. Start A from a real installed Git artifact.
        await manager.load_all()
        root = manager.live_root
        old = manager.generation("failed_target@lab")
        previous_ref = PluginSelection(workspace).read()
        assert root is not None and old is not None
        _save_plugin(entry,
            "import os\n"
            + _v3_source("failed_target", version="b", exports=(
                "if os.environ.get('P0_IMPORT_FAIL') == 'yes':\n"
                "    raise ImportError('fixture B import blocked')\n"
            )),
        )
        subprocess.run(["git", "add", "."], cwd=source, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "B"], cwd=source, check=True, capture_output=True)
        monkeypatch.setenv("P0_IMPORT_FAIL", "yes")
        manager._load_live_generation = observe_load
        # 2. Commit selected B, then fail its import and first cleanup attempt.
        accepted = await manager.install(source=str(source), marketplace="lab", ref_name="", sparse_paths=[], update_id="p0-b")
        assert accepted.selection == "selected"
        operation = manager._operation
        assert operation is not None
        try:
            await operation.task
        except ImportError as error:
            assert str(error) == "fixture B import blocked"
        assert failed is not None
        status = manager.read_update("p0-b")
        selected_ref = PluginSelection(workspace).read()
        during = {
            "previous_ref": previous_ref,
            "selected_ref": selected_ref,
            "failed_state": status.state,
            "failed_fiber": None if failed.fiber is None else failed.fiber.state.value,
            "scope_retained": not failed.scope.closed and manager.generation("failed_target@lab") is failed,
            "old_a_closed": old.scope.closed,
            "recovered_before_retry": status.state == "active" or manager.generation("failed_target@lab") is old,
            "root_before": root.generation_id,
        }
        cleanup_fail = False
        monkeypatch.setenv("P0_IMPORT_FAIL", "no")
        # 3. The explicit retry closes B's retained Scope and mounts a fresh generation.
        await manager.retry_runtime_recovery("failed_target@lab")
        fresh = manager.generation("failed_target@lab")
        assert fresh is not None and fresh.fiber is not None
        return {
            **during,
            "retry_state": fresh.fiber.state.value,
            "failed_scope_closed_after_retry": failed.scope.closed,
            "root_after": manager.live_root.generation_id if manager.live_root is root else None,
        }
    finally:
        monkeypatch.setenv("P0_IMPORT_FAIL", "no")
        if operation is not None and not operation.task.done():
            await operation.task
        await manager.terminate_all()
        await bus.aclose()


async def read_programmatic(parent: Path, monkeypatch) -> dict[str, object]:
    """Read public admission, concurrent Session progress, and final Messages."""

    path = _new_dir(parent, "programmatic")
    async with endpoint(path, monkeypatch) as (address, core):
        parent_id = "programmatic:p0-parent"
        child_id = "programmatic:p0-validation"
        false_id = "programmatic:p0-false"
        async with await AsyncAkashic.connect(address) as client:
            # 1. Admit immutable eligibility through the public RPC boundary.
            admitted = await client.request("programmatic/session/admit", {"session_id": parent_id})
            repeated = await client.request("programmatic/session/admit", {"session_id": parent_id})
            await client.request("programmatic/session/admit", {"session_id": false_id, "persist_memory": False})
            await client.request("programmatic/session/admit", {"session_id": child_id, "persist_memory": True})
            try:
                await client.request("programmatic/session/admit", {"session_id": parent_id, "persist_memory": True})
            except RemoteError:
                conflict_rejected = True
            else:
                conflict_rejected = False
            await client.request("programmatic/message/send", {"session_id": parent_id, "message_id": "parent-input", "text": "parent"})
            await client.request("programmatic/message/send", {"session_id": child_id, "message_id": "validation-input", "text": "validate"})
            # 2. Settle validation while the parent Input remains open.
            validation_writer = core.message_log.writer(child_id, author="assistant", source="programmatic", body_types=(Output,), content={"text": check_text})
            validation_writer.append("validation-output", Output((ContentPart("text", "ready"),), "complete"))
            validation = await client.request("programmatic/message/result", {"session_id": child_id, "input_id": "validation-input"})
            parent_open = await client.request("programmatic/message/result", {"session_id": parent_id, "input_id": "parent-input"})
            parent_writer = core.message_log.writer(parent_id, author="assistant", source="programmatic", body_types=(Output,), content={"text": check_text})
            parent_writer.append("parent-output", Output((ContentPart("text", "done"),), "complete"))
            parent_done = await client.request("programmatic/message/result", {"session_id": parent_id, "input_id": "parent-input"})
        log = core.message_log
        child_rows = log.reader(child_id).snapshot()
        return {
            "excluded_learning": log.reader(parent_id).attributes.learning,
            "false_learning": log.reader(false_id).attributes.learning,
            "eligible_learning": log.reader(child_id).attributes.learning,
            "default_retry_equal": admitted == repeated,
            "conflict_rejected": conflict_rejected,
            "parent_open_during_validation": parent_open["status"] == "open",
            "validation_terminal": validation["status"],
            "parent_terminal": parent_done["status"],
            "validation_bodies": tuple(type(row.body).__name__ for row in child_rows),
        }


async def read_child(parent: Path) -> dict[str, object]:
    """Read a live child work window, durable caller result, and domain file."""

    path = _new_dir(parent, "child")
    async with application(path, block=True) as (_host, log, execution, reply):
        call = asyncio.create_task(execution.execute_call(reply))
        control = CONTROLS[str(path)]
        try:
            # 1. Hold the child model at its Event while the parent still awaits a result.
            await asyncio.wait_for(control.entered.get(), 15)
            parent_rows = log.reader("test:parent").snapshot()
            parent_turns = TurnProjection().project(parent_rows, "fixture")
            parent_open = (
                bool(parent_turns) and parent_turns[-1].status == "open"
                and sum(isinstance(row.body, ToolResult) for row in parent_rows) == 0
            )
            control.release.set()
            result = await asyncio.wait_for(call, 15)
            # 2. Read the child history, the file effect, and the caller's saved ToolResult.
            children = [key for key in log.catalog().snapshot_heads() if key.startswith("subagent:")]
            assert len(children) == 1
            child_rows = log.reader(children[0]).snapshot()
            request = next(
                part for part in child_rows[0].body.parts
                if isinstance(part, ContentPart) and part.kind == "subagent.request"
            )
            assert isinstance(request.value, Mapping)
            job_id = request.value["job_id"]
            domain = path / "workspace" / "subagent-runs" / job_id / "answer.txt"
            writer = log.writer("test:parent", author="assistant", source="fixture", body_types=(Output,), content={"text": check_text})
            writer.append("parent-terminal", Output((ContentPart("text", "done"),), "complete"))
            final_rows = log.reader("test:parent").snapshot()
            return {
                "parent_open_during_child": parent_open,
                "call_outcome": result.outcome,
                "child_bodies": tuple(type(row.body).__name__ for row in child_rows),
                "domain_file": domain.read_text(),
                "caller_tool_results": sum(isinstance(row.body, ToolResult) for row in final_rows),
                "parent_terminal": final_rows[-1].body.finish if isinstance(final_rows[-1].body, Output) else None,
            }
        finally:
            control.release.set()
            if not call.done():
                await call


async def read_push(parent: Path) -> dict[str, object]:
    """Read the caller tool, target Message, and Delivery after one real send."""

    path = _new_dir(parent, "push")
    source = path / "plugins"
    sources(source)
    for name in ("content", "tools", "message_push"):
        shutil.copytree(Path(__file__).parents[2] / "plugins" / name, source / name)
    workspace = path / "workspace"
    workspace.mkdir()
    log = MessageLog(workspace / "sessions.db")
    store = ArtifactStore(workspace / "sessions.db")
    initialize_plugin_workspace(workspace)
    bus = EventBus()
    artifacts = ChannelAttachmentArtifactStore(workspace=workspace, metadata_store=store)
    manager = PluginManager([source], event_bus=bus, workspace=workspace, installed_cache_root=path / "home", message_log=log, channel_attachment_store=artifacts)
    tasks = Tasks()
    try:
        # 1. Mount the real MessagePush, Tools, and Delivery providers.
        await manager.load_all()
        await manager.start_runtime()
        root = manager.live_root
        sender = manager.generation("test_sender")
        assert root is not None and sender is not None and sender.fiber is not None
        bindings = root.service_value(BINDINGS)
        tools = root.service_value(TOOLS)
        all_tools = root.service_value(ALL_TOOLS)
        assert bindings is not None and tools is not None and all_tools is not None
        binding = await tools.bind_scoped(all_tools().select("message_push"), bindings)
        log.ensure_session("test:room", SessionAttributes())
        writer = log.writer("test:room", author="user", source="test", body_types=(Input,), content={"text": check_text})
        writer.append("active-input", Input((ContentPart("text", "in progress"),)))
        before = log.reader("test:room").snapshot()
        before_turn = TurnProjection().project(before, "test")

        async def authorize(_binding, _arguments):
            return {"approved": True}

        execution = ToolExecution(log.owner("plugin:tools"), tasks, lambda key: open_tool(bindings, key), authorize, task_key="effects")
        # 2. Send once, repeat the same caller key, then read both durable owners.
        first = await execution.execute("push-once", binding, {"target_channel": "test", "target_chat_id": "room", "message": "notice"})
        repeated = await execution.execute("push-once", binding, {"target_channel": "test", "target_chat_id": "room", "message": "notice"})
        after = log.reader("test:room").snapshot()
        after_turn = TurnProjection().project(after, "test")
        identity = message_id("program:push-once")
        delivery = DeliveryRecords(log.owner("plugin:delivery"), "message_push").read(identity, "test")[1]
        sent = next(workspace.rglob("sent.jsonl")).read_text().splitlines()
        return {
            "tool_outcome": first.outcome,
            "tool_phase": log.owner("plugin:tools").read("program:push-once").value["phase"],
            "delivery_phase": delivery.phase,
            "target_turn_open_across_push": (
                len(before) == 1 and isinstance(before[0].body, Input)
                and len(before_turn) == 1 and before_turn[0].status == "open"
                and len(after_turn) == 1 and after_turn[0].status == "open"
                and after_turn[0].message_ids == before_turn[0].message_ids
            ),
            "target_input_unchanged": after[0] == before[0],
            "new_target_bodies": tuple((type(row.body).__name__, row.source) for row in after[len(before):]),
            "push_message_id_match": len(after) == len(before) + 1 and after[-1].message_id == identity,
            "repeat_same_receipt": repeated == first,
            "send_count": len(sent),
        }
    finally:
        await tasks.close()
        await manager.terminate_all()
        await bus.aclose()
        log.close()
        store.close()
