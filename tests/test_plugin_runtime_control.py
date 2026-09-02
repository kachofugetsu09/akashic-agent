from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import certifi
import pytest

from agent.control.context import mint_plugin_child_capability
from agent.control.models import TurnRequest
from agent.control.ports import ControlExecutionResult
from agent.control.runtime import ConversationRuntime
from agent.control.service import ControlService
from agent.plugins.artifacts import read_pointers, resolve_pointer
from agent.plugins.manager import PluginManager
from agent.plugins.install import PluginInstallResult, install_git_plugin
from agent.plugins.reload_journal import ReloadJournal
from agent.plugins.turn_rollout import TurnPluginRollout
from agent.tools.registry import ToolRegistry
from bootstrap.app import AppRuntime
from bus.event_bus import EventBus
from session.manager import SessionManager


@pytest.mark.asyncio
async def test_turn_owned_candidate_promotes_after_exact_child(
    tmp_path: Path,
) -> None:
    """Run the real parent-child rollout through snapshot, journal, and pointer owners."""

    # 1. Build one stable Root and one immutable candidate with Tool and Skill assets.
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    builtin = tmp_path / "builtin" / "baseline"
    _write_v3_plugin(builtin, name="baseline")
    source = tmp_path / "candidate"
    _write_tool_skill_plugin(source)
    _commit(source)
    event_bus = EventBus()
    sessions = SessionManager(workspace)
    manager = PluginManager(
        plugin_dirs=[builtin.parent],
        event_bus=event_bus,
        tool_registry=ToolRegistry(),
        session_manager=sessions,
        workspace=workspace,
        installed_cache_root=tmp_path / "plugins-home" / "cache",
    )
    await manager.load_all()
    stable = manager.current_snapshot
    assert stable is not None and stable.composition_root is not None

    async def uninstall(_plugin_id: str) -> dict[str, object]:
        raise AssertionError("install promotion must not call uninstall")

    rollout = TurnPluginRollout(manager, workspace=workspace, uninstall=uninstall)
    parent_release = asyncio.Event()
    installed = asyncio.Event()
    install_result: PluginInstallResult | None = None
    install_status: dict[str, object] | None = None
    seen_snapshots: dict[str, object] = {}

    async def execute(request: TurnRequest) -> ControlExecutionResult:
        nonlocal install_result, install_status
        selector = cast(Any, request.metadata.get("runtime", "stable"))
        lease = manager.snapshot_store.lease(selector=selector)
        async with lease as snapshot:
            assert snapshot.composition_root is not None
            seen_snapshots[request.input] = snapshot
            if request.input == "install candidate":
                turn_id = str(request.metadata["turnId"])
                install_result, install_status = await rollout.install(
                    turn_id,
                    source=str(source),
                    marketplace="lab",
                    ref_name="",
                    sparse_paths=[],
                )
                installed.set()
                await parent_release.wait()
            return ControlExecutionResult(response="ok")

    runtime = ConversationRuntime(
        sessions.control_store,
        execute,
        turn_terminal=rollout.turn_terminal,
    )
    service = ControlService(
        runtime,
        sessions,
        workspace,
        plugin_child_binding=rollout.child_binding,
        plugin_turn_barrier=rollout.wait_for_turn_boundary,
    )
    parent_handle = None
    child_handle = None
    try:
        # 2. The real parent Turn owns install while retaining its stable lease.
        parent_thread = service.start_thread({})
        parent_thread_id = str(parent_thread["id"])
        parent_handle = await service.start_turn(
            parent_thread_id,
            "install candidate",
            {},
        )
        await asyncio.wait_for(installed.wait(), timeout=10)
        assert install_result is not None and install_status is not None
        candidate = manager.latest_snapshot
        assert candidate is not None and candidate is not stable
        assert seen_snapshots["install candidate"] is stable

        # 3. Reserve and consume the registered one-shot capability through Control.
        capability = mint_plugin_child_capability(parent_handle.id)
        assert capability
        child_thread = service.start_thread(
            {},
            plugin_rollout_capability=capability,
        )
        child_thread_id = str(child_thread["id"])
        child_handle = await service.start_turn(
            child_thread_id,
            "check candidate",
            {},
            attached=True,
        )
        child_result = await child_handle.result()
        assert child_result.status.value == "completed"
        await runtime.wait_thread_available(child_thread_id)
        assert seen_snapshots["check candidate"] is candidate

        # 4. No revert plus a normal parent terminal is the sole commit grant.
        parent_release.set()
        parent_result = await parent_handle.result()
        assert parent_result.status.value == "completed"
        await runtime.wait_thread_available(parent_thread_id)
        await rollout.wait_for_turn_boundary()
        await manager.snapshot_store.retry_drains()

        assert manager.current_snapshot is candidate
        assert manager.latest_snapshot is candidate
        assert manager.ready_candidate is None
        assert "已经成功提交" in rollout.consume_fact()

        # 5. Verify durable recovery facts, not only in-memory publication state.
        tx_id = str(install_status["candidate_reload_tx_id"])
        journal = ReloadJournal(workspace)
        record = journal.get(tx_id)
        assert record.phase == "complete"
        events = journal.events(tx_id)
        details = [event.details for event in events]
        assert any(item.get("event") == "turn_operation_registered" for item in details)
        assert any(
            item.get("event") == "candidate_child_terminal"
            and item.get("identity_match") is True
            and item.get("candidate_checked") is True
            for item in details
        )
        assert "promoting" in [event.phase for event in events]
        assert "committed" in [event.phase for event in events]

        plugin_base = install_result.installed_path.parents[1]
        pointers = read_pointers(plugin_base)
        assert pointers is not None and pointers.stable == pointers.latest
        active = manager.generation("candidate@lab")
        assert active is not None
        assert resolve_pointer(plugin_base, pointers.stable) == active.plugin_dir
    finally:
        parent_release.set()
        for handle in (child_handle, parent_handle):
            if handle is not None and handle.record()["status"] == "in_progress":
                _ = await handle.result()
        await service.shutdown()
        await rollout.shutdown()
        await runtime.shutdown()
        if manager.ready_candidate is not None:
            await manager.drop_candidate("candidate@lab")
        await manager.terminate_all()
        sessions.close()
        await event_bus.aclose()


@pytest.mark.asyncio
async def test_runtime_install_waits_until_latest_is_leasable(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin" / "baseline"
    _write_v3_plugin(builtin, name="baseline")
    source = tmp_path / "candidate"
    _write_v3_plugin(source, name="candidate", static_manifest=True)
    _commit(source)
    bus = EventBus()
    manager = PluginManager(
        plugin_dirs=[builtin.parent],
        event_bus=bus,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "plugins-home" / "cache",
    )
    await manager.load_all()
    stable = manager.current_snapshot
    assert stable is not None
    app = object.__new__(AppRuntime)
    app.workspace = tmp_path / "workspace"
    app.core = SimpleNamespace(plugin_manager=manager)

    result = await app._install_plugin(str(source), "lab", "", [])

    assert result["pluginId"] == "candidate@lab"
    assert result["publicationState"] == "latest_ready"
    candidate = result["candidate"]
    assert isinstance(candidate, dict)
    assert (
        candidate["candidateRuntimeRevision"]
        == manager.candidate_status()["candidate_source_revision"]
    )
    assert candidate["candidateReloadTransactionId"]
    assert candidate["candidateError"] == ""
    assert manager.current_snapshot is stable
    assert "candidate@lab" not in stable.generations
    assert manager.latest_snapshot is not None
    assert "candidate@lab" in manager.latest_snapshot.generations
    latest_lease = manager.snapshot_store.lease(selector="latest")
    await latest_lease.release()

    blocked_source = tmp_path / "blocked"
    _write_v3_plugin(blocked_source, name="blocked", static_manifest=True)
    _commit(blocked_source)
    with pytest.raises(
        RuntimeError,
        match="已有插件候选等待处理: plugin=candidate@lab phase=latest_ready",
    ):
        await app._install_plugin(str(blocked_source), "lab", "", [])
    assert not (manager.installed_plugins_home / "cache" / "lab" / "blocked").exists()

    promoted = await app._promote_plugin("candidate@lab")

    assert promoted["publication_state"] == "promoted"
    assert manager.current_snapshot is manager.latest_snapshot
    await manager.terminate_all()
    await bus.aclose()


@pytest.mark.asyncio
async def test_installed_mcp_update_keeps_old_artifact_until_lease_drains(
    tmp_path: Path,
) -> None:
    source, manager, app, bus, old_artifact = await _start_runtime_mcp(tmp_path)
    old_lease = manager.snapshot_store.lease()
    latest_lease = None
    promoted_lease = None
    plugin_id = "runtime_mcp@lab"
    old_generation = old_lease.snapshot.generations[plugin_id]
    old_runtime = _composition_runtime(manager, old_generation)
    old_server = _mcp_server(old_runtime)

    try:
        # 1. 同版本提交新 revision，并等待真实 latest MCP 可租用。
        _write_runtime_mcp_source(source, runtime_version="v2")
        _commit_all(source, "runtime-v2")
        updated = await app._install_plugin(str(source), "lab", "", [])
        new_artifact = Path(str(updated["installedPath"]))
        assert updated["version"] == "1.0.0"
        assert new_artifact != old_artifact
        assert old_artifact.is_dir() and new_artifact.is_dir()

        latest_lease = manager.snapshot_store.lease(selector="latest")
        latest_generation = latest_lease.snapshot.generations[plugin_id]
        latest_runtime = _composition_runtime(manager, latest_generation)
        latest_server = _mcp_server(latest_runtime)
        assert latest_runtime.generation_id != old_runtime.generation_id

        # 2. 更新后旧 MCP 延迟读取自己的 CA bundle，新旧调用各自保持代际身份。
        old_probe = await _call_runtime_probe(old_server)
        latest_probe = await _call_runtime_probe(latest_server)
        assert {
            key: old_probe[key]
            for key in (
                "artifact",
                "ca_bundle",
                "ca_certificates",
                "data_dir",
                "runtime_version",
                "workspace",
            )
        } == {
            "artifact": str(old_artifact),
            "ca_bundle": str(_runtime_ca_bundle(old_artifact)),
            "ca_certificates": old_probe["ca_certificates"],
            "data_dir": str(
                tmp_path / "workspace" / "plugin-data" / "runtime_mcp-lab"
            ),
            "runtime_version": "v1",
            "workspace": str(tmp_path / "workspace"),
        }
        assert int(old_probe["ca_certificates"]) > 0
        assert isinstance(old_probe["pid"], int)
        assert latest_probe["artifact"] == str(new_artifact)
        assert "runtime/plugin-validation" in str(latest_probe["data_dir"])
        assert latest_probe["pid"] != old_probe["pid"]
        assert latest_probe["runtime_version"] == "v2"

        # 3. 候选 lease 排空后，promotion 必须等待旧 stable lease 排空。
        candidate_snapshot = latest_lease.snapshot
        await latest_lease.release()
        latest_lease = None
        promote_task = asyncio.create_task(app._promote_plugin(plugin_id))
        await asyncio.sleep(0)
        assert not promote_task.done()
        await old_lease.release()
        old_lease = None
        promoted = await asyncio.wait_for(promote_task, timeout=30)
        assert promoted["publication_state"] == "promoted"
        promoted_lease = manager.snapshot_store.lease()
        assert promoted_lease.snapshot is candidate_snapshot
        promoted_generation = promoted_lease.snapshot.generations[plugin_id]
        promoted_runtime = _composition_runtime(manager, promoted_generation)
        assert promoted_runtime is not latest_runtime
        assert promoted_runtime.generation_id == latest_runtime.generation_id
        assert promoted_runtime.mode == "formal"

        await promoted_lease.release()
        promoted_lease = None
        await manager.snapshot_store.retry_drains()
        assert manager._composition_generation_host.get(old_generation.generation_id) is None
        assert old_artifact.is_dir()

        # 4. 已排空 artifact 仍保留，只有显式卸载才删除 cache。
        plugin_base = old_artifact.parents[1]
        data_path = Path(str(updated["dataPath"]))
        _ = await app._uninstall_plugin(plugin_id)
        assert not plugin_base.exists()
        assert data_path.is_dir()
    finally:
        for lease in (latest_lease, promoted_lease, old_lease):
            if lease is not None and lease.active:
                await lease.release()
        if manager.ready_candidate is not None:
            await manager.drop_candidate(plugin_id)
        await manager.terminate_all()
        await bus.aclose()


@pytest.mark.asyncio
async def test_mcp_candidate_uses_isolated_data_and_exact_read_only_surface(
    tmp_path: Path,
) -> None:
    source, manager, app, bus, _old_artifact = await _start_runtime_mcp(tmp_path)
    plugin_id = "runtime_mcp@lab"
    production_data = tmp_path / "workspace" / "plugin-data" / "runtime_mcp-lab"
    marker = production_data / "production-marker.json"
    marker.write_text('{"owner":"production"}\n', encoding="utf-8")
    production_before = marker.read_bytes()
    production_digest_before = _directory_digest(production_data)

    try:
        _write_runtime_mcp_source(source, runtime_version="v2")
        _commit_all(source, "runtime-v2-isolated")
        await app._install_plugin(str(source), "lab", "", [])

        candidate = manager.ready_candidate
        assert candidate is not None
        validation_root = (
            tmp_path
            / "workspace"
            / "runtime"
            / "plugin-validation"
            / candidate.generation_id
        )
        validation_workspace = validation_root / "workspace"
        validation_data = (
            validation_workspace / "plugin-data" / "runtime_mcp-lab"
        )
        assert candidate.data_dir == validation_data
        assert candidate.production_data_dir == production_data
        assert (validation_data / marker.name).read_bytes() == production_before
        assert not (tmp_path / "workspace" / "candidate-mcp-started.json").exists()
        assert marker.read_bytes() == production_before
        assert _directory_digest(production_data) == production_digest_before

        candidate_runtime = _composition_runtime(manager, candidate)
        candidate_server = _mcp_server(candidate_runtime)
        probe = await _call_runtime_probe(
            candidate_server,
        )
        runtime_workspace = Path(str(probe["workspace"]))
        runtime_data = Path(str(probe["data_dir"]))
        assert runtime_workspace.name == "workspace"
        assert runtime_workspace.is_relative_to(validation_root / "composition")
        assert runtime_data == (
            runtime_workspace / "plugin-data" / "runtime_mcp-lab"
        )
        assert (runtime_workspace / "candidate-mcp-started.json").is_file()
        assert (runtime_data / "candidate-mcp-started.json").is_file()
        assert (runtime_data / marker.name).read_bytes() == production_before
        assert marker.read_bytes() == production_before
        assert _directory_digest(production_data) == production_digest_before

        registry = manager.latest_snapshot.tool_registry
        assert registry is not None
        assert registry.get_source_tool_names(
            "mcp", "runtime_probe", risk="read-only"
        ) == {"mcp_runtime_probe__probe"}
        assert registry.get_non_read_only_source_tool_names(
            "mcp", "runtime_probe"
        ) == set()
        await app._promote_plugin(plugin_id)

        active = manager.generation(plugin_id)
        assert active is not None and active.data_dir == production_data
        assert marker.read_bytes() == production_before
        assert _directory_digest(production_data) == production_digest_before
        assert not validation_root.exists()
    finally:
        if manager.ready_candidate is not None:
            await manager.drop_candidate(plugin_id)
        await manager.terminate_all()
        await bus.aclose()


@pytest.mark.asyncio
async def test_mcp_hot_reload_oracle_rejects_deleted_old_ca_bundle(
    tmp_path: Path,
) -> None:
    source, manager, app, bus, old_artifact = await _start_runtime_mcp(tmp_path)
    old_lease = manager.snapshot_store.lease()
    latest_lease = None
    plugin_id = "runtime_mcp@lab"
    old_generation = old_lease.snapshot.generations[plugin_id]
    old_runtime = _composition_runtime(manager, old_generation)
    old_server = _mcp_server(old_runtime)

    try:
        # 1. 建立新 latest 后模拟旧安装器提前删除 CA bundle。
        _write_runtime_mcp_source(source, runtime_version="v2")
        _commit_all(source, "runtime-v2")
        _ = await app._install_plugin(str(source), "lab", "", [])
        latest_lease = manager.snapshot_store.lease(selector="latest")
        _runtime_ca_bundle(old_artifact).unlink()

        # 2. 旧 MCP 在调用时才读取路径，oracle 必须命中原事故而不是静默切新代。
        error_result = await old_server.route().call("probe", {})
        assert error_result.tool_error
        assert "cacert.pem" in error_result.output or "No such file" in error_result.output
        assert old_lease.snapshot is manager.current_snapshot
        latest_generation = latest_lease.snapshot.generations[plugin_id]
        latest_runtime = _composition_runtime(manager, latest_generation)
        latest_probe = await _call_runtime_probe(_mcp_server(latest_runtime))
        assert latest_probe["runtime_version"] == "v2"
    finally:
        if latest_lease is not None and latest_lease.active:
            await latest_lease.release()
        if manager.ready_candidate is not None:
            await manager.drop_candidate(plugin_id)
        if old_lease.active:
            await old_lease.release()
        await manager.terminate_all()
        await bus.aclose()


@pytest.mark.asyncio
async def test_runtime_install_and_watcher_share_candidate_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builtin = tmp_path / "builtin" / "baseline"
    _write_v3_plugin(builtin, name="baseline")
    sources: dict[str, Path] = {}
    for name in ("alpha", "beta"):
        source = tmp_path / name
        _write_v3_plugin(source, name=name, static_manifest=True)
        _commit(source)
        sources[name] = source
    bus = EventBus()
    manager = PluginManager(
        plugin_dirs=[builtin.parent],
        event_bus=bus,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "plugins-home" / "cache",
    )
    await manager.load_all()
    _ = await asyncio.to_thread(
        install_git_plugin,
        workspace=tmp_path / "workspace",
        source=str(sources["alpha"]),
        marketplace="lab",
        plugins_home=manager.installed_plugins_home,
        stage_candidate=True,
    )
    app = object.__new__(AppRuntime)
    app.workspace = tmp_path / "workspace"
    app.core = SimpleNamespace(plugin_manager=manager)

    install, watcher = await asyncio.gather(
        app._install_plugin(str(sources["beta"]), "lab", "", []),
        manager.reconcile_changed(),
        return_exceptions=True,
    )

    assert isinstance(install, RuntimeError)
    assert "plugin=alpha@lab phase=latest_ready" in str(install)
    assert not (manager.installed_plugins_home / "cache" / "lab" / "beta").exists()
    assert not isinstance(watcher, BaseException)
    assert manager.candidate_status()["candidate_plugin_id"] == "alpha@lab"
    await manager.switch_ready("alpha@lab")
    alpha_entrypoint = sources["alpha"] / "plugin.py"
    alpha_entrypoint.write_text(
        alpha_entrypoint.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    _commit(sources["alpha"])

    entered = threading.Event()
    release = threading.Event()

    def blocking_install(**kwargs: Any) -> PluginInstallResult:
        entered.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release plugin install")
        return install_git_plugin(**kwargs)

    monkeypatch.setattr("agent.plugins.manager.install_git_plugin", blocking_install)
    cancelled_install = asyncio.create_task(
        app._install_plugin(str(sources["alpha"]), "lab", "", [])
    )
    assert await asyncio.to_thread(entered.wait, 5)
    cancelled_install.cancel()
    waiting_watcher = asyncio.create_task(manager.reconcile_changed())
    await asyncio.sleep(0)
    assert not waiting_watcher.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_install
    _ = await waiting_watcher
    cancelled_status = manager.candidate_status()
    assert cancelled_status["candidate_plugin_id"] == "alpha@lab"
    assert cancelled_status["candidate_state"] == "aborted"
    assert (
        cancelled_status["latest_snapshot_id"]
        == cancelled_status["stable_snapshot_id"]
    )
    await manager.terminate_all()
    await bus.aclose()

async def _start_runtime_mcp(
    tmp_path: Path,
) -> tuple[Path, PluginManager, AppRuntime, EventBus, Path]:
    """安装并晋升一个带真实 MCP 子进程的初始 stable generation。"""

    # 1. 从独立 Git source 安装首个不可变 artifact。
    source = tmp_path / "runtime-mcp-source"
    _write_runtime_mcp_source(source, runtime_version="v1")
    _commit_all(source, "runtime-v1")
    bus = EventBus()
    manager = PluginManager(
        plugin_dirs=[],
        event_bus=bus,
        tool_registry=ToolRegistry(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "plugins-home" / "cache",
    )
    await manager.load_all()
    app = object.__new__(AppRuntime)
    app.workspace = tmp_path / "workspace"
    app.core = SimpleNamespace(plugin_manager=manager)

    # 2. 首次安装先成为 latest，显式 promote 后才提供 stable lease。
    installed = await app._install_plugin(str(source), "lab", "", [])
    assert installed["publicationState"] == "latest_ready"
    _ = await app._promote_plugin("runtime_mcp@lab")
    return source, manager, app, bus, Path(str(installed["installedPath"]))


def _write_runtime_mcp_source(source: Path, *, runtime_version: str) -> None:
    """写入在每次工具调用时解析 artifact 内 CA bundle 的 MCP 插件。"""

    # 1. 插件版本保持不变，用 server 内容变化制造同版本新 revision。
    source.mkdir(parents=True, exist_ok=True)
    _ = (source / "plugin.py").write_text(
        "from agent.plugin_composition import MCP_SERVERS, McpServerDefinition\n"
        "api_version = 3\n"
        "name = 'runtime_mcp'\n"
        "version = '1.0.0'\n"
        "inject = (MCP_SERVERS,)\n"
        "skill_roots = ('skills',)\n"
        "async def apply(ctx, config):\n"
        "    await ctx.require(MCP_SERVERS).register(\n"
        "        ctx, McpServerDefinition(\n"
        "            name='runtime_probe', command=('python', 'mcp/server.py'),\n"
        "            required_tools=('probe',),\n"
        "            candidate_read_only_tools=('probe',),\n"
        "        ),\n"
        "    )\n",
        encoding="utf-8",
    )
    _ = (source / "mcp").mkdir(parents=True, exist_ok=True)
    _ = (source / "mcp" / "requirements.txt").write_text(
        "certifi\n",
        encoding="utf-8",
    )
    skill_dir = source / "skills" / "runtime-probe"
    skill_dir.mkdir(parents=True, exist_ok=True)
    _ = (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: runtime-probe\n"
        "description: Validate the runtime MCP candidate.\n"
        "---\n\n"
        "# Runtime probe\n",
        encoding="utf-8",
    )

    # 2. server 不缓存文件内容，确保更新后的调用会触发旧绝对路径读取。
    _ = (source / "mcp" / "server.py").write_text(
        "import json, os, ssl, sys\n"
        "from pathlib import Path\n"
        f"RUNTIME_VERSION = {runtime_version!r}\n"
        "ARTIFACT = Path(__file__).resolve().parent.parent\n"
        "DATA_DIR = Path(os.environ['AKA_PLUGIN_DATA_DIR'])\n"
        "WORKSPACE = Path(os.environ['AKASHIC_WORKSPACE'])\n"
        "if 'plugin-validation' in WORKSPACE.parts:\n"
        "    (WORKSPACE / 'candidate-mcp-started.json').write_text('started\\n', encoding='utf-8')\n"
        "    (DATA_DIR / 'candidate-mcp-started.json').write_text('started\\n', encoding='utf-8')\n"
        "TOOLS = ["
        "{'name': 'probe', 'description': 'probe runtime', "
        "'inputSchema': {'type': 'object', 'properties': {}}}, "
        "{'name': 'poll_feed', 'description': 'poll and persist feed cursor', "
        "'inputSchema': {'type': 'object', 'properties': {}}}]\n"
        "for line in sys.stdin:\n"
        "    message = json.loads(line)\n"
        "    if 'id' not in message:\n"
        "        continue\n"
        "    try:\n"
        "        method = message.get('method')\n"
        "        result = {}\n"
        "        if method == 'initialize':\n"
        "            result = {'protocolVersion': '2025-11-25'}\n"
        "        elif method == 'tools/list':\n"
        "            result = {'tools': TOOLS}\n"
        "        elif method == 'tools/call':\n"
        "            py_version = f'python{sys.version_info.major}.{sys.version_info.minor}'\n"
        "            ca_bundle = ARTIFACT / 'mcp' / '.venv' / 'lib' / py_version / "
        "'site-packages' / 'certifi' / 'cacert.pem'\n"
        "            context = ssl.create_default_context(cafile=str(ca_bundle))\n"
        "            probe = {'artifact': str(ARTIFACT), 'ca_bundle': str(ca_bundle), "
        "'data_dir': os.environ.get('AKA_PLUGIN_DATA_DIR', ''), "
        "'ca_certificates': context.cert_store_stats()['x509_ca'], 'pid': os.getpid(), "
        "'runtime_version': RUNTIME_VERSION, "
        "'workspace': os.environ.get('AKASHIC_WORKSPACE', '')}\n"
        "            result = {'content': [{'type': 'text', 'text': json.dumps(probe, sort_keys=True)}]}\n"
        "        response = {'jsonrpc': '2.0', 'id': message['id'], 'result': result}\n"
        "    except Exception as error:\n"
        "        response = {'jsonrpc': '2.0', 'id': message['id'], "
        "'error': {'code': -32000, 'message': f'{type(error).__name__}: {error}'}}\n"
        "    print(json.dumps(response), flush=True)\n",
        encoding="utf-8",
    )
    ca_bundle = _runtime_ca_bundle(source)
    ca_bundle.parent.mkdir(parents=True, exist_ok=True)
    _ = ca_bundle.write_bytes(Path(certifi.where()).read_bytes())
    _ = (source / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        "name = \"runtime_mcp\"\n"
        "version = \"1.0.0\"\n"
        "api_version = 3\n"
        "entrypoint = \"plugin.py\"\n\n"
        "[[python]]\n"
        "requirements = \"mcp/requirements.txt\"\n\n"
        "[[mcp]]\n"
        "name = \"runtime_probe\"\n"
        "command = [\"python\", \"mcp/server.py\"]\n"
        "required_tools = [\"probe\"]\n"
        "candidate_read_only_tools = [\"probe\"]\n",
        encoding="utf-8",
    )


def _runtime_ca_bundle(root: Path) -> Path:
    """返回与 PATH 实际启动的 MCP Python 版本一致的 certifi 路径。"""

    python_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return (
        root
        / "mcp"
        / ".venv"
        / "lib"
        / python_dir
        / "site-packages"
        / "certifi"
        / "cacert.pem"
    )


def _directory_digest(root: Path) -> str:
    """按相对路径和内容计算目录内全部普通文件摘要。"""

    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


async def _call_runtime_probe(server: Any) -> dict[str, Any]:
    """调用 v3 MCP route，并严格解析结构化代际证据。"""

    result = await server.route().call("probe", {})
    if result.tool_error:
        raise AssertionError(f"MCP probe 调用失败: {result.output}")
    raw = result.output
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise AssertionError(f"MCP probe 返回值不是对象: {parsed!r}")
    return parsed


def _composition_runtime(manager: PluginManager, generation: Any) -> Any:
    """Return one exact v3 composition runtime for a generation."""

    runtime = manager._composition_generation_host.get(generation.generation_id)
    assert runtime is not None and runtime.mcp is not None
    return runtime


def _mcp_server(runtime: Any) -> Any:
    """Return the exact runtime MCP server view under test."""

    assert runtime.mcp is not None
    return runtime.mcp["runtime_probe"]


def _write_v3_plugin(
    plugin_dir: Path,
    *,
    name: str,
    version: str = "1.0.0",
    static_manifest: bool = False,
) -> None:
    """Write a minimal module-level v3 plugin fixture."""

    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.py").write_text(
        "api_version = 3\n"
        f"name = {name!r}\n"
        f"version = {version!r}\n\n"
        "async def apply(ctx, config):\n"
        "    return None\n",
        encoding="utf-8",
    )
    if static_manifest:
        (plugin_dir / "akashic.plugin.toml").write_text(
            "schema_version = 1\n"
            f"name = {json.dumps(name)}\n"
            f"version = {json.dumps(version)}\n"
            "api_version = 3\n"
            "entrypoint = \"plugin.py\"\n",
            encoding="utf-8",
        )


def _write_tool_skill_plugin(plugin_dir: Path) -> None:
    """Write one ordinary v3 candidate that contributes both Tool and Skill."""

    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.py").write_text(
        "from agent.plugin_composition import TOOL_CATALOG, PluginToolDefinition\n\n"
        "api_version = 3\n"
        "name = 'candidate'\n"
        "version = '1.0.0'\n"
        "inject = (TOOL_CATALOG,)\n"
        "skill_roots = ('skills',)\n\n"
        "async def candidate_probe(context, arguments):\n"
        "    del context, arguments\n"
        "    return 'candidate-ready'\n\n"
        "async def apply(ctx, config):\n"
        "    del config\n"
        "    await ctx.require(TOOL_CATALOG).register(ctx, PluginToolDefinition(\n"
        "        name='candidate_probe',\n"
        "        description='Check the candidate.',\n"
        "        parameters={\n"
        "            'type': 'object',\n"
        "            'properties': {},\n"
        "            'required': [],\n"
        "            'additionalProperties': False,\n"
        "        },\n"
        "        handler_export='candidate_probe',\n"
        "        risk='read-only',\n"
        "    ))\n",
        encoding="utf-8",
    )
    skill = plugin_dir / "skills" / "candidate-check"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\n"
        "name: candidate-check\n"
        "description: Check the installed candidate.\n"
        "---\n\n"
        "# Candidate check\n",
        encoding="utf-8",
    )
    (plugin_dir / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        'name = "candidate"\n'
        'version = "1.0.0"\n'
        "api_version = 3\n"
        'entrypoint = "plugin.py"\n',
        encoding="utf-8",
    )


def _commit_all(repo: Path, message: str) -> None:
    """提交测试插件的完整 source tree，并支持同仓库连续 revision。"""

    # 1. 首次提交时建立独立身份，后续只追加 commit。
    if not (repo / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=repo,
            check=True,
        )
    subprocess.run(["git", "add", "--force", "--all"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True)


def _commit(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "candidate"], cwd=repo, check=True)
