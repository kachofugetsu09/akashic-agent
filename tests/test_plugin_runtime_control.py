from __future__ import annotations

import asyncio
import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import certifi
import pytest

from agent.mcp.client import McpToolExecutionError
from agent.plugins.manager import PluginManager
from agent.plugins.install import PluginInstallResult, install_git_plugin
from agent.tools.registry import ToolRegistry
from bootstrap.app import AppRuntime
from bus.event_bus import EventBus


@pytest.mark.asyncio
async def test_runtime_install_waits_until_latest_is_leasable(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin" / "baseline"
    builtin.mkdir(parents=True)
    (builtin / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class BaselinePlugin(Plugin):\n"
        "    name = 'baseline'\n",
        encoding="utf-8",
    )
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class CandidatePlugin(Plugin):\n"
        "    name = 'candidate'\n"
        "    version = '1.0.0'\n",
        encoding="utf-8",
    )
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
    blocked_source.mkdir()
    (blocked_source / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class BlockedPlugin(Plugin):\n"
        "    name = 'blocked'\n"
        "    version = '1.0.0'\n",
        encoding="utf-8",
    )
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
    assert old_generation.mcp_catalog is not None
    old_server = old_generation.mcp_catalog.servers["runtime_probe"]
    old_client = old_server.client
    old_process = old_client._process
    assert old_process is not None and old_process.pid is not None

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
        assert latest_generation.mcp_catalog is not None
        latest_server = latest_generation.mcp_catalog.servers["runtime_probe"]
        latest_process = latest_server.client._process
        assert latest_process is not None and latest_process.pid is not None
        assert latest_process.pid != old_process.pid

        # 2. 更新后旧 MCP 延迟读取自己的 CA bundle，新旧调用各自保持代际身份。
        old_probe = await _call_runtime_probe(old_server.tools[0])
        latest_probe = await _call_runtime_probe(latest_server.tools[0])
        assert old_probe == {
            "artifact": str(old_artifact),
            "ca_bundle": str(_runtime_ca_bundle(old_artifact)),
            "ca_certificates": old_probe["ca_certificates"],
            "pid": old_process.pid,
            "runtime_version": "v1",
        }
        assert int(old_probe["ca_certificates"]) > 0
        assert latest_probe["artifact"] == str(new_artifact)
        assert latest_probe["pid"] == latest_process.pid
        assert latest_probe["runtime_version"] == "v2"

        # 3. promote 不抢占旧 lease；最后一个旧 reader 离开后才关闭旧进程。
        promoted = await app._promote_plugin(plugin_id)
        assert promoted["publication_state"] == "promoted"
        promoted_lease = manager.snapshot_store.lease()
        assert promoted_lease.snapshot is latest_lease.snapshot
        assert old_process.returncode is None

        await latest_lease.release()
        latest_lease = None
        await promoted_lease.release()
        promoted_lease = None
        assert old_process.returncode is None
        await old_lease.release()
        await manager.snapshot_store.retry_drains()
        assert old_client._process is None
        assert old_process.returncode is not None
        assert latest_process.returncode is None
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
            await manager.discard_latest_candidate(plugin_id)
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
    assert old_generation.mcp_catalog is not None
    old_server = old_generation.mcp_catalog.servers["runtime_probe"]

    try:
        # 1. 建立新 latest 后模拟旧安装器提前删除 CA bundle。
        _write_runtime_mcp_source(source, runtime_version="v2")
        _commit_all(source, "runtime-v2")
        _ = await app._install_plugin(str(source), "lab", "", [])
        latest_lease = manager.snapshot_store.lease(selector="latest")
        _runtime_ca_bundle(old_artifact).unlink()

        # 2. 旧 MCP 在调用时才读取路径，oracle 必须命中原事故而不是静默切新代。
        with pytest.raises(McpToolExecutionError, match="cacert.pem|No such file"):
            await old_server.tools[0].execute()
        assert old_lease.snapshot is manager.current_snapshot
        latest_generation = latest_lease.snapshot.generations[plugin_id]
        assert latest_generation.mcp_catalog is not None
        latest_probe = await _call_runtime_probe(
            latest_generation.mcp_catalog.servers["runtime_probe"].tools[0]
        )
        assert latest_probe["runtime_version"] == "v2"
    finally:
        if latest_lease is not None and latest_lease.active:
            await latest_lease.release()
        if manager.ready_candidate is not None:
            await manager.discard_latest_candidate(plugin_id)
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
    builtin.mkdir(parents=True)
    (builtin / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class BaselinePlugin(Plugin):\n"
        "    name = 'baseline'\n",
        encoding="utf-8",
    )
    sources: dict[str, Path] = {}
    for name in ("alpha", "beta"):
        source = tmp_path / name
        source.mkdir()
        (source / "plugin.py").write_text(
            "from agent.plugins import Plugin\n"
            f"class {name.title()}Plugin(Plugin):\n"
            f"    name = '{name}'\n"
            "    version = '1.0.0'\n",
            encoding="utf-8",
        )
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
    await manager.discard_latest_candidate("alpha@lab")

    entered = threading.Event()
    release = threading.Event()

    def blocking_install(**kwargs: Any) -> PluginInstallResult:
        entered.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release plugin install")
        return install_git_plugin(**kwargs)

    monkeypatch.setattr("agent.plugins.manager.install_git_plugin", blocking_install)
    cancelled_install = asyncio.create_task(
        app._install_plugin(str(sources["beta"]), "lab", "", [])
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
    assert manager.candidate_status()["candidate_plugin_id"] == "beta@lab"
    assert manager.candidate_status()["candidate_state"] == "latest_ready"
    await manager.discard_latest_candidate("beta@lab")
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
        "from agent.plugins import McpServerSpec, Plugin\n"
        "class RuntimeMcpPlugin(Plugin):\n"
        "    name = 'runtime_mcp'\n"
        "    version = '1.0.0'\n"
        "    @classmethod\n"
        "    def mcp_servers(cls):\n"
        "        return [McpServerSpec(name='runtime_probe', command=('python', 'server.py'))]\n",
        encoding="utf-8",
    )

    # 2. server 不缓存文件内容，确保更新后的调用会触发旧绝对路径读取。
    _ = (source / "server.py").write_text(
        "import json, os, ssl, sys\n"
        "from pathlib import Path\n"
        f"RUNTIME_VERSION = {runtime_version!r}\n"
        "ARTIFACT = Path(__file__).resolve().parent\n"
        "TOOLS = [{'name': 'probe', 'description': 'probe runtime', "
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
        "'ca_certificates': context.cert_store_stats()['x509_ca'], 'pid': os.getpid(), "
        "'runtime_version': RUNTIME_VERSION}\n"
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


def _runtime_ca_bundle(root: Path) -> Path:
    """返回与 PATH 实际启动的 MCP Python 版本一致的 certifi 路径。"""

    completed = subprocess.run(
        [
            "python",
            "-c",
            "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    python_dir = completed.stdout.strip()
    if not python_dir.startswith("python"):
        raise AssertionError(f"MCP Python 版本目录无效: {python_dir!r}")
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


async def _call_runtime_probe(tool: Any) -> dict[str, Any]:
    """调用 MCP probe，并严格解析结构化代际证据。"""

    raw = await tool.execute()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise AssertionError(f"MCP probe 返回值不是对象: {parsed!r}")
    return parsed


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
    subprocess.run(["git", "add", "plugin.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "candidate"], cwd=repo, check=True)
