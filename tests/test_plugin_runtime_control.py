from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.plugin_workspace import initialize_plugin_workspace
from agent.plugin_composition.mcp_slots import MCP_SERVERS
from agent.plugins.generation import PluginGeneration
from agent.plugins.manager import PluginManager
from agent.plugins.install import install_git_plugin
from agent.control.client import ControlClient
from agent.control.service import ControlService
from bus.event_bus import EventBus
from infra.control.socket import SocketAppServer
from session.log import MessageCatalog, MessageLog, SessionAttributes
from session.message import ContentPart, ContentReferences, Input


@pytest.mark.asyncio
async def test_live_mcp_update_keeps_archive_and_data_after_owner_drain(
    tmp_path: Path,
) -> None:
    """Selected B starts after A drains while A's archive and data remain readable."""
    source, manager, bus, old_artifact = await _start_runtime_mcp(tmp_path)
    plugin_id = "runtime_mcp@lab"
    old = manager.generation(plugin_id)
    assert old is not None and old.fiber is not None
    old_probe = await _call_runtime_probe(_mcp_server(_mcp_registration(manager)))
    old_ca_bundle = _runtime_ca_bundle(manager, old)
    data = old.data_dir
    marker = data / "retained.json"
    marker.write_text('{"keep": true}\n', encoding="utf-8")
    operation = None
    try:
        async with old.fiber.context.runtime_scope():
            _write_runtime_mcp_source(source, runtime_version="v2")
            _commit_all(source, "runtime-v2")
            accepted = await manager.install(
                source=str(source), marketplace="lab", ref_name="", sparse_paths=[],
                update_id="mcp-v2",
            )
            assert accepted.state == "accepted"
            operation = manager._operation
            assert operation is not None and not operation.task.done()
            assert old_artifact.is_dir()
            assert old_probe["artifact"] == str(old.code_dir)
            assert old_probe["ca_bundle"] == str(old_ca_bundle)
        assert operation is not None
        await asyncio.wait_for(operation.task, 30)
        current = manager.generation(plugin_id)
        assert current is not None and current is not old
        assert current.archive_ref != old.archive_ref
        assert current.fiber is not None
        assert old.scope.closed
        assert old_artifact.is_dir()
        assert marker.read_text(encoding="utf-8") == '{"keep": true}\n'
        current_probe = await _call_runtime_probe(_mcp_server(_mcp_registration(manager)))
        assert current_probe["runtime_version"] == "v2"
        assert current_probe["artifact"] == str(current.code_dir)
        assert current_probe["data_dir"] == str(data)
        assert current_probe["workspace"] == str(tmp_path / "workspace")
        assert current_probe["pid"] != old_probe["pid"]
        assert int(current_probe["ca_certificates"]) > 0
    finally:
        if operation is not None and not operation.task.done():
            await asyncio.gather(operation.task, return_exceptions=True)
        await manager.terminate_all()
        await bus.aclose()


@pytest.mark.asyncio
async def test_live_mcp_rejects_corrupt_python_environment(tmp_path: Path) -> None:
    """The selected archive fails loudly when its pinned CA file disappears."""
    _source, manager, bus, _artifact = await _start_runtime_mcp(tmp_path)
    try:
        current = manager.generation("runtime_mcp@lab")
        assert current is not None
        server = _mcp_server(_mcp_registration(manager))
        assert (await _call_runtime_probe(server))["runtime_version"] == "v1"
        _runtime_ca_bundle(manager, current).unlink()
        with pytest.raises(RuntimeError, match="Python 环境内容缺失或损坏"):
            await _call_runtime_probe(server)
    finally:
        await manager.terminate_all()
        await bus.aclose()


@pytest.mark.asyncio
async def test_socket_uninstall_returns_accepted_before_local_owner_closes(
    tmp_path: Path,
) -> None:
    """Socket receives Manager accepted; disconnect does not cancel the real owner task."""

    source = tmp_path / "runtime-mcp-source"
    _write_runtime_mcp_source(source, runtime_version="v1")
    _commit_all(source, "runtime-v1")
    provider = tmp_path / "providers/assets"
    shutil.copytree(
        Path(__file__).parents[1] / "plugins/assets", provider,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        Path(__file__).parents[1] / "plugins/mcp", provider.parent / "mcp",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    bus = EventBus()
    workspace = tmp_path / "workspace"
    initialize_plugin_workspace(workspace)
    manager = PluginManager(
        plugin_dirs=[provider.parent],
        event_bus=bus,
        workspace=workspace,
        installed_cache_root=tmp_path / "plugins-home" / "cache",
    )
    installed = install_git_plugin(
        workspace=workspace,
        source=str(source),
        marketplace="lab",
        plugins_home=manager.installed_plugins_home,
    )
    await manager.load_all()
    plugin_id = "runtime_mcp@lab"
    artifact = installed.installed_path
    production_data = workspace / "plugin-data" / "runtime_mcp-lab"
    production_marker = production_data / "retained.json"
    production_marker.write_text('{"keep": true}\n', encoding="utf-8")
    target = manager.generation(plugin_id)
    assert target is not None and target.fiber is not None
    target_call = target.fiber.context.fiber.acquire_call(
        target.fiber.context.fiber.activation_token
    )

    message_log = MessageLog(workspace / "sessions.db")
    message_log.ensure_session("socket-marker", SessionAttributes())
    message_writer = message_log.writer(
        "socket-marker", author="user", source="conversation", body_types=(Input,),
        content={"text": lambda _part: ContentReferences()},
    )
    message_writer.append(
        "socket-input", Input((ContentPart("text", "socket input"),)),
    )
    message_reader = message_log.reader("socket-marker")
    messages_before = message_reader.snapshot()

    async def reject_accept(_session, _message_id, _incoming):
        raise AssertionError("卸载测试不应接纳消息")

    async def no_reply_status(_session):
        if False:
            yield {}

    service = ControlService(
        MessageCatalog(message_log),
        workspace,
        accept=reject_accept,
        reply_status=no_reply_status,
        attachments=lambda _ids: (),
        plugin_status=manager.plugin_status,
        plugin_uninstall=manager.uninstall,
    )
    server = SocketAppServer(tmp_path / "control.sock", service)
    await server.start()
    request_task: asyncio.Task[object] | None = None
    client = None
    try:
        client = await ControlClient.connect(str(server.endpoint))
        request_task = asyncio.create_task(
            client.request("plugin/uninstall", {"plugin_id": plugin_id})
        )
        response = await request_task
        assert isinstance(response, dict)
        assert response["plugin_id"] == plugin_id
        assert response["state"] == "accepted"
        operation = manager._operation
        assert operation is not None and not operation.task.done()
        assert artifact.is_dir()
        status = await client.request("plugin/status", {})
        assert isinstance(status, dict)
        plugins = status["plugins"]
        assert isinstance(plugins, list)
        builtin_status = {item["plugin_id"]: item for item in plugins if isinstance(item, dict)}
        assert builtin_status["assets"]["cache_exists"] is False
        assert builtin_status["mcp"]["cache_exists"] is False
        target_status = next(item for item in plugins if isinstance(item, dict) and item["plugin_id"] == plugin_id)
        assert target_status["installed"] is True
        assert target_status["enabled"] is False
        assert target_status["cache_exists"] is True

        await client.close()
        client = None
        assert not operation.task.done()
        target_call.release()
        result = await asyncio.gather(operation.task, return_exceptions=True)
        assert len(result) == 1 and isinstance(result[0], dict)
        assert result[0]["plugin_id"] == plugin_id
        assert result[0]["state"] == "removed"
        assert not artifact.exists()
        assert production_marker.read_text(encoding="utf-8") == '{"keep": true}\n'
        assert message_reader.snapshot() == messages_before
        client = await ControlClient.connect(str(server.endpoint))
        status = await client.request("plugin/status", {})
        assert isinstance(status, dict)
        plugins = status["plugins"]
        assert isinstance(plugins, list)
        operation_status = status["operation"]
        assert isinstance(operation_status, dict)
        target_status = next(item for item in plugins if isinstance(item, dict) and item["plugin_id"] == plugin_id)
        assert target_status["installed"] is False
        assert target_status["cache_exists"] is False
        assert operation_status["state"] == "done"
    finally:
        if request_task is not None and not request_task.done():
            request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)
        if client is not None:
            await client.close()
        if not target_call._released:
            target_call.release()
        await server.stop()
        await service.shutdown()
        message_log.close()
        await manager.terminate_all()
        await bus.aclose()


async def _start_runtime_mcp(
    tmp_path: Path,
) -> tuple[Path, PluginManager, EventBus, Path]:
    """Install one real archived MCP source, then boot its selected Root."""
    source = tmp_path / "runtime-mcp-source"
    _write_runtime_mcp_source(source, runtime_version="v1")
    _commit_all(source, "runtime-v1")
    provider = tmp_path / "providers/assets"
    shutil.copytree(
        Path(__file__).parents[1] / "plugins/assets", provider,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        Path(__file__).parents[1] / "plugins/mcp", provider.parent / "mcp",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    bus = EventBus()
    workspace = tmp_path / "workspace"
    initialize_plugin_workspace(workspace)
    manager = PluginManager(
        plugin_dirs=[provider.parent], event_bus=bus, workspace=workspace,
        installed_cache_root=tmp_path / "plugins-home" / "cache",
    )
    installed = install_git_plugin(
        workspace=workspace, source=str(source), marketplace="lab",
        plugins_home=manager.installed_plugins_home,
    )
    await manager.load_all()
    return source, manager, bus, installed.installed_path


def _mcp_registration(manager: PluginManager):
    """Read the registration from the exact live Root."""
    root = manager.live_root
    assert root is not None
    return root.context.require(MCP_SERVERS)._entries["runtime_probe"]


def _write_runtime_mcp_source(source: Path, *, runtime_version: str) -> None:
    """写入每次调用都读取实际 Python 环境 CA 的 MCP 插件。"""

    # 1. 插件版本保持不变，用 server 内容变化制造同版本新 revision。
    source.mkdir(parents=True, exist_ok=True)
    _ = (source / "plugin.py").write_text(
        "from agent.plugin_composition import MCP_SERVERS, McpServerDefinition\n"
        "api_version = 3\n"
        "name = 'runtime_mcp'\n"
        "version = '1.0.0'\n"
        "from agent.plugin_composition.assets import INSTALLED_ASSETS\n"
        "inject = (MCP_SERVERS, INSTALLED_ASSETS)\n"
        "async def apply(ctx):\n"
        "    await ctx.require(INSTALLED_ASSETS).register(ctx, 'skills', 'skills')\n"
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
        "import certifi, json, os, ssl, sys\n"
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
        "            if message['params']['name'] == 'poll_feed':\n"
        "                (DATA_DIR / 'feed-cursor.json').write_text('advanced\\n', encoding='utf-8')\n"
        "            ca_bundle = Path(certifi.where())\n"
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


def _runtime_ca_bundle(manager: PluginManager, generation: PluginGeneration) -> Path:
    """从启动命令固定的 interpreter 取得真实环境 CA，不猜测安装目录。"""
    interpreter = manager._resolve_runtime_command(
        generation, ("python", "mcp/server.py"), "."
    )[0]
    result = subprocess.run(
        [interpreter, "-I", "-B", "-c", "import certifi; print(certifi.where())"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


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

    async with server() as opened:
        async with opened.route() as route:
            result = await route.call("probe", {})
    if result.tool_error:
        raise AssertionError(f"MCP probe 调用失败: {result.output}")
    raw = result.output
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise AssertionError(f"MCP probe 返回值不是对象: {parsed!r}")
    return parsed


def _mcp_server(entry):
    from agent.plugin_composition.mcp_slots import MCP_SERVERS
    return lambda: entry.ctx.require(MCP_SERVERS).open(entry.ctx, entry.definition.name)


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
