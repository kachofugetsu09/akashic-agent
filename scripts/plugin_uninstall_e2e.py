#!/usr/bin/env python3
"""通过真实安装、控制连接与 watcher 验证插件局部卸载。"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

PROVIDER = """from agent.plugin_composition import ServiceKey
api_version = 3
name = 'z_registry'
version = '1'
inject = ()
class Registry:
    def __init__(self):
        self.entries = {}
    async def register(self, ctx):
        async def setup():
            self.entries[ctx.runtime.plugin_id] = ctx
            def close():
                del self.entries[ctx.runtime.plugin_id]
            return close
        await ctx.effect(setup)
async def apply(ctx):
    await ctx.provide(ServiceKey('e2e.registry'), Registry())
"""
CONTRIBUTOR = """from agent.plugin_composition import ServiceKey
api_version = 3
name = 'annotation'
version = '1'
inject = (ServiceKey('e2e.registry'), ServiceKey('content.v2'))
async def decode(source, references):
    return (), {}
async def apply(ctx):
    await ctx.require(inject[0]).register(ctx)
    await ctx.require(inject[1]).register(ctx, {
        'name': 'annotation', 'prompt': 'E2E annotation', 'decode': decode, 'content': {}})
"""
PEER = """from agent.plugin_composition import ServiceKey
api_version = 3
name = 'peer'
version = '1'
inject = ()
async def apply(ctx):
    marker = ctx.runtime.data_dir / 'activations.txt'
    marker.write_text(marker.read_text() + 'start\\n' if marker.exists() else 'start\\n')
    await ctx.provide(ServiceKey('e2e.peer'), 'old')
"""


GATE = """import asyncio
from agent.plugin_composition import ServiceKey
api_version = 3
name = 'clock_gate'
version = '1'
inject = ()
async def apply(ctx):
    await ctx.provide(ServiceKey('e2e.gate'), (asyncio.Event(), asyncio.Event()))
"""
READER = """from agent.plugin_composition import ServiceKey
api_version = 3
name = 'a_reader'
version = '1'
inject = (ServiceKey('e2e.registry'),)
async def apply(ctx):
    marker = ctx.runtime.data_dir / 'activations.txt'
    marker.write_text(marker.read_text() + 'start\\n' if marker.exists() else 'start\\n')
    await ctx.provide(ServiceKey('e2e.reader'), ctx.require(inject[0]))
"""


async def recover(manager, client, root, workspace):
    """真实截止时间取消 provider 后，从已选归档恢复并核对效果次数。"""
    from agent.plugin_composition import ServiceKey, FiberState
    source = root / 'sources/z_registry'
    text = (source / 'plugin.py').read_text().replace("version = '1'", "version = '2'")
    (source / 'plugin.py').write_text(text)
    git(source, 'add', '.')
    git(source, '-c', 'user.name=E2E', '-c', 'user.email=e2e@example.invalid',
        '-c', 'commit.gpgSign=false', 'commit', '-qm', 'provider update')
    entered, release = manager.live_root.context.require(ServiceKey('e2e.gate'))
    peer = manager.generation('peer@lab')
    peer_bytes = (peer.data_dir / 'activations.txt').read_bytes()
    manager.POST_PUBLISH_TIMEOUT_SECONDS = 0.5
    accepted = await client.request('plugin/install', {
        'source': str(source), 'marketplace': 'lab', 'ref': '', 'sparse': [], 'update_id': 'deadline-e2e'})
    assert accepted['state'] == 'accepted', accepted
    await asyncio.wait_for(entered.wait(), 5)
    await wait_for(lambda: manager._operation.task.done(), 'deadline settled')
    assert manager._operation.revoked and manager._operation.committed
    assert asyncio.get_running_loop().time() >= manager._operation.deadline
    assert manager.plugin_status()['operation']['state'] == 'cancelled'
    assert manager.generation('z_registry@lab') is None
    consumer = manager.generation('a_reader@lab')
    assert consumer.fiber.state is FiberState.PENDING
    selected_before = (workspace / 'runtime/plugin-stable.json').read_bytes()
    release.set()
    manager.POST_PUBLISH_TIMEOUT_SECONDS = 300
    await manager.reconcile_changed(plugin_ids=frozenset({'a_reader@lab', 'z_registry@lab'}))
    assert (workspace / 'runtime/plugin-stable.json').read_bytes() == selected_before
    assert consumer is manager.generation('a_reader@lab')
    assert consumer.fiber.state is FiberState.ACTIVE
    assert manager.generation('z_registry@lab').instance.version == '2'
    assert manager.live_root.context.require(ServiceKey('e2e.reader')) is manager.live_root.context.require(ServiceKey('e2e.registry'))
    assert (consumer.data_dir / 'activations.txt').read_text().count('start') == 2
    assert manager.generation('peer@lab') is peer
    assert (peer.data_dir / 'activations.txt').read_bytes() == peer_bytes
    return {'selected_provider_recovery': 'passed', 'pending_consumer_recovery': 'passed',
            'no_selection_rewrite': 'passed', 'unrelated_effects_preserved': 'passed'}


NOTES = """from agent.plugin_composition.ui import UI
api_version = 3
name = 'notes-ui'
version = '1'
inject = (UI,)
async def apply(ctx):
    from . import dashboard
    await ctx.require(UI).register(ctx, web='web_module.js', dashboard=lambda: dashboard,
                                   requires=('shell.pages.v1',))
"""
NOTES_WEB = """export function activate(ctx) {
  return ctx.ui.inject('shell.pages.v1', mount => mount.register({
    id: 'e2e-notes', label: 'Notes', route: 'e2e-notes', iconSvg: '<svg viewBox="0 0 24 24"><path d="M4 4h16v16H4z"/></svg>', section: 'settings',
    render(host) {
      const draft = document.createElement('textarea');
      draft.setAttribute('aria-label', 'E2E draft');
      draft.style.maxWidth = '100%';
      host.append(draft);
      return () => draft.remove();
    }
  }));
}
"""
NOTES_DASHBOARD = """inject = ()
def register(app, context):
    @app.get('/api/dashboard/e2e/notes')
    def read_note():
        return {'note': (context.data_root / 'note.txt').read_text()}
"""


async def web_client_checks(workspace):
    """通过实际 Unix HTTP listener 核对无回复能力时的独立读取。"""
    import httpx
    transport = httpx.AsyncHTTPTransport(uds=str(workspace / 'runtime/web-chat.sock'))
    async with httpx.AsyncClient(transport=transport, base_url='http://localhost') as client:
        for path in ['/api/chat/health', '/api/chat/web-ui/bootstrap', '/api/chat/web-ui/state']:
            response = await client.get(path)
            assert response.status_code == 200, (path, response.status_code, response.text)


async def web_browser_checks(core, root, workspace, config):
    """启动正式 Web Shell，验证实际 catalog 身份与 Chromium 中的草稿。"""
    import socket
    import httpx
    import uvicorn
    from bootstrap.web_shell import create_web_shell_app
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    url = f'http://127.0.0.1:{listener.getsockname()[1]}'
    shell = uvicorn.Server(uvicorn.Config(create_web_shell_app(config, workspace), log_level='error'))
    task = asyncio.create_task(shell.serve(sockets=[listener]))
    try:
        await wait_for(lambda: shell.started or task.done(), 'Web Shell startup')
        if task.done():
            task.result()
            raise RuntimeError('Web Shell exited before startup')
        async with httpx.AsyncClient(base_url=url) as client:
            response = await client.get('/api/chat/web-ui/bootstrap')
            assert response.status_code == 200, response.text
            bootstrap = response.json()
            module = next(item for item in bootstrap['modules'] if item['pluginId'] == 'notes-ui@lab')
            headers = {'X-Akashic-Web-Snapshot': bootstrap['snapshotId'],
                       'X-Akashic-Web-Catalog': bootstrap['catalogId'],
                       'X-Akashic-Web-Module': module['pluginId'],
                       'X-Akashic-Web-Generation': module['generationId']}
            valid = await client.get('/api/dashboard/e2e/notes', headers=headers)
            assert valid.status_code == 200 and valid.json()['note'] == 'kept-note', valid.text
            invalid = await client.get('/api/dashboard/e2e/notes', headers={**headers, 'X-Akashic-Web-Catalog': '0' * 64})
            assert invalid.status_code == 409 and invalid.json()['code'] == 'stale_catalog', invalid.text
        process = await asyncio.create_subprocess_exec(
            'node', str(ROOT / 'scripts/plugin_uninstall_browser.mjs'), url, str(root),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        output, _ = await asyncio.wait_for(process.communicate(), 40)
        (root / 'browser.log').write_bytes(output)
        assert process.returncode == 0, output.decode()
    finally:
        shell.should_exit = True
        await task
        listener.close()


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def install_fixture(root: Path, workspace: Path, home: Path, name: str, text: str, files: dict[str, str] | None = None) -> None:
    """按正式 Git 安装链准备普通插件，不编辑安装 cache。"""
    from agent.plugins.install import install_git_plugin
    source = root / 'sources' / name
    source.mkdir(parents=True)
    (source / 'plugin.py').write_text(text)
    for filename, content in (files or {}).items():
        (source / filename).write_text(content)
    git(source, 'init', '-q')
    git(source, 'add', '.')
    git(source, '-c', 'user.name=E2E', '-c', 'user.email=e2e@example.invalid',
        '-c', 'commit.gpgSign=false', 'commit', '-qm', 'fixture')
    install_git_plugin(workspace=workspace, source=str(source), marketplace='lab', plugins_home=home)


async def wait_for(check, message: str) -> None:
    """只轮询可观察的运行状态，超时保留明确失败。"""
    async with asyncio.timeout(15):
        while not check():
            await asyncio.sleep(0.02)


async def run(root: Path, core: Path, case: str) -> dict[str, object]:
    """隔离全部路径并通过实际控制协议触发卸载和源码更新。"""
    # 1. 初始化独占状态，复制候选源码，保留完整输入和报告。
    sys.path[:0] = [str(core), str(core / 'sdk/python/src')]
    for key in tuple(os.environ):
        if key.startswith('AKASHIC_'):
            del os.environ[key]
    home, workspace = root / 'plugin-home', root / 'workspace'
    os.environ.update(HOME=str(root / 'home'), XDG_CONFIG_HOME=str(root / 'home/.config'),
                      AKASHIC_PLUGIN_HOME=str(home), AKASHIC_CORE_ROOT=str(core))
    (root / 'home').mkdir()
    plugins = root / 'plugins'
    shutil.copytree(core / 'plugins', plugins, ignore=shutil.ignore_patterns('__pycache__'))
    os.environ['AKASHIC_EXTRA_PLUGIN_DIRS'] = str(plugins)
    config = root / 'config.toml'
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path[:2])}
    with (root / 'init.log').open('w') as output:
        subprocess.run([sys.executable, str(core / 'main.py'), 'init', '--config', str(config),
                        '--workspace', str(workspace)], env=env, check=True, stdout=output, stderr=output)
    provider = PROVIDER
    fixtures = [('annotation', CONTRIBUTOR), ('peer', PEER)]
    if case == 'recovery':
        fixtures += [('clock_gate', GATE), ('a_reader', READER)]
        provider = provider.replace('inject = ()', "inject = (ServiceKey('e2e.gate'),)").replace(
            'async def apply(ctx):', "async def apply(ctx):\n    if version == '2':\n        entered, release = ctx.require(inject[0])\n        entered.set()\n        await release.wait()")
    for name, source in [('z_registry', provider), *fixtures]:
        install_fixture(root, workspace, home, name, source)
    if case in {'ui', 'client'}:
        install_fixture(root, workspace, home, 'notes-ui', NOTES,
                        {'web_module.js': NOTES_WEB, 'dashboard.py': NOTES_DASHBOARD})
        (workspace / 'plugin-data/notes-ui-lab/note.txt').write_text('kept-note')
    from akashic_sdk import AsyncAkashic
    from agent.config_models import Config
    from agent.plugin_composition import FiberState, ServiceKey
    from agent.plugins.watcher import PluginWatcher
    from bootstrap.app import AppRuntime
    from session.log import MessageLog
    from session.message import Input
    # 一条真实 Message 作为受保护的历史事实。
    log = MessageLog(workspace / 'sessions.db')
    log.writer('e2e-session', author='user', source='e2e', body_types=(Input,), content={}).append(
        'kept-message', Input(()))
    log.close()
    def message_rows():
        with sqlite3.connect(workspace / 'sessions.db') as db:
            return db.execute('SELECT * FROM messages ORDER BY seq').fetchall()
    messages_before = message_rows()
    app = AppRuntime(Config.load(config, workspace=workspace), workspace)
    release, held = asyncio.Event(), asyncio.Event()
    holder = None
    try:
        await app.start()
        manager = app.core.plugin_manager
        # 2. 让 watcher 以已有漂移为基线；这正是部署后才卸载的现场条件。
        app.plugin_watcher.stop()
        await app.plugin_watcher_task
        if case == 'client':
            assert manager.generation('reply').fiber.state is FiberState.PENDING
            await web_client_checks(workspace)
            return {'web_without_reply': 'passed'}
        if case == 'recovery':
            async with await AsyncAkashic.connect(str(workspace / 'akashic.sock')) as client:
                result = await recover(manager, client, root, workspace)
            assert message_rows() == messages_before
            return result
        peer_source = root / 'sources/peer'
        (peer_source / 'plugin.py').write_text(PEER.replace("'old'", "'new'"))
        git(peer_source, 'add', '.')
        git(peer_source, '-c', 'user.name=E2E', '-c', 'user.email=e2e@example.invalid',
            '-c', 'commit.gpgSign=false', 'commit', '-qm', 'changed source')
        # 正式安装准备新源码，但不提交运行选择；模拟选中归档与来源不同。
        from agent.plugins.install import install_git_plugin
        install_git_plugin(workspace=workspace, source=str(peer_source), marketplace='lab', plugins_home=home)
        content_marker = plugins / 'content/.akashic-source.json'
        content_marker.write_text(json.dumps({'commit': 'different-source-evidence'}))
        notified = asyncio.Event()
        async def after_reconcile():
            notified.set()
        app.plugin_watcher = PluginWatcher(manager, baseline_revision=manager.watch_revision(),
                                            interval_seconds=0.02, after_reconcile=after_reconcile)
        app.plugin_watcher_task = asyncio.create_task(app.plugin_watcher.run())
        provider, peer = manager.generation('z_registry@lab'), manager.generation('peer@lab')
        content = manager.generation('content')
        tokens = [(g, g.fiber, g.fiber.context.fiber.activation_token, g.archive_ref) for g in (provider, peer, content)]
        peer_effects = (peer.data_dir / 'activations.txt').read_bytes()
        target = manager.generation('annotation@lab')
        target_data = target.data_dir / 'keep.bin'
        target_data.write_bytes(b'user-owned-data')
        archive_before = (target.code_dir / 'plugin.py').read_bytes()
        registry = manager.live_root.context.require(ServiceKey('e2e.registry'))
        if case == 'ui':
            from agent.plugin_composition.ui import WEB_UI
            ui = manager.live_root.context.require(WEB_UI)
            bootstrap_before = await ui.bootstrap()
        async def hold_contribution():
            content_service = manager.live_root.context.require(ServiceKey('content.v2'))
            async with content_service.bind() as view:
                assert 'E2E annotation' in view.prompts
                held.set()
                await release.wait()
        holder = asyncio.create_task(hold_contribution())
        await wait_for(lambda: held.is_set() or holder.done(), 'Content view acquisition')
        if holder.done():
            holder.result()
        async with await AsyncAkashic.connect(str(workspace / 'akashic.sock')) as client:
            accepted = await client.request('plugin/uninstall', {'plugin_id': 'annotation@lab'})
            assert accepted['state'] == 'accepted', accepted
            await wait_for(lambda: target.fiber.state is FiberState.UNLOADING, 'target draining')
            assert provider.fiber.state is FiberState.ACTIVE
            if case == 'ui':
                assert await ui.bootstrap() == bootstrap_before
                assert (await ui.state())['updating'] is True
                await web_client_checks(workspace)
                await web_browser_checks(core, root, workspace, config)
            release.set()
            await holder
            await wait_for(lambda: manager.plugin_status()['operation']['state'] == 'done', 'uninstall done')
            await asyncio.wait_for(notified.wait(), 15)
            # 3. 验证无关实例、外部效果、历史与用户数据的完整内容。
            for generation, fiber, token, archive_ref in tokens:
                assert manager.generation(generation.plugin_id) is generation, generation.plugin_id
                assert generation.fiber is fiber and fiber.context.fiber.activation_token is token, generation.plugin_id
                assert generation.archive_ref == archive_ref, generation.plugin_id
            assert (peer.data_dir / 'activations.txt').read_bytes() == peer_effects
            assert target_data.read_bytes() == b'user-owned-data'
            assert (target.code_dir / 'plugin.py').read_bytes() == archive_before
            assert message_rows() == messages_before
            assert 'annotation@lab' not in registry.entries
            async with manager.live_root.context.require(ServiceKey('content.v2')).bind() as view:
                assert 'E2E annotation' not in view.prompts
            status = await client.request('plugin/status', {})
            assert all(p['selected_ref'] is None for p in status['plugins']
                       if p['plugin_id'] == 'annotation@lab')
            assert not (home / 'cache/lab/annotation').exists()
            # 来源证据本身变化不会触发实例换代；真实代码和配置变化仍生效。
            content_marker.write_text(json.dumps({'commit': 'another-source-evidence'}))
            (peer_source / 'plugin.py').write_text(PEER.replace("'old'", "'newest'"))
            git(peer_source, 'add', '.')
            git(peer_source, '-c', 'user.name=E2E', '-c', 'user.email=e2e@example.invalid',
                '-c', 'commit.gpgSign=false', 'commit', '-qm', 'next real change')
            notified.clear()
            install_git_plugin(workspace=workspace, source=str(peer_source), marketplace='lab', plugins_home=home)
            await asyncio.wait_for(notified.wait(), 15)
            assert manager.generation('peer@lab') is not peer
            assert manager.generation('content') is content
            assert manager.live_root.context.require(ServiceKey('e2e.peer')) == 'newest'
            from agent.plugin_composition.config_input import save_config
            updated = manager.generation('peer@lab')
            notified.clear()
            save_config(updated.data_dir, {'label': 'changed'})
            await asyncio.wait_for(notified.wait(), 15)
            assert manager.generation('peer@lab') is not updated
            assert manager.generation('peer@lab').config_projection == {'label': 'changed'}
        return {'config_update': 'passed', 'uninstall_scope': 'passed', 'real_source_update': 'passed',
                'provenance_only_change': 'passed', 'protected_messages': 'passed',
                'retained_plugin_data': 'passed',
                **({'web_during_uninstall': 'passed', 'stale_identity_rejected': 'passed',
                    'browser_draft_focus': 'passed'} if case == 'ui' else {})}
    finally:
        release.set()
        if holder is not None:
            await asyncio.gather(holder, return_exceptions=True)
        await app.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--case', choices=['uninstall', 'recovery', 'ui', 'client'], default='uninstall')
    args = parser.parse_args()
    root = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix='plugin-uninstall-e2e-'))
    if args.output:
        root.mkdir(parents=True, exist_ok=False)
    print(f'EVIDENCE {root}', flush=True)
    try:
        result = asyncio.run(run(root, args.core_root.resolve(), args.case))
    except BaseException as error:
        (root / 'result.json').write_text(json.dumps({'status': 'failed', 'error': repr(error)}, indent=2))
        raise
    result['core'] = git(args.core_root.resolve(), 'rev-parse', 'HEAD')
    result['source_sha256'] = {name: hashlib.sha256((args.core_root / name).read_bytes()).hexdigest()
                               for name in ['agent/plugins/manager.py', 'agent/plugins/watcher.py',
                                            'plugins/ui/plugin.py', 'plugins/akashic_clients/plugin.py',
                                            'plugins/akashic_clients/capabilities.py', 'plugins/akashic_clients/channel.py',
                                            'frontend/dashboard/src/webHost.ts', 'frontend/dashboard/src/main.tsx']}
    (root / 'result.json').write_text(json.dumps({'status': 'passed', **result}, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
