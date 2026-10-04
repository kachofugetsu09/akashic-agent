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
inject = (ServiceKey('e2e.registry'),)
async def apply(ctx):
    await ctx.require(inject[0]).register(ctx)
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


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def install_fixture(root: Path, workspace: Path, home: Path, name: str, text: str) -> None:
    """按正式 Git 安装链准备普通插件，不编辑安装 cache。"""
    from agent.plugins.install import install_git_plugin
    source = root / 'sources' / name
    source.mkdir(parents=True)
    (source / 'plugin.py').write_text(text)
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


async def check_provenance(app, manager, plugins, workspace):
    """来源标签变更后执行真实全量检查与手动 watcher 唤醒。"""
    from agent.plugins.watcher import PluginWatcher
    target = manager.generation('aa_marker')
    token = target.fiber.context.fiber.activation_token
    selected = (workspace / 'runtime/plugin-stable.json').read_bytes()
    effects = (target.data_dir / 'activations.txt').read_bytes()
    archived_marker = (target.code_dir / '.akashic-source.json').read_bytes()
    marker = plugins / 'aa_marker/.akashic-source.json'
    marker.write_text(json.dumps({'commit': 'new-source-evidence'}))
    try:
        await manager.reconcile_changed()
    except RuntimeError as error:
        # 无模型的隔离组合允许既有消费者 PENDING，但不能换代目标。
        assert '未 ACTIVE' in str(error), str(error)
    assert manager.generation('aa_marker') is target
    assert target.fiber.context.fiber.activation_token is token
    assert (workspace / 'runtime/plugin-stable.json').read_bytes() == selected
    assert (target.code_dir / '.akashic-source.json').read_bytes() == archived_marker
    app.plugin_watcher = PluginWatcher(manager, baseline_revision=manager.watch_revision(), interval_seconds=1)
    app.plugin_watcher_task = asyncio.create_task(app.plugin_watcher.run())
    previous = manager._operation
    app.plugin_watcher.wake()
    await wait_for(lambda: manager._operation is not previous and manager._operation.task.done(), 'manual full check')
    app.plugin_watcher.stop()
    await app.plugin_watcher_task
    assert manager.generation('aa_marker') is target
    assert target.fiber.context.fiber.activation_token is token
    assert (target.data_dir / 'activations.txt').read_bytes() == effects
    assert (workspace / 'runtime/plugin-stable.json').read_bytes() == selected
    return {'full_provenance_check': 'passed', 'manual_wake_provenance': 'passed',
            'archive_provenance_retained': 'passed', 'no_selection_rewrite': 'passed'}


async def run(root: Path, core: Path, case: str, seed_core: Path | None) -> dict[str, object]:
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
    if case == 'provenance':
        marker_plugin = plugins / 'aa_marker'
        marker_plugin.mkdir()
        (marker_plugin / 'plugin.py').write_text(PEER.replace("name = 'peer'", "name = 'aa_marker'")
                                                .replace('e2e.peer', 'e2e.marker'))
        (marker_plugin / '.akashic-source.json').write_text(json.dumps({'commit': 'original-source-evidence'}))
    os.environ['AKASHIC_EXTRA_PLUGIN_DIRS'] = str(plugins)
    config = root / 'config.toml'
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path[:2])}
    with (root / 'init.log').open('w') as output:
        subprocess.run([sys.executable, str(core / 'main.py'), 'init', '--config', str(config),
                        '--workspace', str(workspace)], env=env, check=True, stdout=output, stderr=output)
    for name, source in [('z_registry', PROVIDER), ('annotation', CONTRIBUTOR), ('peer', PEER)]:
        install_fixture(root, workspace, home, name, source)
    if seed_core is not None:
        # 用旧版真实启动生成 selection，候选版直接读取其完整归档。
        seed = """import asyncio, sys
from agent.config_models import Config
from bootstrap.app import AppRuntime
from pathlib import Path
async def run():
    app = AppRuntime(Config.load(Path(sys.argv[1]), workspace=Path(sys.argv[2])), Path(sys.argv[2]))
    try:
        await app.start()
    finally:
        await app.shutdown()
asyncio.run(run())
"""
        seed_env = {**env, 'PYTHONPATH': os.pathsep.join([str(seed_core), str(seed_core / 'sdk/python/src')]),
                    'AKASHIC_CORE_ROOT': str(seed_core)}
        with (root / 'seed.log').open('w') as output:
            subprocess.run([sys.executable, '-c', seed, str(config), str(workspace)], cwd=root,
                           env=seed_env, check=True, stdout=output, stderr=output, timeout=60)
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
        if case == 'provenance':
            result = await check_provenance(app, manager, plugins, workspace)
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
        async def hold_contribution():
            async with registry.entries['annotation@lab'].runtime_scope():
                held.set()
                await release.wait()
        holder = asyncio.create_task(hold_contribution())
        await held.wait()
        async with await AsyncAkashic.connect(str(workspace / 'akashic.sock')) as client:
            accepted = await client.request('plugin/uninstall', {'plugin_id': 'annotation@lab'})
            assert accepted['state'] == 'accepted', accepted
            await wait_for(lambda: target.fiber.state is FiberState.UNLOADING, 'target draining')
            assert provider.fiber.state is FiberState.ACTIVE
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
            status = await client.request('plugin/status', {})
            assert not any(p['plugin_id'] == 'annotation@lab' for p in status['plugins'])
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
                'retained_plugin_data': 'passed', 'core': git(core, 'rev-parse', 'HEAD'),
                'source_sha256': {str(p.relative_to(core)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [core / 'agent/plugins/manager.py', core / 'agent/plugins/watcher.py']}}
    finally:
        release.set()
        if holder is not None:
            await asyncio.gather(holder, return_exceptions=True)
        await app.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--case', choices=['uninstall', 'provenance'], default='uninstall')
    parser.add_argument('--seed-core', type=Path)
    args = parser.parse_args()
    root = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix='plugin-uninstall-e2e-'))
    if args.output:
        root.mkdir(parents=True, exist_ok=False)
    print(f'EVIDENCE {root}', flush=True)
    try:
        result = asyncio.run(run(root, args.core_root.resolve(), args.case,
                                 args.seed_core.resolve() if args.seed_core else None))
    except BaseException as error:
        (root / 'result.json').write_text(json.dumps({'status': 'failed', 'error': repr(error)}, indent=2))
        raise
    (root / 'result.json').write_text(json.dumps({'status': 'passed', **result}, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
