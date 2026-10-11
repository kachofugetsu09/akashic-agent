"""旧 Core 安装外置插件后，验证预检、启停迁移和真实发布。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.plugins.selection import PluginSelection, SelectionConflictError
from scripts.deployment_composition_scenario import commit, distribution, git, manager, plugin
from scripts.install_plugin_distribution import publish_distribution

SEED = '''import asyncio, sys
from pathlib import Path
from bootstrap.init_workspace import init_workspace
from agent.plugins.manager import PluginManager
from agent.migrations.runner import MigrationRunner
base = Path(sys.argv[1])
init_workspace(config_path=base/'config.toml', workspace=base/'workspace')
MigrationRunner(repo_root=Path.cwd(), config_path=base/'config.toml', workspace=base/'workspace').run()
async def run():
    runtime = PluginManager([], workspace=base/'workspace', installed_cache_root=base/'home/cache')
    await runtime.load_all()
    await runtime.install(source=str(base/'external'), marketplace='lab', ref_name='', sparse_paths=[], update_id='seed')
    await runtime._operation.task
    await runtime.terminate_all()
asyncio.run(run())
'''


async def run(old_core: Path, *, disabled: bool) -> None:
    """由旧安装器产生真实状态，当前发布器负责迁移及最终启停检查。"""
    base = Path(tempfile.mkdtemp(prefix='deployment-legacy-choices-'))
    work, home, config = base/'workspace', base/'home', base/'config.toml'
    os.environ.update(HOME=str(base/'user'), AKASHIC_PLUGIN_HOME=str(home),
                      AKASHIC_PLUGIN_DISTRIBUTION='', AKASHIC_EXTRA_PLUGIN_DIRS='')
    external = base/'external'
    external.mkdir()
    git(external, 'init', '-q')
    source = external/'plugin.py'
    source.write_text("api_version=3\nname='outside'\nversion='1'\nasync def apply(ctx): pass\n")
    commit(external)
    environment = {**os.environ, 'PYTHONPATH': os.pathsep.join((str(old_core), str(old_core/'sdk/python/src')))}
    subprocess.run([sys.executable, '-c', SEED, str(base)], cwd=old_core, env=environment, check=True)
    # 1. 保留旧版安装产生的选择，覆盖升级时用户明确禁用的场景。
    if disabled:
        config.write_text(config.read_text() + '\n[agent.plugins]\ndisabled_builtin = ["outside@lab"]\n')
    before = config.read_bytes()
    legacy = (home/'manifest.toml').read_bytes()
    selection = PluginSelection(work)
    original = selection.read()
    assert not (work/'bundle.patch.toml').exists()
    source.write_text(source.read_text().replace("version='1'", "version='2'"))
    revision = commit(external)
    bundle = base/'external.bundle'
    git(external, 'bundle', 'create', str(bundle), 'HEAD')
    repo = base/'repo'
    repo.mkdir()
    git(repo, 'init', '-q')
    plugin(repo, 'alpha')
    shutil.copytree(ROOT/'plugins/ledger', repo/'plugins/ledger', ignore=shutil.ignore_patterns('__pycache__'))
    target = distribution(repo, base/'distribution', ['alpha', 'ledger'], ['alpha'])
    plan = base/'plan.json'
    plan.write_text(json.dumps({'schema_version': 1, 'expected_root_ref': original, 'targets': [{
        'plugin_id': 'outside@lab', 'bundle_relative_path': bundle.name,
        'bundle_sha256': hashlib.sha256(bundle.read_bytes()).hexdigest(), 'target_commit': revision,
    }]}))
    options = dict(distribution=target, workspace=work, plugins_home=home, config_path=config, plan=plan, inputs=base)
    # 2. 预检不得提前迁移配置；停止期迁移后才裁定最终启停。
    assert publish_distribution(**options, preflight_only=True)['status'] == 'preflight_ok'
    assert config.read_bytes() == before and selection.read() == original
    assert not (work/'bundle.patch.toml').exists()
    if disabled:
        try:
            publish_distribution(**options)
        except SelectionConflictError as error:
            assert '迁移后的外置目标未启用: outside@lab' in str(error), error
        else:
            raise AssertionError('发布器接受了迁移后禁用的目标')
        assert selection.read() == original
    else:
        assert publish_distribution(**options)['status'] == 'selected_not_started'
        runtime = await manager(work, home, target)
        try:
            assert runtime.generation('outside@lab').instance.version == '2'
        finally:
            await runtime.terminate_all()
    assert (home/'manifest.toml').read_bytes() == legacy
    print(json.dumps({'evidence': str(base), 'disabled': disabled, 'passed': True}), flush=True)


if __name__ == '__main__':
    old_core = Path(sys.argv[1]).resolve(strict=True)
    asyncio.run(run(old_core, disabled=False))
    asyncio.run(run(old_core, disabled=True))
