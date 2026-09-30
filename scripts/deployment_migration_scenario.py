"""在全新临时安装中验证迁移的数据归属、配置恢复和旧账本升级。"""
import asyncio
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.deployment_composition_scenario import ROOT, plugin, git, commit, distribution, ensure_profile, manager, migration, selected, snapshot
from agent.plugins.manifest import workspace_plugin_data_dir, set_plugin_enabled
from agent.plugin_composition.archive import decode_config
from agent.plugins.selection import PluginSelection
from agent.plugins.install import install_git_plugin
from agent.migrations.runner import MigrationRunner

async def setup(tag):
    """创建只属于本次场景的完整发行版与 workspace。"""
    root = Path(tempfile.mkdtemp(prefix=tag))
    print('EVIDENCE', root, flush=True)
    repo = root / 'repo'
    repo.mkdir()
    git(repo, 'init', '-q')
    plugin(repo, 'alpha')
    old = distribution(repo, root / 'old', ['alpha'], ['alpha'])
    work, home, config = root / 'workspace', root / 'home', root / 'config.toml'
    receipt = work / 'runtime/distribution-install.json'
    (root / 'home-config').mkdir()
    os.environ['HOME'] = str(root / 'home-config')
    os.environ['AKASHIC_PLUGIN_HOME'] = str(home)
    os.environ.pop('AKASHIC_PLUGIN_DISTRIBUTION', None)
    subprocess.run([sys.executable, str(ROOT / 'main.py'), 'init', '--config', str(config), '--workspace', str(work)], check=True, stdout=subprocess.DEVNULL)
    ensure_profile(old, old / 'profiles/default.json', workspace=work, plugins_home=home, config_path=config, receipt_path=receipt)
    return root, repo, old, work, home, config, receipt

async def main():
    """依次验证三条真实故障路径，保留所有临时证据。"""
    root, repo, old, work, home, config, receipt = await setup('review-owner-fixed-')
    m = await manager(work, home, old)
    ext = root / 'external'
    ext.mkdir()
    git(ext, 'init', '-q')
    (ext / 'plugin.py').write_text("api_version=3\nname='alpha'\nversion='external'\nasync def apply(ctx): pass\n")
    commit(ext)
    before_cache = snapshot(home / 'cache')
    before_selection = PluginSelection(work).read()
    try:
        install_git_plugin(workspace=work, source=str(ext), marketplace='release', plugins_home=home)
    except ValueError as error:
        assert '不能接管内置数据身份' in str(error), error
    else:
        raise AssertionError('new same-ID external install accepted')
    assert snapshot(home / 'cache') == before_cache
    assert PluginSelection(work).read() == before_selection
    await m.terminate_all()
    # 构造旧版曾允许的同 ID 安装：暂存尚未引入该规则时不存在的 receipt。
    # 这只准备历史 fixture，不是当前允许的操作流程。
    historical_receipt = receipt.with_name('historical-receipt.json')
    receipt.rename(historical_receipt)
    try:
        m = await manager(work, home)
        await m.install(source=str(ext), marketplace='release', ref_name='', sparse_paths=[], update_id='same-id-external')
        await m._operation.task
        await m.terminate_all()
    finally:
        historical_receipt.rename(receipt)
    assert selected(work)['alpha@release'][1]['source_type'] == 'installed'
    data = workspace_plugin_data_dir(work, 'alpha', 'release')
    (data / 'external-owned').write_text('external')
    migration(repo, 'alpha', 'review_builtin_upgrade', "from agent.migrations.context import current_migration_context\ndef run(connection):\n    root=current_migration_context().bundle_data_roots['alpha']\n    (root/'external-owned').write_text('wrong builtin write')\nstep(run)\n")
    new = distribution(repo, root / 'new', ['alpha'], ['alpha'])
    for disabled in [False, True]:
        set_plugin_enabled('alpha@release', enabled=not disabled, plugins_home=home)
        before = PluginSelection(work).read()
        for startup in [False, True]:
            os.environ['AKASHIC_PLUGIN_DISTRIBUTION'] = str(new)
            try:
                if startup:
                    MigrationRunner(repo_root=ROOT, config_path=config, workspace=work, startup_selection=True).run()
                else:
                    ensure_profile(new, new / 'profiles/default.json', workspace=work, plugins_home=home, config_path=config, receipt_path=receipt)
            except RuntimeError as error:
                assert '数据身份' in str(error), error
            else:
                raise AssertionError('same ID external data accepted')
            assert (data / 'external-owned').read_text() == 'external'
            assert PluginSelection(work).read() == before
    print('PASS same-ID enabled/disabled ensure/startup', flush=True)

    root, repo, old, work, home, config, receipt = await setup('review-config-fixed-')
    m = await manager(work, home, old)
    def fail(ref):
        raise RuntimeError('injected file publish failure')
    m._publish_config_input = fail
    await m.apply_config_input('alpha@release', 'failed-file-publish', m.read_config_input('alpha@release')['input_ref'], {'user': 'committed-new'})
    try:
        await m._operation.task
    except Exception:
        pass
    await m.terminate_all()
    assert decode_config(selected(work)['alpha@release'][1]['config']) == {'user': 'committed-new'}
    before = PluginSelection(work).read()
    try:
        ensure_profile(old, old / 'profiles/default.json', workspace=work, plugins_home=home, config_path=config, receipt_path=receipt)
    except RuntimeError as error:
        assert '配置提交尚待原 runtime 恢复' in str(error), error
    else:
        raise AssertionError('unsettled configuration accepted')
    assert PluginSelection(work).read() == before
    migration(repo, 'alpha', 'review_config_upgrade', 'step("CREATE TABLE review_config_done (value INTEGER)")\n')
    new = distribution(repo, root / 'new', ['alpha'], ['alpha'])
    os.environ['AKASHIC_PLUGIN_DISTRIBUTION'] = str(new)
    try:
        MigrationRunner(repo_root=ROOT, config_path=config, workspace=work, startup_selection=True).run()
    except RuntimeError as error:
        assert '配置提交尚待原 runtime 恢复' in str(error), error
    else:
        raise AssertionError('startup migrated before config recovery')
    os.environ['AKASHIC_PLUGIN_DISTRIBUTION'] = str(old)
    assert MigrationRunner(repo_root=ROOT, config_path=config, workspace=work, startup_selection=True).run().state == 'current'
    m = await manager(work, home, old)
    assert m.read_config_update('alpha@release', 'failed-file-publish')['state'] == 'active'
    await m.terminate_all()
    ensure_profile(new, new / 'profiles/default.json', workspace=work, plugins_home=home, config_path=config, receipt_path=receipt)
    assert decode_config(selected(work)['alpha@release'][1]['config']) == {'user': 'committed-new'}
    print('PASS config failure blocked/recovered/settled/migrated', flush=True)

    root, repo, old, work, home, config, receipt = await setup('review-old-journal-')
    m = await manager(work, home, old)
    await m.terminate_all()
    with sqlite3.connect(work / 'runtime/plugin-reloads.sqlite3') as connection:
        connection.execute('DROP TABLE config_updates')
    with sqlite3.connect(work / 'migrations.sqlite3') as connection:
        connection.execute("DELETE FROM _yoyo_migration WHERE migration_id='20260928_01_plugin_config_updates'")
    ensure_profile(old, old / 'profiles/default.json', workspace=work, plugins_home=home,
                   config_path=config, receipt_path=receipt)
    with sqlite3.connect(work / 'runtime/plugin-reloads.sqlite3') as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE name='config_updates'").fetchone()
        assert connection.execute("PRAGMA integrity_check").fetchone() == ('ok',)
    print('PASS old journal migration', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
