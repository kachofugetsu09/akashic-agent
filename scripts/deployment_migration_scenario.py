"""在全新临时安装中验证迁移的数据归属、配置恢复和旧账本升级。"""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.deployment_composition_scenario import ROOT, plugin, git, commit, distribution, ensure_bundle, manager, migration, selected, snapshot
from agent.plugins.manifest import workspace_plugin_data_dir
from agent.plugins.bundles import set_plugin_choice
from agent.plugin_composition.config_input import config_bytes, load_config
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
    ensure_bundle(old, old / 'bundles/base.toml', workspace=work, plugins_home=home, config_path=config, receipt_path=receipt)
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
        set_plugin_choice(work, 'alpha@release', enabled=not disabled)
        before = PluginSelection(work).read()
        for startup in [False, True]:
            os.environ['AKASHIC_PLUGIN_DISTRIBUTION'] = str(new)
            try:
                if startup:
                    MigrationRunner(repo_root=ROOT, config_path=config, workspace=work, startup_selection=True).run()
                else:
                    ensure_bundle(new, new / 'bundles/base.toml', workspace=work, plugins_home=home, config_path=config, receipt_path=receipt)
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
    m._publish_config_update = fail
    await m.apply_config_input('alpha@release', 'failed-file-publish', m.read_config_input('alpha@release')['input_ref'], {'user': 'committed-new'})
    try:
        await m._operation.task
    except Exception:
        pass
    await m.terminate_all()
    assert selected(work)['alpha@release'][1]['config_revision'] == hashlib.sha256(config_bytes({'user': 'committed-new'})).hexdigest()
    before = PluginSelection(work).read()
    before_data = snapshot(work / 'plugin-data')
    # Every product launcher uses this installer before starting the recovery owner.
    restarted = subprocess.run([
        sys.executable, str(ROOT / 'scripts/install_plugin_distribution.py'),
        '--distribution', str(old), '--bundle', str(old / 'bundles/base.toml'),
        '--workspace', str(work), '--plugins-home', str(home), '--config', str(config),
        '--ensure-bundle', '--receipt', str(receipt),
    ], check=True, capture_output=True, text=True)
    assert json.loads(restarted.stdout)['recovery_pending'] is True
    assert PluginSelection(work).read() == before
    assert snapshot(work / 'plugin-data') == before_data
    plugin(repo, 'alpha', '2')
    changed = distribution(repo, root / 'changed-without-migrations', ['alpha'], ['alpha'])
    try:
        ensure_bundle(changed, changed / 'bundles/base.toml', workspace=work, plugins_home=home,
                       config_path=config, receipt_path=receipt)
    except RuntimeError as error:
        assert '配置提交尚待原 runtime 恢复' in str(error), error
    else:
        raise AssertionError('changed code published before config recovery')
    assert PluginSelection(work).read() == before
    assert snapshot(work / 'plugin-data') == before_data
    migration(repo, 'alpha', 'review_config_upgrade', 'step("CREATE TABLE review_config_done (value INTEGER)")\n')
    new = distribution(repo, root / 'new', ['alpha'], ['alpha'])
    try:
        ensure_bundle(new, new / 'bundles/base.toml', workspace=work, plugins_home=home,
                       config_path=config, receipt_path=receipt)
    except RuntimeError as error:
        assert '配置提交尚待原 runtime 恢复' in str(error), error
    else:
        raise AssertionError('new distribution published before config recovery')
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
    ensure_bundle(new, new / 'bundles/base.toml', workspace=work, plugins_home=home, config_path=config, receipt_path=receipt)
    assert selected(work)['alpha@release'][1]['config_revision'] == hashlib.sha256(config_bytes({'user': 'committed-new'})).hexdigest()
    assert load_config(work / 'plugin-data/alpha-release')[0] == {'user': 'committed-new'}
    print('PASS config failure blocked/recovered/settled/migrated', flush=True)

    root, repo, old, work, home, config, receipt = await setup('review-old-journal-')
    m = await manager(work, home, old)
    await m.terminate_all()
    with sqlite3.connect(work / 'runtime/plugin-reloads.sqlite3') as connection:
        connection.execute('DROP TABLE config_updates')
    with sqlite3.connect(work / 'migrations.sqlite3') as connection:
        connection.execute("DELETE FROM _yoyo_migration WHERE migration_id='20260928_01_plugin_config_updates'")
    ensure_bundle(old, old / 'bundles/base.toml', workspace=work, plugins_home=home,
                   config_path=config, receipt_path=receipt)
    with sqlite3.connect(work / 'runtime/plugin-reloads.sqlite3') as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE name='config_updates'").fetchone()
        assert connection.execute("PRAGMA integrity_check").fetchone() == ('ok',)
    print('PASS old journal migration', flush=True)


async def check_contract_restart():
    """实际安装中升级旧 journal，并核对备份和原记录保留。"""
    root, repo, old, work, home, config, receipt = await setup('review-contract-journal-')
    m = await manager(work, home, old)
    await m.terminate_all()
    path = work / 'runtime/plugin-reloads.sqlite3'
    # 构造前一发行版的真实形状；现有 journal、输入和安装记录均保留。
    with sqlite3.connect(path) as connection:
        columns = [row[1] for row in connection.execute('PRAGMA table_info(plugin_updates)')]
        fields = ','.join(columns)
        before = connection.execute('SELECT rowid,' + fields + ' FROM plugin_updates ORDER BY rowid').fetchall()
        assert before
        other = {table: connection.execute('SELECT * FROM ' + table).fetchall()
                 for table in ('reload_transactions', 'reload_events', 'config_updates')}
        connection.execute('DROP TABLE plugin_contract_restarts')
    with sqlite3.connect(work / 'migrations.sqlite3') as connection:
        connection.execute("DELETE FROM _yoyo_migration WHERE migration_id='20261011_01_plugin_contract_restart'")
    outcome = MigrationRunner(repo_root=ROOT, config_path=config, workspace=work, startup_selection=True).run()
    assert '20261011_01_plugin_contract_restart' in outcome.migrations
    backups = list(path.parent.glob(path.name + '.before-contract-restart.*.bak'))
    assert len(backups) == 1
    for checked in (path, backups[0]):
        with sqlite3.connect(checked) as connection:
            assert connection.execute('SELECT rowid,' + fields + ' FROM plugin_updates ORDER BY rowid').fetchall() == before
            for table, rows in other.items():
                assert connection.execute('SELECT * FROM ' + table).fetchall() == rows
            assert connection.execute('PRAGMA integrity_check').fetchone() == ('ok',)
    with sqlite3.connect(path) as connection:
        assert not connection.execute('SELECT 1 FROM plugin_contract_restarts').fetchall()
    assert MigrationRunner(repo_root=ROOT, config_path=config, workspace=work, startup_selection=True).run().state == 'current'
    print('PASS contract restart migration retained rows/backup/reapply', flush=True)


async def check_web_providers():
    """Publish an old explicit Web choice through the real preflight and migration path."""
    import tomlkit
    from agent.plugins.bundles import load_bundle, plugin_choices
    from scripts.install_plugin_distribution import publish_distribution

    for mode in ('enabled', 'disabled', 'explicit-provider'):
        root, repo, old, work, home, config, receipt = await setup('review-web-split-')
        # 1. The previous default bundle omitted Web tools; enabling them used a plugin-ID row.
        plugin(repo, 'standard_web')
        old = distribution(repo, root / 'old-web', ['alpha', 'standard_web'], ['alpha'])
        set_plugin_choice(work, 'standard_web@release', enabled=mode != 'disabled', distribution=old)
        ensure_bundle(old, old / 'bundles/base.toml', workspace=work, plugins_home=home,
                      config_path=config, receipt_path=receipt)
        patch = work / 'bundle.patch.toml'
        original_row = tomlkit.parse(patch.read_text())['rows']['standard_web@release'].unwrap()
        if mode == 'explicit-provider':
            set_plugin_choice(work, 'web_search_exa@release', enabled=False)
        # This fixture represents the ledger just before the new upgrade step existed.
        with sqlite3.connect(work / 'migrations.sqlite3') as connection:
            connection.execute("DELETE FROM _yoyo_migration WHERE migration_id='20261019_06_web_provider_split'")
        for name in ('web_search_exa', 'web_fetch_http'):
            plugin(repo, name)
        target = distribution(repo, root / 'new-web',
                              ['alpha', 'standard_web', 'web_search_exa', 'web_fetch_http'], ['alpha'])
        # Use the product's actual defaults, including each row ID and disabled state.
        bundle = target / 'bundles/base.toml'
        document = tomlkit.parse(bundle.read_text())
        for row in load_bundle(ROOT / 'bundles/base.toml'):
            if row.plugin in {'standard_web@release', 'web_search_exa@release', 'web_fetch_http@release'}:
                document['rows'][row.id] = {'plugin': row.plugin, 'disabled': row.disabled}
        bundle.write_text(tomlkit.dumps(document))
        report_path = target / 'distribution.json'
        report = json.loads(report_path.read_text())
        report['bundles'][0]['sha256'] = hashlib.sha256(bundle.read_bytes()).hexdigest()
        report_path.write_text(json.dumps(report))
        plan = root / 'plan.json'
        plan.write_text(json.dumps({'schema_version': 1, 'expected_root_ref': PluginSelection(work).read(), 'targets': []}))
        options = dict(distribution=target, workspace=work, plugins_home=home,
                       config_path=config, plan=plan, inputs=root)
        # 2. Online preflight must preserve choices; actual publication runs the upgrade.
        before = snapshot(work)
        assert publish_distribution(**options, preflight_only=True)['status'] == 'preflight_ok'
        assert snapshot(work) == before
        assert publish_distribution(**options)['status'] == 'selected_not_started'
        choices = plugin_choices(work, target)
        expected = mode != 'disabled'
        assert choices['standard_web@release'] == expected
        assert choices['web_fetch_http@release'] == expected
        assert choices['web_search_exa@release'] == (expected and mode != 'explicit-provider')
        assert tomlkit.parse(patch.read_text())['rows']['standard_web@release'].unwrap() == original_row
        runtime = await manager(work, home, target)
        try:
            for name in ('standard_web', 'web_search_exa', 'web_fetch_http'):
                assert (name + '@release' in selected(work)) == choices[name + '@release']
        finally:
            await runtime.terminate_all()
        print('PASS Web provider upgrade', mode, flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ["--web-providers"]:
        asyncio.run(check_web_providers())
    else:
        asyncio.run(check_contract_restart() if sys.argv[1:] == ["--contract-restart"] else main())
