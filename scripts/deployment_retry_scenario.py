#!/usr/bin/env python3
"""Isolated explicit external replacement and interrupted-publication acceptance."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.deployment_migration_scenario import setup
from scripts.deployment_composition_scenario import manager, commit, git, selected, snapshot
from scripts.install_plugin_distribution import publish_distribution
from agent.plugins.selection import PluginSelection


async def main() -> None:
    root, _, distribution, workspace, home, config, receipt = await setup('deployment-retry-')
    runtime = await manager(workspace, home, distribution)
    external = root / 'external'
    external.mkdir()
    git(external, 'init', '-q')
    (external / 'plugin.py').write_text("api_version=3\nname='outside'\nversion='1'\nasync def apply(ctx): pass\n")
    commit(external)
    await runtime.install(source=str(external), marketplace='thirdparty', ref_name='',
                          sparse_paths=[], update_id='outside-v1')
    await runtime._operation.task
    await runtime.terminate_all()
    selection = PluginSelection(workspace)
    original = selection.read()
    original_input, descriptor = selected(workspace)['outside@thirdparty']
    # The selection format is portable, but an unchanged old interpreter is not.
    # Represent an input from the preceding Python minor without running it here.
    old_descriptor = dict(descriptor)
    old_descriptor['runtime'] = {**descriptor['runtime'], 'python_tag': 'cpython-311'}
    old_input = selection.archive.save_descriptor(old_descriptor)
    components = selection.archive.read_descriptor(original)['components']
    baseline = selection.commit(tuple(old_input if ref == original_input else ref for ref in components),
                                expected_ref=original)
    data_before = snapshot(workspace / 'plugin-data')
    receipt_before = receipt.read_bytes()
    (external / 'plugin.py').write_text("api_version=3\nname='outside'\nversion='2'\nasync def apply(ctx): pass\n")
    revision = commit(external)
    bundle = root / 'external.bundle'
    git(external, 'bundle', 'create', str(bundle), 'HEAD')
    plan = root / 'plan.json'
    options = dict(distribution=distribution, workspace=workspace, plugins_home=home,
                   config_path=config, plan=plan, inputs=root)
    document = {'schema_version': 1, 'expected_root_ref': baseline, 'targets': []}
    plan.write_text(json.dumps(document))
    try:
        publish_distribution(**options, preflight_only=True)
    except RuntimeError as error:
        assert 'explicit reinstall required' in str(error), error
    else:
        raise AssertionError('incompatible preserved runtime accepted')
    document['targets'] = [{'plugin_id': 'outside@thirdparty',
                            'bundle_relative_path': bundle.name,
                            'bundle_sha256': hashlib.sha256(bundle.read_bytes()).hexdigest(),
                            'target_commit': revision}]
    plan.write_text(json.dumps(document))
    assert publish_distribution(**options, preflight_only=True)['status'] == 'preflight_ok'
    print('PASS explicit replacement permits old runtime metadata', flush=True)
    with patch('scripts.install_plugin_distribution._prepare_distribution_inputs',
               side_effect=OSError('injected interruption after external install')):
        try:
            publish_distribution(**options)
        except OSError as error:
            assert 'injected interruption' in str(error), error
        else:
            raise AssertionError('fault was not exercised')
    assert selection.read() == baseline
    assert selected(workspace)['outside@thirdparty'][0] == old_input
    assert publish_distribution(**options, preflight_only=True)['status'] == 'preflight_ok'
    result = publish_distribution(**options)
    assert result['status'] == 'selected_not_started'
    assert result['new_root_ref'] != baseline
    assert snapshot(workspace / 'plugin-data') == data_before
    assert receipt.read_bytes() == receipt_before
    assert selection.archive.read_descriptor(original_input) == descriptor
    assert selected(workspace)['outside@thirdparty'][1]['runtime']['python_tag'] == sys.implementation.cache_tag
    runtime = await manager(workspace, home, distribution)
    assert runtime.generation('outside@thirdparty').instance.version == '2'
    await runtime.terminate_all()
    print('PASS identical retry after external install; v2 ready; old inputs/data/receipt retained', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
