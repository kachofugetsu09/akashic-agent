#!/usr/bin/env python3
"""Isolated explicit external replacement and interrupted-publication acceptance."""
from __future__ import annotations

import asyncio
import shutil
import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.deployment_migration_scenario import setup
from scripts.deployment_composition_scenario import manager, commit, git, selected, snapshot
from scripts.install_plugin_distribution import publish_distribution, _current_artifact
from agent.plugins.selection import PluginSelection
from agent.plugins.python_environment import ENVIRONMENT_FILE, PythonEnvironments, read_environment_refs
from agent.plugins.static_manifest import load_static_plugin_manifest


async def same_commit_environment(previous_python: Path | None) -> None:
    root, _, distribution, workspace, home, config, receipt = await setup('deployment-same-commit-')
    runtime = await manager(workspace, home, distribution)
    external = root / 'external'
    external.mkdir()
    git(external, 'init', '-q')
    (external / 'plugin.py').write_text("api_version=3\nname='outside'\nversion='1'\nasync def apply(ctx): pass\n")
    (external / 'requirements.txt').write_text('')
    revision = commit(external)
    await runtime.install(source=str(external), marketplace='thirdparty', ref_name='',
                          sparse_paths=[], update_id='outside-v1')
    await runtime._operation.task
    await runtime.terminate_all()
    selection = PluginSelection(workspace)
    original = selection.read()
    original_input, descriptor = selected(workspace)['outside@thirdparty']
    artifact, _, _ = _current_artifact(workspace=workspace, plugins_home=home, plugin_id='outside@thirdparty')
    identity = load_static_plugin_manifest(artifact)
    current_refs = read_environment_refs(artifact, identity)
    assert current_refs
    old_refs = {}
    old_tag = 'cpython-311'
    if previous_python is not None:
        old_tag = subprocess.check_output([
            str(previous_python), '-c', 'import sys; print(sys.implementation.cache_tag)',
        ], text=True).strip()
        assert old_tag != sys.implementation.cache_tag, 'Use a different Python minor'
    # Without --previous-python this models only the old environment identity.
    # With it, create and later compare real interpreters from two Python minors.
    for name, ref in current_refs.items():
        owner = PythonEnvironments(workspace)
        original_root = owner.open(ref)
        record = json.loads((original_root / 'environment.json').read_text())
        location = uuid4().hex + uuid4().hex
        old_root = owner.path / location
        if previous_python is not None:
            environment = old_root / name / '.venv'
            subprocess.run([str(previous_python), '-m', 'venv', '--without-pip', '--copies', str(environment)],
                           check=True)
        else:
            shutil.copytree(original_root, old_root, symlinks=True)
        record['input'] = {**record['input'], 'base': {'executable': str(previous_python or '/previous/python3.11')}}
        (old_root / 'environment.json').write_text(json.dumps(record))
        old_refs[name] = location
    old_environment_bytes = json.dumps(old_refs).encode()
    (artifact / ENVIRONMENT_FILE).write_bytes(old_environment_bytes)
    old_descriptor = dict(descriptor)
    old_descriptor['python_environments'] = old_refs
    old_descriptor['runtime'] = {**descriptor['runtime'], 'python_tag': old_tag}
    old_input = selection.prepare(old_descriptor)
    components = selection.components(original)
    baseline = selection.commit(tuple(old_input if ref == original_input else ref for ref in components),
                                expected_ref=original)
    bundle = root / 'external.bundle'
    git(external, 'bundle', 'create', str(bundle), 'HEAD')
    plan = root / 'plan.json'
    plan.write_text(json.dumps({'schema_version': 1, 'expected_root_ref': baseline, 'targets': [{
        'plugin_id': 'outside@thirdparty', 'bundle_relative_path': bundle.name,
        'bundle_sha256': hashlib.sha256(bundle.read_bytes()).hexdigest(), 'target_commit': revision,
    }]}))
    result = publish_distribution(distribution=distribution, workspace=workspace, plugins_home=home,
                                  config_path=config, plan=plan, inputs=root)
    assert result['status'] == 'selected_not_started'
    new_input = selected(workspace)['outside@thirdparty'][1]
    assert dict(new_input['python_environments']) == current_refs
    for name, ref in current_refs.items():
        interpreter = PythonEnvironments(workspace).open(ref) / name / '.venv/bin/python'
        actual_tag = subprocess.check_output([
            str(interpreter), '-c', 'import sys; print(sys.implementation.cache_tag)',
        ], text=True).strip()
        assert actual_tag == new_input['runtime']['python_tag']
    assert (artifact / ENVIRONMENT_FILE).read_bytes() == old_environment_bytes
    new_artifact, _, _ = _current_artifact(workspace=workspace, plugins_home=home, plugin_id='outside@thirdparty')
    assert new_artifact != artifact
    runtime = await manager(workspace, home, distribution)
    assert runtime.generation('outside@thirdparty').instance.version == '1'
    await runtime.terminate_all()
    print('PASS same-commit reinstall replaces stale environment refs and retains old artifact', flush=True)


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
    old_input = selection.prepare(old_descriptor)
    components = selection.components(original)
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
    assert selection.read_input(original_input) == descriptor
    assert selected(workspace)['outside@thirdparty'][1]['runtime']['python_tag'] == sys.implementation.cache_tag
    runtime = await manager(workspace, home, distribution)
    assert runtime.generation('outside@thirdparty').instance.version == '2'
    await runtime.terminate_all()
    print('PASS identical retry after external install; v2 ready; old inputs/data/receipt retained', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-python', type=Path, help='Optional installed interpreter from a different Python minor')
    arguments = parser.parse_args()
    asyncio.run(main())
    asyncio.run(same_commit_environment(arguments.previous_python))
