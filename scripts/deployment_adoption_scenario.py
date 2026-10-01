#!/usr/bin/env python3
"""隔离验证历史多次升级的显式归属转换，不接触已有 workspace。"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.deployment_composition_scenario import (
    commit, distribution, git, manager, plugin, selected, snapshot,
)
from scripts.install_plugin_distribution import install_profile, ensure_profile, publish_distribution, _write_receipt
from agent.plugins.artifacts import read_pointers, resolve_pointer
from agent.plugins.distribution_sources import distribution_sources, distribution_migration_sources
from agent.plugins.install import install_git_plugin
from agent.plugins.input_preparation import prepare_plugin_input
from agent.plugins.manifest import set_plugin_enabled
from agent.plugins.selection import PluginSelection, SelectionConflictError
from agent.plugins.static_manifest import load_static_plugin_manifest


async def run():
    """通过真实 bundle、installer、Root 和 Manager 检查转换与重启。"""
    root = Path(tempfile.mkdtemp(prefix="akashic-distribution-adoption-"))
    print("EVIDENCE", root, flush=True)
    repo = root / "source"
    repo.mkdir()
    git(repo, "init", "-q")
    names = ["alpha", "retired", "disabled"]
    first_names = ["alpha", "disabled"]
    for name in first_names:
        plugin(repo, name)
    first = distribution(repo, root / "first", first_names, first_names)
    state = root / "state"
    state.mkdir()
    work, home, config = state / "workspace", state / "plugin-home", state / "config.toml"
    os.environ["HOME"] = str(root / "home")
    Path(os.environ["HOME"]).mkdir()
    os.environ["AKASHIC_PLUGIN_HOME"] = str(home)
    os.environ.pop("AKASHIC_PLUGIN_DISTRIBUTION", None)
    subprocess.run([sys.executable, str(ROOT / "main.py"), "init", "--config", str(config),
                    "--workspace", str(work)], check=True, stdout=subprocess.DEVNULL)
    first_receipt = install_profile(first, first / "profiles/default.json", workspace=work,
                                    plugins_home=home, config_path=config)
    m = await manager(work, home)
    await m.terminate_all()
    # 1. 建立旧安装器允许的多次正式 bundle 升级；最后恢复真实首次回执。
    for name in names:
        plugin(repo, name, "2")
    (repo / "plugins/alpha").rename(repo / "plugins/alpha_legacy")
    (repo / "plugins/retired").rename(repo / "plugins/retired_source")
    historical = distribution(repo, root / "historical", ["alpha_legacy", "retired_source", "disabled"], names)
    install_profile(historical, historical / "profiles/default.json", workspace=work,
                    plugins_home=home, config_path=config)
    selection = PluginSelection(work)
    prepared = []
    for name in names:
        base = home / "cache/release" / name
        artifact = resolve_pointer(base, read_pointers(base).stable)
        identity = load_static_plugin_manifest(artifact)
        prepared.append(prepare_plugin_input(
            {"name": name, "marketplace": "release", "plugin_root": str(artifact),
             "module_path": str(artifact / "plugin.py"), "manifest_digest": identity.identity_digest,
             "source_type": "installed"}, workspace=work, archive=selection.archive,
        ).archive_ref)
    selection.commit(tuple(prepared), expected_ref=selection.read())
    m = await manager(work, home)
    await m.terminate_all()
    receipt = work / "runtime/distribution-install.json"
    _write_receipt(receipt, first_receipt)
    external = root / "external"
    external.mkdir()
    git(external, "init", "-q")
    (external / "plugin.py").write_text("api_version=3\nname='outside'\nversion='1'\nasync def apply(ctx): pass\n")
    commit(external)
    m = await manager(work, home)
    await m.install(source=str(external), marketplace="thirdparty", ref_name="", sparse_paths=[], update_id="outside")
    await m._operation.task
    await m.terminate_all()
    set_plugin_enabled("disabled@release", enabled=False, plugins_home=home)
    selection = PluginSelection(work)
    baseline = selection.read()
    old = selected(work)
    for name in names:
        (work / "plugin-data" / f"{name}-release" / "retained.bin").write_bytes(name.encode())
    cache_before, data_before, receipt_before = snapshot(home / "cache"), snapshot(work / "plugin-data"), receipt.read_bytes()
    for name in ["alpha", "disabled"]:
        plugin(repo, name, "3")
    target = distribution(repo, root / "target", ["alpha", "disabled"], ["alpha", "disabled"])
    target_commit = json.loads((target / "distribution.json").read_text())["source_commit"]
    entries = []
    for name in names:
        plugin_id = f"{name}@release"
        ref, descriptor = old[plugin_id]
        base = home / "cache/release" / name
        pointers = read_pointers(base)
        artifact = resolve_pointer(base, pointers.stable)
        evidence = historical / "distribution.json"
        entries.append({
            "plugin_id": plugin_id, "component_ref": ref, "artifact_pointer": pointers.stable.path,
            "source_revision": git(artifact, "rev-parse", "HEAD"), "code_sha256": descriptor["code"],
            "manifest_digest": load_static_plugin_manifest(artifact).identity_digest,
            "data_dir": descriptor["data_dir"],
            "source_commit": json.loads(evidence.read_text())["source_commit"],
            "source_path": json.loads((artifact / ".akashic-source.json").read_text())["path"],
            "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        })
    plan = root / "plan.json"
    document = {"schema_version": 1, "expected_root_ref": baseline, "targets": []}
    plan.write_text(json.dumps(document))

    def publish(preflight=False):
        return publish_distribution(distribution=target, workspace=work, plugins_home=home,
                                    config_path=config, plan=plan, inputs=root, preflight_only=preflight)

    try:
        publish(True)
    except SelectionConflictError:
        pass
    else:
        raise AssertionError("unapproved history adopted")
    document["distribution_adoption"] = {"distribution_source_commit": target_commit, "entries": entries}
    plan.write_text(json.dumps(document))
    # 2. 预检只读；脏代码和错误 data root 在任何迁移前拒绝。
    before = snapshot(state)
    assert publish(True)["status"] == "preflight_ok"
    assert snapshot(state) == before
    original_data = entries[0]["data_dir"]
    entries[0]["data_dir"] = "plugin-data/outside-thirdparty"
    plan.write_text(json.dumps(document))
    try:
        publish(True)
    except SelectionConflictError:
        pass
    else:
        raise AssertionError("changed data identity accepted")
    entries[0]["data_dir"] = original_data
    plan.write_text(json.dumps(document))
    from scripts.install_plugin_distribution import load_deployment_plan
    proof = load_deployment_plan(plan)[3]
    orphan = selection.archive.save_descriptor(proof)
    assert not distribution_sources(work, home, target).legacy_ids
    assert selection.read() == baseline
    result = publish()
    new_root = selection.read()
    assert result["new_root_ref"] == new_root != baseline
    assert selection.archive.read_descriptor(new_root)["distribution_adoption_ref"] == orphan
    assert selected(work)["outside@thirdparty"][0] == old["outside@thirdparty"][0]
    assert "retired@release" not in selected(work) and "disabled@release" not in selected(work)
    # 3. 普通第二次启动、配置提交与停用均保留同一凭证及退役归属。
    for _ in range(2):
        ensure_profile(target, target / "profiles/default.json", workspace=work, plugins_home=home,
                       config_path=config, receipt_path=receipt)
        assert len(distribution_migration_sources(work, home, target)) == 2
        m = await manager(work, home, target)
        await m.terminate_all()
        assert "retired@release" not in selected(work)
    current = selection.read()
    inherited = selection.commit(tuple(ref for ref, _ in selected(work).values()), expected_ref=current)
    assert selection.archive.read_descriptor(inherited)["distribution_adoption_ref"] == orphan
    assert len(distribution_sources(work, home, target).ignored_installed_roots) == 3
    from scripts.rollback_plugin_install import _selected_code
    assert _selected_code(selection, inherited, "outside@thirdparty")[0] == old["outside@thirdparty"][1]["code"]
    plan.write_text(json.dumps({"schema_version": 1, "expected_root_ref": inherited, "targets": []}))
    assert publish()["new_root_ref"] == inherited
    retired_source = historical / json.loads((historical / "distribution.json").read_text())["plugins"][1]["file"]
    try:
        install_git_plugin(workspace=work, source=str(retired_source), marketplace="release", plugins_home=home)
    except ValueError as error:
        assert "不能接管" in str(error)
    else:
        raise AssertionError("retired data identity reused")
    artifact = home / "cache/release/retired" / entries[1]["artifact_pointer"]
    source = artifact / "plugin.py"
    original = source.read_bytes()
    source.write_bytes(original + b"\n# foreign change\n")
    try:
        distribution_migration_sources(work, home, target)
    except SelectionConflictError:
        pass
    else:
        raise AssertionError("retired cache drift accepted")
    source.write_bytes(original)
    assert snapshot(home / "cache") == cache_before
    assert snapshot(work / "plugin-data") == data_before
    assert receipt.read_bytes() == receipt_before
    (root / "result.json").write_text(json.dumps({"result": "passed", "root": new_root, "adoption": orphan}))
    print("PASS historical-adoption/read-only-preflight/wrong-data/orphan/restart/retirement/disabled/external/preservation/drift", flush=True)


if __name__ == "__main__":
    asyncio.run(run())
