#!/usr/bin/env python3
"""Manual isolated deployment acceptance; no provider requests or existing state.

Run with the repository environment. The output directory is newly created and
retained with full evidence. --with-wheels also resolves one public test package.
"""

import argparse
import asyncio, hashlib, json, os, shutil, sqlite3, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path)
parser.add_argument("--with-wheels", action="store_true")
args = parser.parse_args()

from scripts.build_plugin_distribution import _bundle_plugin
from scripts.distribution_runtime import prepare_wheels
from session.log import MessageLog
from session.message import Input
from scripts.install_plugin_distribution import (
    install_profile,
    ensure_profile,
    _write_receipt,
    publish_distribution,
)
from agent.plugins.install import install_git_plugin
from agent.migrations.runner import MigrationRunner
from agent.plugins.manager import PluginManager
from agent.plugins.manifest import (
    load_plugin_manifest,
    set_plugin_enabled,
    workspace_plugin_data_dir,
)
from agent.plugins.distribution_sources import distribution_sources
from agent.plugins.selection import PluginSelection
from bus.event_bus import EventBus


def git(repo, *args):
    return (
        subprocess.check_output(
            ["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL
        )
        .decode()
        .strip()
    )


def commit(repo):
    git(repo, "add", ".")
    git(
        repo,
        "-c",
        "user.name=scenario",
        "-c",
        "user.email=scenario@example.invalid",
        "-c",
        "commit.gpgSign=false",
        "commit",
        "-qm",
        "scenario",
    )
    return git(repo, "rev-parse", "HEAD")


def plugin(repo, name, version="1"):
    d = repo / "plugins" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "plugin.py").write_text(
        f"api_version=3\nname={name!r}\nversion={version!r}\nasync def apply(ctx):\n    pass\n"
    )
    if name == "alpha":
        (d / "requirements.txt").write_text("")
    return d


def distribution(repo, out, names, defaults):
    sha = commit(repo)
    out.mkdir()
    stamp = git(repo, "show", "-s", "--format=%cI", sha)
    rows = [_bundle_plugin(repo, sha, stamp, "plugins/" + n, out, set()) for n in names]
    (out / "core.tar").write_bytes(b"isolated fixture core identity")
    profile = {
        "schema_version": 1,
        "name": "scenario",
        "marketplace": "release",
        "description": "isolated",
        "initialization": {
            "plugin_configs": [{"owner": "alpha", "config": {"user": "initial"}}]
        },
        "plugins": [
            {"name": n, "depends_on": [], "reason": "isolated"} for n in defaults
        ],
    }
    (out / "profiles").mkdir()
    (out / "profiles/default.json").write_text(json.dumps(profile))
    report = {
        "schema_version": 2,
        "source_commit": sha,
        "source_tree": git(repo, "rev-parse", "HEAD^{tree}"),
        "core": {
            "file": "core.tar",
            "sha256": hashlib.sha256((out / "core.tar").read_bytes()).hexdigest(),
        },
        "plugins": rows,
        "profiles": [
            {
                "path": "profiles/default.json",
                "sha256": hashlib.sha256(
                    (out / "profiles/default.json").read_bytes()
                ).hexdigest(),
            }
        ],
    }
    (out / "distribution.json").write_text(json.dumps(report))
    return out


def snapshot(path):
    return {
        str(p.relative_to(path)): p.read_bytes()
        for p in path.rglob("*")
        if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts
    }


def selected(workspace):
    s = PluginSelection(workspace)
    r = s.read()
    return {
        s.archive.read_descriptor(c)["plugin_id"]: (c, s.archive.read_descriptor(c))
        for c in s.archive.read_descriptor(r)["components"]
    }


async def manager(workspace, home, dist=None):
    ds = distribution_sources(workspace, home, dist) if dist else None
    m = PluginManager(
        [],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=home / "cache",
        distribution_sources=ds.sources if ds else (),
        ignored_installed_roots=ds.ignored_installed_roots if ds else frozenset(),
    )
    await m.load_all()
    return m


async def run():
    if args.output is None:
        root = Path(tempfile.mkdtemp(prefix="akashic-builtin-transitions-"))
    else:
        root = args.output.resolve()
        root.mkdir(parents=True, exist_ok=False)
    identity = {
        "commit": git(ROOT, "rev-parse", "HEAD"),
        "tree": git(ROOT, "rev-parse", "HEAD^{tree}"),
        "dirty": bool(git(ROOT, "status", "--porcelain")),
    }
    print("EVIDENCE", root, identity, flush=True)
    repo = root / "fixture-source"
    repo.mkdir()
    git(repo, "init", "-q")
    oldnames = ["alpha", "gone", "disabled", "optional"]
    for n in oldnames:
        plugin(repo, n)
    old = distribution(repo, root / "old", oldnames, oldnames)
    state = root / "state"
    state.mkdir()
    work = state / "workspace"
    home = state / "plugin-home"
    config = state / "config.toml"
    receipt = work / "runtime/distribution-install.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "init",
            "--config",
            str(config),
            "--workspace",
            str(work),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    r = install_profile(
        old,
        old / "profiles/default.json",
        workspace=work,
        plugins_home=home,
        config_path=config,
    )
    _write_receipt(receipt, r)
    MigrationRunner(
        repo_root=ROOT,
        config_path=config,
        workspace=work,
        installed_cache_root=home / "cache",
    ).run()
    # Real legacy Manager creates the durable old selection.
    m = await manager(work, home)
    await m.apply_config_input(
        "alpha@release",
        "scenario-config",
        m.read_config_input("alpha@release")["input_ref"],
        {"user": "kept", "nested": {"x": [1, 2]}},
    )
    await m._operation.task
    await m.terminate_all()
    set_plugin_enabled("disabled@release", enabled=False, plugins_home=home)
    # Preserve bytes for all business state, cache and first receipt.
    for n in oldnames:
        (workspace_plugin_data_dir(work, n, "release") / "sentinel.bin").write_bytes(
            ("keep-" + n).encode()
        )
    log = MessageLog(work / "sessions.db")
    log.writer(
        "retained-session",
        author="user",
        source="scenario",
        body_types=(Input,),
        content={},
    ).append("retained-message", Input(()))
    log.close()
    with sqlite3.connect(work / "sessions.db") as db:
        messages_before = "\n".join(db.iterdump())
    messages_bytes = (work / "sessions.db").read_bytes()
    (work / "opaque-business.db").write_bytes(
        b"not-a-database: do not inspect or rewrite"
    )
    cache_before = snapshot(home / "cache")
    data_before = snapshot(work / "plugin-data")
    receipt_before = receipt.read_bytes()
    config_before = config.read_bytes()
    ext = root / "external"
    ext.mkdir()
    git(ext, "init", "-q")
    (ext / "plugin.py").write_text(
        "api_version=3\nname='outside'\nversion='1'\nasync def apply(ctx):\n    pass\n"
    )
    commit(ext)
    m = await manager(work, home)
    await m.install(
        source=str(ext),
        marketplace="thirdparty",
        ref_name="",
        sparse_paths=[],
        update_id="external-install",
    )
    await m._operation.task
    await m.terminate_all()
    external_before = selected(work)["outside@thirdparty"][0]
    plugin(repo, "alpha", "2")
    plugin(repo, "disabled", "2")
    plugin(repo, "optional", "2")
    plugin(repo, "newcomer", "1")
    shutil.rmtree(repo / "plugins/gone")
    if args.with_wheels:
        (repo / "plugins/alpha/requirements.txt").write_text("packaging==25.0\n")
    new = distribution(
        repo,
        root / "new",
        ["alpha", "disabled", "optional", "newcomer"],
        ["alpha", "disabled", "newcomer"],
    )
    if args.with_wheels:
        prepare_wheels(new)
    result = ensure_profile(
        new,
        new / "profiles/default.json",
        workspace=work,
        plugins_home=home,
        config_path=config,
        receipt_path=receipt,
    )
    now = selected(work)
    assert set(now) == {
        "alpha@release",
        "optional@release",
        "newcomer@release",
        "outside@thirdparty",
    }, set(now)
    assert now["outside@thirdparty"][0] == external_before
    assert (
        receipt.read_bytes() == receipt_before and config.read_bytes() == config_before
    )
    assert snapshot(work / "plugin-data") == {
        **data_before,
        **{
            k: v
            for k, v in snapshot(work / "plugin-data").items()
            if k.startswith("outside-thirdparty/")
        },
    }
    for key, value in cache_before.items():
        assert snapshot(home / "cache")[key] == value, key
    assert (
        now["alpha@release"][1]["distribution_source"]
        == json.loads((new / "distribution.json").read_text())["source_commit"]
    )
    stable = PluginSelection(work).read()
    ensure_profile(
        new,
        new / "profiles/default.json",
        workspace=work,
        plugins_home=home,
        config_path=config,
        receipt_path=receipt,
    )
    assert PluginSelection(work).read() == stable
    m = await manager(work, home, new)
    assert m.generation("alpha@release").instance.version == "2"
    assert m.generation("gone@release") is None
    assert m.generation("alpha@release").config_projection == {
        "user": "kept",
        "nested": {"x": [1, 2]},
    }
    assert m.generation("optional@release").instance.version == "2"
    # Existing enable is a durable choice applied by normal deployment/startup.
    set_plugin_enabled("disabled@release", enabled=True, plugins_home=home)
    await m.terminate_all()
    ensure_profile(
        new,
        new / "profiles/default.json",
        workspace=work,
        plugins_home=home,
        config_path=config,
        receipt_path=receipt,
    )
    m = await manager(work, home, new)
    assert m.generation("disabled@release").instance.version == "2"
    set_plugin_enabled("alpha@release", enabled=False, plugins_home=home)
    await m.reconcile_changed()
    assert m.generation("alpha@release") is None
    set_plugin_enabled("alpha@release", enabled=True, plugins_home=home)
    await m.terminate_all()
    ensure_profile(
        new,
        new / "profiles/default.json",
        workspace=work,
        plugins_home=home,
        config_path=config,
        receipt_path=receipt,
    )
    m = await manager(work, home, new)
    assert m.generation("alpha@release").instance.version == "2"
    await m.apply_config_input(
        "alpha@release",
        "new-config",
        m.read_config_input("alpha@release")["input_ref"],
        {"user": "kept", "runtime": "image"},
    )
    await m._operation.task
    await m.reconcile_changed()
    assert m.generation("alpha@release").config_projection["runtime"] == "image"
    command = m._resolve_runtime_command(
        m.generation("alpha@release"),
        ("python", "-c", "import sys; print(sys.version_info.major)"),
        ".",
    )
    assert subprocess.check_output(command, text=True).strip() == "3"
    if args.with_wheels:
        command = m._resolve_runtime_command(
            m.generation("alpha@release"),
            ("python", "-c", "import packaging; print(packaging.__version__)"),
            ".",
        )
        assert subprocess.check_output(command, text=True).strip() == "25.0"
    ref = m.generation("alpha@release").archive_ref
    assert m._archive.read_descriptor(ref)["python_environments"]
    await m.uninstall("newcomer@release")
    await m._operation.task
    assert load_plugin_manifest(home)["newcomer@release"] is False
    await m.terminate_all()
    ensure_profile(
        new,
        new / "profiles/default.json",
        workspace=work,
        plugins_home=home,
        config_path=config,
        receipt_path=receipt,
    )
    assert "newcomer@release" not in selected(work)
    assert "gone@release" not in selected(work)
    # Formal publish defaults to exactly the same composition, preflight is read-only.
    plan = root / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "expected_root_ref": PluginSelection(work).read(),
                "targets": [],
                "migrations": [],
            }
        )
    )
    state_before = snapshot(state)
    pre = publish_distribution(
        distribution=new,
        workspace=work,
        plugins_home=home,
        config_path=config,
        plan=plan,
        inputs=root,
        preflight_only=True,
    )
    assert snapshot(state) == state_before
    pub = publish_distribution(
        distribution=new,
        workspace=work,
        plugins_home=home,
        config_path=config,
        plan=plan,
        inputs=root,
    )
    assert pub["new_root_ref"] == json.loads(plan.read_text())["expected_root_ref"]
    # Corrupt new image bytes are rejected before any selection change.
    before = PluginSelection(work).read()
    entry = new / "sources/alpha/plugin.py"
    original = entry.read_bytes()
    entry.write_text("def invalid(:\n")
    try:
        ensure_profile(
            new,
            new / "profiles/default.json",
            workspace=work,
            plugins_home=home,
            config_path=config,
            receipt_path=receipt,
        )
    except Exception:
        pass
    else:
        raise AssertionError("bad source accepted")
    assert PluginSelection(work).read() == before
    entry.write_bytes(original)
    with sqlite3.connect(work / "sessions.db") as db:
        assert "\n".join(db.iterdump()) == messages_before
    assert (work / "sessions.db").read_bytes() == messages_bytes
    assert (
        work / "opaque-business.db"
    ).read_bytes() == b"not-a-database: do not inspect or rewrite"
    print(
        "PASS update/removal/optional/default/disabled/config/external/byte-preservation/restart/enable/environment/uninstall/preflight/publish/failure",
        flush=True,
    )
    (root / "result.json").write_text(
        json.dumps(
            {
                "result": "passed",
                "candidate_checkout": str(ROOT),
                "source_identity": identity,
                "with_wheels": args.with_wheels,
                "scenarios": 16,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(run())
