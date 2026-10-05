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
from agent.plugin_composition.config_input import load_config
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


def migration(repo, name, migration_id, body):
    """为真实插件制品写入一个有摘要的 Yoyo bundle。"""
    plugin_root = repo / "plugins" / name
    package_name = f"scenario_{name}_migrations"
    package = plugin_root / package_name
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text("")
    code = "from yoyo import step\n" + body
    (package / f"{migration_id}.py").write_text(code)
    files = sorted(package.glob("*.py"))
    catalog = (f'schema_version = 1\nbundle_id = "{name}"\nversion = "1"\n'
               f'migration_root = "{package_name}"\npackage_name = "{package_name}"\n')
    for file in files:
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        catalog += f'\n[[files]]\npath = "{file.name}"\nsha256 = "{digest}"\n'
    for file in files:
        if file.name == "__init__.py":
            continue
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        catalog += (f'\n[[migrations]]\nid = "{file.stem}"\npath = "{file.name}"\n'
                    f'depends = []\ntransactional = true\nsha256 = "{digest}"\n')
    (plugin_root / "migration.catalog.toml").write_text(catalog)


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


def seed_and_check_preflight(core, root, distribution, work, home, config):
    """旧 Core 生成真实选择，新发布 CLI 接受它并拒绝归档损坏。"""
    seed = """import asyncio, sys
from pathlib import Path
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
async def run():
    m = PluginManager([], event_bus=EventBus(), workspace=Path(sys.argv[1]),
                      installed_cache_root=Path(sys.argv[2]) / 'cache')
    await m.load_all()
    await m.terminate_all()
asyncio.run(run())
"""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(core), str(core / "sdk/python/src")))}
    with (root / "seed.log").open("w") as output:
        subprocess.run([sys.executable, "-c", seed, str(work), str(home)], cwd=root,
                       env=env, stdout=output, stderr=output, check=True, timeout=60)
    plan = root / "seed-plan.json"
    plan.write_text(json.dumps({"schema_version": 1,
                               "expected_root_ref": PluginSelection(work).read(), "targets": []}))
    command = [sys.executable, str(ROOT / "scripts/install_plugin_distribution.py"),
               "--publish", "--preflight-only", "--distribution", str(distribution),
               "--profile", str(distribution / "profiles/default.json"),
               "--workspace", str(work), "--plugins-home", str(home),
               "--config", str(config), "--plan", str(plan), "--inputs", str(root)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "sdk/python/src")))}
    before = snapshot(work.parent)
    with (root / "seed-preflight.log").open("w") as output:
        subprocess.run(command, env=env, stdout=output, stderr=output, check=True, timeout=60)
    assert snapshot(work.parent) == before
    selection = PluginSelection(work)
    descriptor = selected(work)["alpha@release"][1]
    code = selection.archive.open(descriptor["code"])
    # 运行代码和来源证据的损坏都必须被同一个完整归档校验拒绝。
    for filename in ("plugin.py", ".akashic-source.json"):
        path = code / filename
        content = path.read_bytes()
        mode = path.stat().st_mode & 0o777
        path.chmod(mode | 0o200)
        try:
            path.write_bytes(content + b"\n# altered archive\n")
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
            (root / f"corrupt-{filename.lstrip('.')}.log").write_text(result.stdout + result.stderr)
            assert result.returncode != 0 and "selected 源码身份不一致: alpha@release" in result.stderr
        finally:
            path.write_bytes(content)
            path.chmod(mode)
        assert snapshot(work.parent) == before


async def run(args):
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
    (root / "home-config").mkdir()
    os.environ["HOME"] = str(root / "home-config")
    os.environ["AKASHIC_PLUGIN_HOME"] = str(home)
    os.environ.pop("AKASHIC_PLUGIN_DISTRIBUTION", None)
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
    ).run()
    if args.seed_core:
        seed_and_check_preflight(args.seed_core.resolve(), root, old, work, home, config)
    # Real Manager reads the durable selection and applies a config update.
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
    (ext / "migration.catalog.toml").write_text("external format: not an Akashic Yoyo catalog")
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
    ref = m.generation("alpha@release").input_ref
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
    # Core 更新使用当前镜像路径；配置、依赖和外置插件选择保持不变。
    refs_before = selected(work)
    environments_before = snapshot(work / "runtime/plugin-python-environments")
    (repo / "core-only.txt").write_text("new Core, unchanged plugins\n")
    core_only = distribution(repo, root / "core-only", ["alpha", "disabled", "optional", "newcomer"],
                             ["alpha", "disabled", "newcomer"])
    if args.with_wheels:
        prepare_wheels(core_only)
    ensure_profile(core_only, core_only / "profiles/default.json", workspace=work, plugins_home=home,
                   config_path=config, receipt_path=receipt)
    current_inputs = selected(work)
    assert current_inputs["outside@thirdparty"] == refs_before["outside@thirdparty"]
    for plugin_id, (_, descriptor) in current_inputs.items():
        if descriptor["source_type"] == "builtin":
            assert Path(descriptor["code"]) == core_only / "sources" / plugin_id.split("@")[0]
            assert descriptor["config_revision"] == refs_before[plugin_id][1]["config_revision"]
            assert descriptor["python_environments"] == refs_before[plugin_id][1]["python_environments"]
    assert snapshot(work / "runtime/plugin-python-environments") == environments_before
    # Changed plugin source gets a new input without rebuilding unchanged dependencies.
    requirements_before = (repo / "plugins/alpha/requirements.txt").read_bytes()
    plugin(repo, "alpha", "3")
    (repo / "plugins/alpha/requirements.txt").write_bytes(requirements_before)
    code_changed = distribution(repo, root / "code-changed", ["alpha", "disabled", "optional", "newcomer"],
                                ["alpha", "disabled", "newcomer"])
    if args.with_wheels:
        prepare_wheels(code_changed)
    ensure_profile(code_changed, code_changed / "profiles/default.json", workspace=work, plugins_home=home,
                   config_path=config, receipt_path=receipt)
    changed = selected(work)
    assert changed["alpha@release"][0] != refs_before["alpha@release"][0]
    assert changed["alpha@release"][1]["python_environments"] == refs_before["alpha@release"][1]["python_environments"]
    assert snapshot(work / "runtime/plugin-python-environments") == environments_before
    assert changed["outside@thirdparty"] == refs_before["outside@thirdparty"]
    assert Path(changed["optional@release"][1]["code"]) == code_changed / "sources/optional"
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
    # 新发行版自动先迁移；显式停用不影响内置数据升级。
    set_plugin_enabled("disabled@release", enabled=False, plugins_home=home)
    expected_bytes = PluginSelection(work).path.read_bytes()
    migration(repo, "alpha", "scenario_alpha_upgrade", f"""
import json
from agent.migrations.context import current_migration_context
def upgrade(connection):
    context = current_migration_context()
    assert (context.workspace / "runtime/plugin-stable.json").read_bytes() == {expected_bytes!r}
    path = context.bundle_data_roots["alpha"] / "config.input.json"
    value = json.loads(path.read_text())
    value["config"][1]["upgraded"] = True
    path.write_text(json.dumps(value))
    connection.execute("CREATE TABLE scenario_alpha_done (value INTEGER)")
    connection.execute("INSERT INTO scenario_alpha_done VALUES (1)")
step(upgrade)
""")
    migration(repo, "disabled", "scenario_disabled_upgrade",
              'step("CREATE TABLE scenario_disabled_done (value INTEGER)")\n')
    migrated = distribution(repo, root / "migrated", ["alpha", "disabled", "optional", "newcomer"],
                            ["alpha", "disabled", "newcomer"])
    if args.with_wheels:
        prepare_wheels(migrated)
    ensure_profile(migrated, migrated / "profiles/default.json", workspace=work, plugins_home=home,
                   config_path=config, receipt_path=receipt)
    assert "disabled@release" not in selected(work)
    assert selected(work)["outside@thirdparty"][0] == external_before
    from agent.plugin_composition.config_input import decode_config
    assert load_config(work / "plugin-data/alpha-release")[0]["upgraded"] is True
    with sqlite3.connect(work / "migrations.sqlite3") as ledger:
        assert ledger.execute("SELECT COUNT(*) FROM scenario_alpha_done").fetchone() == (1,)
        assert ledger.execute("SELECT COUNT(*) FROM scenario_disabled_done").fetchone() == (0,)
    # 模拟迁移已成功而组合尚未提交；重试仍收录已迁移配置，不重跑 step。
    PluginSelection(work).path.write_bytes(expected_bytes)
    ensure_profile(migrated, migrated / "profiles/default.json", workspace=work, plugins_home=home,
                   config_path=config, receipt_path=receipt)
    assert load_config(work / "plugin-data/alpha-release")[0]["upgraded"] is True
    # 正常 runtime 启动也只检查当前发行版，不解析外置 catalog。
    previous_distribution = os.environ.get("AKASHIC_PLUGIN_DISTRIBUTION")
    os.environ["AKASHIC_PLUGIN_DISTRIBUTION"] = str(migrated)
    try:
        assert MigrationRunner(repo_root=ROOT, config_path=config, workspace=work,
                               startup_selection=True).run().state == "current"
    finally:
        if previous_distribution is None:
            os.environ.pop("AKASHIC_PLUGIN_DISTRIBUTION")
        else:
            os.environ["AKASHIC_PLUGIN_DISTRIBUTION"] = previous_distribution
    # 发布入口的只读预检列出待迁移，执行时自动迁移；失败不改变 selection。
    migration(repo, "optional", "scenario_publish_upgrade",
              'step("CREATE TABLE scenario_publish_done (value INTEGER)")\n')
    published = distribution(repo, root / "published", ["alpha", "disabled", "optional", "newcomer"],
                             ["alpha", "disabled", "newcomer"])
    if args.with_wheels:
        prepare_wheels(published)
    plan.write_text(json.dumps({"schema_version": 1, "expected_root_ref": PluginSelection(work).read(),
                                "targets": []}))
    before_preflight = snapshot(state)
    preflight = publish_distribution(distribution=published, workspace=work, plugins_home=home,
                                    config_path=config, plan=plan, inputs=root, preflight_only=True)
    assert preflight["migration_ids"] == ["scenario_publish_upgrade"]
    after_preflight = snapshot(state)
    assert {k: v for k, v in after_preflight.items()
            if not k.startswith(("workspace/runtime/plugin-archives/", "workspace/runtime/plugin-python-environments/"))} == {
        k: v for k, v in before_preflight.items()
        if not k.startswith(("workspace/runtime/plugin-archives/", "workspace/runtime/plugin-python-environments/"))}
    publish_distribution(distribution=published, workspace=work, plugins_home=home,
                         config_path=config, plan=plan, inputs=root)
    migration(repo, "optional", "scenario_failed_upgrade",
              'def fail(connection):\n    raise RuntimeError("expected migration failure")\nstep(fail)\n')
    failed = distribution(repo, root / "failed", ["alpha", "disabled", "optional", "newcomer"],
                          ["alpha", "disabled", "newcomer"])
    if args.with_wheels:
        prepare_wheels(failed)
    before_failure = PluginSelection(work).read()
    try:
        ensure_profile(failed, failed / "profiles/default.json", workspace=work, plugins_home=home,
                       config_path=config, receipt_path=receipt)
    except RuntimeError as error:
        assert "expected migration failure" in str(error)
    else:
        raise AssertionError("failed migration accepted")
    assert PluginSelection(work).read() == before_failure
    assert (work / "sessions.db").read_bytes() == messages_bytes
    # Core 同样无需逐 ID 审批，真实 Yoyo 成功账本保证二次启动不重跑。
    core_source = root / "core-fixture"
    core_migrations = core_source / "migrations/core"
    core_migrations.mkdir(parents=True)
    (core_migrations / "scenario_core_upgrade.py").write_text(
        'from yoyo import step\nstep("CREATE TABLE scenario_core_done (value INTEGER)")\n')
    runner = MigrationRunner(repo_root=core_source, workspace=work, config_path=config)
    assert runner.check() == ("scenario_core_upgrade",)
    assert runner.run().migrations == ("scenario_core_upgrade",)
    assert runner.run().state == "current"
    print(
        "PASS update/removal/optional/default/disabled/config/external/byte-preservation/restart/enable/environment/uninstall/preflight/publish/failure/core-yoyo/builtin-yoyo/external-exclusion/migration-retry",
        flush=True,
    )
    (root / "result.json").write_text(
        json.dumps(
            {
                "result": "passed",
                "candidate_checkout": str(ROOT),
                "source_identity": identity,
                "with_wheels": args.with_wheels,
                "seed_core": str(args.seed_core.resolve()) if args.seed_core else None,
                "cross_version_preflight": bool(args.seed_core),
                "scenarios": 24,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--with-wheels", action="store_true")
    parser.add_argument("--seed-core", type=Path, help="用旧 Core 生成已选归档后验证新发布 CLI")
    args = parser.parse_args()
    asyncio.run(run(args))
