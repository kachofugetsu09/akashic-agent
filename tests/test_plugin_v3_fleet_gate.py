from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


_GATE_PATH = Path(__file__).resolve().parents[1] / "docker/debug/plugin_v3_fleet_gate.py"
_SPEC = importlib.util.spec_from_file_location("plugin_v3_fleet_gate", _GATE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
gate = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = gate
_SPEC.loader.exec_module(gate)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_v3_artifact(root: Path, source: str) -> None:
    root.mkdir()
    (root / "plugin.py").write_text(source, encoding="utf-8")
    (root / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        'name = "fixture"\n'
        'version = "3.0.0"\n'
        "api_version = 3\n"
        'entrypoint = "plugin.py"\n',
        encoding="utf-8",
    )


def test_lock_pins_exact_fleet_and_excludes_retired_plugins() -> None:
    plugins = gate._load_lock(gate.DEFAULT_LOCK)

    assert tuple(item.id for item in plugins) == gate.EXPECTED_PLUGIN_IDS
    assert len(plugins) == len(gate.EXPECTED_PLUGIN_IDS)
    assert not {item.id for item in plugins} & set(gate.EXCLUDED_PLUGIN_IDS)
    assert all(item.requested_ref == item.resolved_sha for item in plugins)
    assert all(item.change_source_pr_head == item.resolved_sha for item in plugins)
    assert all(len(item.resolved_sha) == 40 for item in plugins)


@pytest.mark.parametrize("excluded", gate.EXCLUDED_PLUGIN_IDS)
def test_lock_hard_rejects_excluded_plugins(tmp_path: Path, excluded: str) -> None:
    raw = json.loads(gate.DEFAULT_LOCK.read_text(encoding="utf-8"))
    revision = "1" * 40
    raw["plugins"].append(
        {
            "id": excluded,
            "repository": "https://github.com/akashic-plugins/retired",
            "requested_ref": revision,
            "resolved_sha": revision,
            "change_source_pr_head": revision,
        }
    )
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="硬排除"):
        gate._load_lock(lock)


def test_lock_rejects_schema_drift_and_non_full_sha(tmp_path: Path) -> None:
    raw = json.loads(gate.DEFAULT_LOCK.read_text(encoding="utf-8"))
    raw["extra"] = True
    lock = tmp_path / "extra.json"
    lock.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="根结构"):
        gate._load_lock(lock)

    raw = json.loads(gate.DEFAULT_LOCK.read_text(encoding="utf-8"))
    raw["plugins"][0]["resolved_sha"] = "a" * 39
    lock = tmp_path / "short-sha.json"
    lock.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="完整 40 位 SHA"):
        gate._load_lock(lock)


def test_e2e_catalog_is_explicitly_not_run() -> None:
    report = gate._e2e_report()

    assert report["status"] == "not_run"
    assert len(report["catalog_sha256"]) == 64
    catalog = report["catalog"]
    assert tuple(item["id"] for item in catalog) == ("E1", "E2", "E3", "E4")
    assert all(item["status"] == "not_run" for item in catalog)
    assert all(item["executed"] is False for item in catalog)


def test_static_gate_accepts_v3_namespace_and_manifest(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    _write_v3_artifact(
        root,
        "from agent.plugin_composition import Context\n"
        "api_version = 3\n"
        'name = "fixture"\n'
        'version = "3.0.0"\n'
        "async def apply(ctx: Context, config: object) -> None:\n"
        "    return None\n",
    )

    evidence = gate._inspect_static_plugin(root, "fixture")

    assert evidence["status"] == "passed"
    assert evidence["manifest"]["api_version"] == 3
    assert evidence["namespace"]["apply_signature"] == "apply(ctx, config)"
    assert evidence["forbidden_v2_imports"] == []
    assert evidence["forbidden_v2_classes"] == []


def test_static_gate_rejects_generic_v2_import(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    _write_v3_artifact(
        root,
        "from agent.plugins import Plugin\n"
        "api_version = 3\n"
        'name = "fixture"\n'
        'version = "3.0.0"\n'
        "async def apply(ctx, config) -> None:\n"
        "    return None\n",
    )

    evidence = gate._inspect_static_plugin(root, "fixture")

    assert evidence["status"] == "failed"
    assert evidence["forbidden_v2_imports"][0]["module"] == "agent.plugins"


@pytest.mark.parametrize(
    "legacy_class, expected",
    (
        (
            "class Legacy(Plugin):\n"
            "    pass\n",
            "Plugin",
        ),
        (
            "class LegacyCommands:\n"
            "    def telegram_bot_commands(self):\n"
            "        return []\n",
            "telegram_bot_commands",
        ),
    ),
)
def test_static_gate_rejects_legacy_v2_class_contracts(
    tmp_path: Path,
    legacy_class: str,
    expected: str,
) -> None:
    root = tmp_path / "fixture"
    _write_v3_artifact(
        root,
        "api_version = 3\n"
        'name = "fixture"\n'
        'version = "3.0.0"\n'
        "async def apply(ctx, config):\n"
        "    return None\n\n"
        f"{legacy_class}",
    )

    evidence = gate._inspect_static_plugin(root, "fixture")

    assert evidence["status"] == "failed"
    assert expected in str(evidence["forbidden_v2_classes"])


def test_locked_checkout_records_tree_clean_and_shallow_history(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "--quiet")
    _git(source, "config", "user.email", "test@example.invalid")
    _git(source, "config", "user.name", "test")
    (source / "plugin.py").write_text("api_version = 3\n", encoding="utf-8")
    _git(source, "add", ".")
    _git(source, "commit", "--quiet", "-m", "fixture")
    revision = _git(source, "rev-parse", "HEAD")
    tree = _git(source, "rev-parse", "HEAD^{tree}")
    remote = tmp_path / "remote.git"
    _git(source, "init", "--bare", str(remote))
    _git(source, "push", "--quiet", str(remote), f"HEAD:refs/heads/main")

    lock = gate.PluginLock(
        id="fixture",
        repository=str(remote),
        requested_ref=revision,
        resolved_sha=revision,
        change_source_pr_head=revision,
    )
    evidence = gate._checkout_locked_plugin(lock, tmp_path / "checkout")

    assert evidence.resolved_sha == revision
    assert evidence.tree == tree
    assert evidence.clean is True
    assert evidence.dirty_status == ()
    assert evidence.history == "shallow"
    assert evidence.remote_ref == "refs/heads/main"


def test_locked_checkout_fails_loud_on_unreachable_sha(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "--quiet")
    _git(source, "config", "user.email", "test@example.invalid")
    _git(source, "config", "user.name", "test")
    (source / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(source, "add", ".")
    _git(source, "commit", "--quiet", "-m", "fixture")
    remote = tmp_path / "remote.git"
    _git(source, "clone", "--quiet", "--bare", ".", str(remote))
    lock = gate.PluginLock(
        id="fixture",
        repository=str(remote),
        requested_ref="f" * 40,
        resolved_sha="f" * 40,
        change_source_pr_head="f" * 40,
    )

    with pytest.raises(gate.GateError, match="未找到可达"):
        gate._checkout_locked_plugin(lock, tmp_path / "checkout")


def test_report_schema_keeps_core_and_e2e_evidence() -> None:
    core = {
        "commit": "a" * 40,
        "tree": "b" * 40,
        "dirty": [],
        "clean": True,
        "history": "full",
    }
    report = gate._build_report(gate.DEFAULT_LOCK, core, (), [])

    assert report["status"] == "passed"
    assert len(report["lock_sha256"]) == 64
    assert report["core"]["commit"] == "a" * 40
    assert report["core"]["tree"] == "b" * 40
    assert report["e2e"]["status"] == "not_run"
