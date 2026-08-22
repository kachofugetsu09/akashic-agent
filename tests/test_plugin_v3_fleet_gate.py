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


_LEGACY_V2_CONSUMER_MARKERS = (
    "CoreLegacyChannelAdapter",
    "_core_legacy_",
    "encode_legacy_channel_message",
    "map_legacy_delivery_receipt",
    "PluginContext",
    "from agent.plugins import Plugin",
    "from agent.plugins.base import Plugin",
    "from agent.plugins.context import PluginContext",
    "from agent.plugins.decorators import",
    "api_version = 2",
    "plugin_api_v2",
    "plugin-api-v2",
)


def _source_files_for_consumer_scan(root: Path) -> tuple[Path, ...]:
    """Return executable fleet, CI, and runtime files for the v2 scan."""

    roots = (
        root / "agent",
        root / "bootstrap",
        root / "infra",
        root / "plugins",
        root / "docker" / "debug",
        root / ".github" / "workflows",
    )
    suffixes = {".py", ".yml", ".yaml", ".json"}
    return tuple(
        path
        for directory in roots
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and path.suffix in suffixes
        and not {
            ".git",
            ".venv",
            "__pycache__",
            "reports",
        }.intersection(path.relative_to(root).parts)
    )


def _legacy_v2_consumers(root: Path) -> tuple[tuple[str, int, str], ...]:
    """Find legacy v2 admission edges in executable source and CI inputs."""

    findings: list[tuple[str, int, str]] = []
    for path in _source_files_for_consumer_scan(root):
        relative = path.relative_to(root).as_posix()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for marker in _LEGACY_V2_CONSUMER_MARKERS:
                if marker in line:
                    findings.append((relative, line_number, marker))
    return tuple(findings)


def _retired_fleet_ci_consumers(root: Path) -> tuple[tuple[str, int, str], ...]:
    """Find retired Computer Use or Context Pressure references in fleet/CI inputs."""

    retired_ids = gate.EXCLUDED_PLUGIN_IDS
    paths = tuple(sorted((root / ".github" / "workflows").glob("*.y*ml"))) + tuple(
        sorted((root / "docker" / "debug").glob("*.lock.json"))
    )
    findings: list[tuple[str, int, str]] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for plugin_id in retired_ids:
                if plugin_id in line:
                    findings.append((path.relative_to(root).as_posix(), line_number, plugin_id))
    return tuple(findings)


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


def test_strict_v3_production_has_no_legacy_v2_consumers() -> None:
    assert _legacy_v2_consumers(gate.ROOT) == ()


def test_retired_plugins_have_zero_fleet_and_ci_consumers() -> None:
    assert _retired_fleet_ci_consumers(gate.ROOT) == ()


def test_zero_consumer_oracles_kill_injected_consumers(tmp_path: Path) -> None:
    workflow_root = tmp_path / ".github" / "workflows"
    workflow_root.mkdir(parents=True)
    (workflow_root / "mutant.yml").write_text(
        "run: context_pressure\n",
        encoding="utf-8",
    )
    runtime_root = tmp_path / "agent"
    runtime_root.mkdir()
    (runtime_root / "mutant.py").write_text(
        "from agent.plugins import Plugin\n",
        encoding="utf-8",
    )

    assert _retired_fleet_ci_consumers(tmp_path)
    assert _legacy_v2_consumers(tmp_path)


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
