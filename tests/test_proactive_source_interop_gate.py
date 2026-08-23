from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest


GATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docker"
    / "debug"
    / "proactive_source_interop_gate.py"
)
SPEC = importlib.util.spec_from_file_location("proactive_source_interop_gate", GATE_PATH)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _plugin_repo(tmp_path: Path, plugin_id: str = "fixture") -> Path:
    root = tmp_path / plugin_id
    root.mkdir()
    _ = _git(root, "init", "--quiet")
    _ = _git(root, "config", "user.email", "fixture@example.com")
    _ = _git(root, "config", "user.name", "Fixture")
    (root / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        f'name = "{plugin_id}"\n'
        'version = "3.0.0"\n'
        "api_version = 3\n"
        'entrypoint = "plugin.py"\n',
        encoding="utf-8",
    )
    (root / "plugin.py").write_text(
        "api_version = 3\n"
        f'name = "{plugin_id}"\n'
        "async def apply(ctx, config):\n"
        "    del ctx, config\n",
        encoding="utf-8",
    )
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_plugin.py").write_text(
        "def test_fixture():\n    assert True\n",
        encoding="utf-8",
    )
    _ = _git(root, "add", ".")
    _ = _git(root, "commit", "--quiet", "-m", "test: fixture")
    return root


def test_lock_pins_real_revisions_and_keeps_unresolved_consumers_pending() -> None:
    contract = gate._load_contract(gate.DEFAULT_LOCK)

    assert contract.core_contract == "9da3a988a2bf62b0f550bd4f6bb98c4eeb1f56f5"
    assert tuple(plugin.id for plugin in contract.plugins) == (
        "calendar",
        "fitbit",
        "feed",
        "steam",
        "github-watch",
        "emotion",
        "observe",
    )
    assert all(len(plugin.resolved_sha) == 40 for plugin in contract.plugins)
    assert {item["id"] for item in contract.pending} == {
        "emotion_feedback_interop",
        "proactive_feedback",
    }
    github_watch = next(
        plugin for plugin in contract.plugins if plugin.id == "github-watch"
    )
    assert github_watch.pull_request is None
    assert "content.source.v1" not in github_watch.atoms
    assert contract.retired[0]["disposition"] == "delete_zero_runtime_consumers"


def test_lock_rejects_schema_drift_and_short_revision(tmp_path: Path) -> None:
    raw = json.loads(gate.DEFAULT_LOCK.read_text(encoding="utf-8"))
    raw["extra"] = True
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(gate.GateError, match="根结构"):
        gate._load_contract(invalid)

    del raw["extra"]
    raw["plugins"][0]["resolved_sha"] = "abc"
    invalid.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(gate.GateError, match="完整 SHA"):
        gate._load_contract(invalid)


def test_exact_plugin_verification_rejects_dirty_and_old_proactive_seams(
    tmp_path: Path,
) -> None:
    root = _plugin_repo(tmp_path)
    template = gate._load_contract(gate.DEFAULT_LOCK).plugins[0]
    contract = replace(
        template,
        id="fixture",
        resolved_sha=_git(root, "rev-parse", "HEAD"),
        test_cwd="tests",
        cases=("tests/test_plugin.py",),
    )

    receipt = gate._verify_plugin(contract, root)
    assert receipt["status"] == "verified"

    (root / "plugin.py").write_text(
        (root / "plugin.py").read_text(encoding="utf-8")
        + "\nPROACTIVE_COMPONENTS = ()\n",
        encoding="utf-8",
    )
    with pytest.raises(gate.GateError, match="非 clean"):
        gate._verify_plugin(contract, root)

    _ = _git(root, "add", "plugin.py")
    _ = _git(root, "commit", "--quiet", "-m", "test: mutant")
    mutant = replace(contract, resolved_sha=_git(root, "rev-parse", "HEAD"))
    with pytest.raises(gate.GateError, match="proactive-only seam"):
        gate._verify_plugin(mutant, root)


def test_path_map_requires_exact_absolute_id_bindings(tmp_path: Path) -> None:
    assert gate._parse_path_map([f"fixture={tmp_path}"], "--plugin-root") == {
        "fixture": tmp_path
    }
    with pytest.raises(gate.GateError, match="id=/absolute/path"):
        gate._parse_path_map(["fixture"], "--plugin-root")
    with pytest.raises(gate.GateError, match="绝对路径"):
        gate._parse_path_map(["fixture=relative"], "--plugin-root")


def test_runner_replays_owner_fixture_without_copying_plugin_logic(
    tmp_path: Path,
) -> None:
    root = _plugin_repo(tmp_path)

    receipt = gate._run_cases(
        Path(sys.executable),
        root / "tests",
        ("tests/test_plugin.py",),
        root,
    )

    assert receipt["returncode"] == 0
    assert "1 passed" in receipt["stdout_tail"]


@pytest.mark.asyncio
async def test_generic_coexistence_probe_keeps_non_content_plugin_out_of_mailbox(
    tmp_path: Path,
) -> None:
    root = _plugin_repo(tmp_path)

    receipt = await gate._run_coexistence_probe(
        {
            "plugin_id": "fixture",
            "config_toml": "",
            "expected_content_rows": 0,
        },
        root,
    )

    assert receipt == {
        "plugin_id": "fixture",
        "content_rows": 0,
        "formal_content_write_set": [],
    }
