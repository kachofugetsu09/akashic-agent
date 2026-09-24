"""Black-box checks for the isolated one-Root composition CLI."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "plugin_composition_experiment.py"


def _run_cli(
    workspace: Path,
    scratch: Path,
    *,
    formal_workspace: Path | None = None,
    plugin_home: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the shipped entrypoint with only isolated roots."""

    env = os.environ.copy()
    env.update({
        "AKASHIC_WORKSPACE": str(formal_workspace or scratch / "protected-workspace"),
        "AKASHIC_PLUGIN_HOME": str(plugin_home or scratch / "protected-plugin-home"),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return subprocess.run(
        [sys.executable, "-B", str(CLI), "--workspace", str(workspace)],
        cwd=scratch,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_cli_replaces_provider_in_one_root_and_closes_real_effects(tmp_path: Path) -> None:
    workspace = tmp_path / "experiment"
    run = _run_cli(workspace, tmp_path)
    assert run.returncode == 0, run.stderr

    result_path = workspace / "runtime" / "plugin-composition-result.json"
    assert run.stdout.strip() == str(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    marker_path = workspace / "runtime" / "plugin-composition-experiment.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["run_id"] == result["run_id"]
    assert marker["workspace"] == result["workspace"] == str(workspace)

    receipts = result["receipts"]
    assert set(receipts) == {"pending", "optional", "ready", "removed", "replaced", "disposed"}
    assert not receipts["pending"]["ready"]
    assert "probe-consumer" in receipts["pending"]["required_pending"]
    assert receipts["optional"]["ready"]
    assert "probe-formatter-consumer" in receipts["optional"]["optional_pending"]
    assert receipts["ready"]["ready"]
    assert not receipts["removed"]["ready"]
    assert "probe-consumer" in receipts["removed"]["required_pending"]
    assert receipts["replaced"]["ready"]
    assert not receipts["disposed"]["ready"]
    assert receipts["disposed"]["fibers"] == []
    assert receipts["disposed"]["effects"] == []
    assert result["current_signal_after_replacement"] == "second"
    assert result["external_effect_count"] == 0

    trace = result["trace"]
    for event in (
        "provider:load:first", "consumer:load:first", "consumer:formatted:FIRST",
        "consumer:cleanup:first", "provider:cleanup:first", "provider:load:second",
        "consumer:load:second", "consumer:formatted:SECOND",
        "consumer:cleanup:second", "provider:cleanup:second",
    ):
        assert trace.count(event) == 1
    assert trace.index("consumer:cleanup:first") < trace.index("provider:cleanup:first")
    assert trace.index("provider:cleanup:first") < trace.index("provider:load:second")
    assert trace.index("consumer:cleanup:second") < trace.index("provider:cleanup:second")

    state_path = workspace / "plugin-data" / "probe-provider" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["value"] == "second"
    assert not (workspace / "formal-plugin-data").exists()

    listed = result["workspace_files_before_result"]
    listed_paths = {item["path"] for item in listed}
    actual_paths = {
        str(path.relative_to(workspace))
        for path in workspace.rglob("*")
        if path.is_file() and path != result_path
    }
    assert listed_paths == actual_paths
    assert str(marker_path.relative_to(workspace)) in listed_paths
    assert str(state_path.relative_to(workspace)) in listed_paths
    for item in listed:
        content = (workspace / item["path"]).read_bytes()
        assert item["sha256"] == hashlib.sha256(content).hexdigest()


def test_cli_rejects_existing_workspace_without_writes(tmp_path: Path) -> None:
    workspace = tmp_path / "existing"
    workspace.mkdir()
    sentinel = workspace / "keep.txt"
    sentinel.write_bytes(b"keep existing data\n")

    run = _run_cli(workspace, tmp_path)

    assert run.returncode != 0
    assert "必须尚不存在" in run.stderr
    assert sentinel.read_bytes() == b"keep existing data\n"
    assert list(workspace.iterdir()) == [sentinel]


def test_cli_rejects_source_contained_workspace_without_writes(tmp_path: Path) -> None:
    workspace = ROOT / "tests" / f"experiment-denied-{tmp_path.name}"
    assert not workspace.exists()

    run = _run_cli(workspace, tmp_path)

    assert run.returncode != 0
    assert "不能位于源码 worktree 内" in run.stderr
    assert not workspace.exists()


@pytest.mark.parametrize("root_name", ["formal_workspace", "plugin_home"])
@pytest.mark.parametrize("nested", [False, True])
def test_cli_rejects_protected_roots_without_writes(
    tmp_path: Path, root_name: str, nested: bool,
) -> None:
    protected = tmp_path / root_name
    protected.mkdir()
    sentinel = protected / "keep.txt"
    sentinel.write_bytes(b"protected data\n")
    workspace = protected / "nested" if nested else protected
    if root_name == "formal_workspace":
        run = _run_cli(workspace, tmp_path, formal_workspace=protected)
    else:
        run = _run_cli(workspace, tmp_path, plugin_home=protected)

    assert run.returncode != 0
    assert "不能位于正式状态根内" in run.stderr
    assert sentinel.read_bytes() == b"protected data\n"
    assert list(protected.iterdir()) == [sentinel]


def test_cli_rejects_missing_parent_without_writes(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    run = _run_cli(missing / "experiment", tmp_path)

    assert run.returncode != 0
    assert "父目录不存在" in run.stderr
    assert not missing.exists()
