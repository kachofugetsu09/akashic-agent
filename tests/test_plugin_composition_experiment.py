from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_experiment_runs_full_candidate_promotion_in_isolated_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/plugin_composition_experiment.py",
            "--workspace",
            str(workspace),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr

    result_path = Path(result.stdout.strip())
    evidence = json.loads(result_path.read_text(encoding="utf-8"))
    assert evidence["workspace"] == str(workspace)
    assert evidence["observed_signal"] == "first"
    assert evidence["promoted_signal"] == "second"
    assert evidence["receipts"]["pending"]["ready"] is False
    assert evidence["receipts"]["ready"]["ready"] is True
    assert evidence["receipts"]["removed"]["ready"] is False
    assert evidence["receipts"]["restored"]["ready"] is True
    assert evidence["receipts"]["promoted"]["ready"] is True
    assert evidence["receipts"]["disposed"]["ready"] is False
    assert evidence["receipts"]["restored"]["external_effects"] == []
    assert evidence["receipts"]["restored"]["writes"] == []
    state = json.loads(
        (workspace / "plugin-data/probe-provider/state.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["value"] == "second"


def test_experiment_refuses_an_existing_workspace(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/plugin_composition_experiment.py",
            "--workspace",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert "实验 workspace 必须尚不存在" in result.stderr


def test_experiment_refuses_child_of_formal_workspace(tmp_path: Path) -> None:
    formal_workspace = tmp_path / "formal"
    formal_workspace.mkdir()
    environment = dict(os.environ)
    environment["AKASHIC_WORKSPACE"] = str(formal_workspace)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/plugin_composition_experiment.py",
            "--workspace",
            str(formal_workspace / "candidate"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert "不能位于正式状态根内" in result.stderr
