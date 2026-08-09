from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agent.runtime_identity import RuntimeIdentity


def _checkout(tmp_path: Path) -> tuple[Path, str, str]:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Test"],
        check=True,
    )
    (checkout / "main.py").write_text("print('ok')\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "main.py"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "commit", "-qm", "fixture"], check=True
    )
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    return checkout, commit, tree


def test_runtime_identity_requires_image_and_deployment_commit_match(
    tmp_path: Path,
) -> None:
    checkout, commit, tree = _checkout(tmp_path)
    info = tmp_path / "runtime-info.json"
    info.write_text(
        json.dumps(
            {"schemaVersion": 1, "sourceCommit": commit, "sourceTree": tree}
        ),
        encoding="utf-8",
    )

    identity = RuntimeIdentity.load(
        info,
        expected_commit=commit,
        host_checkout=checkout,
    )

    assert identity.source_commit == commit
    assert identity.source_tree == tree


def test_runtime_identity_rejects_mismatched_commit(tmp_path: Path) -> None:
    checkout, commit, tree = _checkout(tmp_path)
    info = tmp_path / "runtime-info.json"
    info.write_text(
        json.dumps(
            {"schemaVersion": 1, "sourceCommit": commit, "sourceTree": tree}
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="runtime commit 不一致"):
        RuntimeIdentity.load(
            info,
            expected_commit="c" * 40,
            host_checkout=checkout,
        )
