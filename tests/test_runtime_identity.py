from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.runtime_identity import RuntimeIdentity


def test_runtime_identity_requires_image_and_deployment_commit_match(
    tmp_path: Path,
) -> None:
    commit = "a" * 40
    tree = "b" * 40
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
        host_checkout=Path("/srv/data/dev/akasic-agent"),
    )

    assert identity.source_commit == commit
    assert identity.source_tree == tree


def test_runtime_identity_rejects_mismatched_commit(tmp_path: Path) -> None:
    info = tmp_path / "runtime-info.json"
    info.write_text(
        json.dumps(
            {"schemaVersion": 1, "sourceCommit": "a" * 40, "sourceTree": "b" * 40}
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="runtime commit 不一致"):
        RuntimeIdentity.load(
            info,
            expected_commit="c" * 40,
            host_checkout=Path("/srv/data/dev/akasic-agent"),
        )
