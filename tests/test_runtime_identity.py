from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agent.runtime_identity import RuntimeIdentity


def _runtime_info(commit: str, tree: str) -> dict[str, object]:
    return {
        "schemaVersion": 2,
        "sourceCommit": commit,
        "sourceTree": tree,
        "sourceArchiveSha256": "a" * 64,
        "sourceManifestSha256": "b" * 64,
        "baseImage": "archlinux@sha256:" + "c" * 64,
        "archSnapshot": "2026/08/10",
        "pacmanDigest": "d" * 64,
        "requirementsLockSha256": "e" * 64,
        "packageLockSha256": "f" * 64,
        "pythonVersion": "3.14.0",
        "nodeVersion": "v22.23.1",
        "npmVersion": "11.0.0",
    }


def _release_manifest(tmp_path: Path, runtime_info: dict[str, object]) -> Path:
    path = tmp_path / "release.json"
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "imageId": "sha256:" + "1" * 64,
                "runtimeInfo": runtime_info,
            }
        ),
        encoding="utf-8",
    )
    return path


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
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "fixture"], check=True)
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
    runtime_document = _runtime_info(commit, tree)
    info.write_text(
        json.dumps(runtime_document),
        encoding="utf-8",
    )

    identity = RuntimeIdentity.load(
        info,
        _release_manifest(tmp_path, runtime_document),
        expected_commit=commit,
        host_checkout=checkout,
    )

    assert identity.source_commit == commit
    assert identity.source_tree == tree
    assert identity.source_archive_sha256 == "a" * 64
    assert len(identity.environment_digest) == 64


def test_runtime_identity_rejects_mismatched_commit(tmp_path: Path) -> None:
    checkout, commit, tree = _checkout(tmp_path)
    info = tmp_path / "runtime-info.json"
    runtime_document = _runtime_info(commit, tree)
    info.write_text(
        json.dumps(runtime_document),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="runtime commit 不一致"):
        RuntimeIdentity.load(
            info,
            _release_manifest(tmp_path, runtime_document),
            expected_commit="c" * 40,
            host_checkout=checkout,
        )


def test_runtime_identity_rejects_unpinned_environment(tmp_path: Path) -> None:
    checkout, commit, tree = _checkout(tmp_path)
    document = _runtime_info(commit, tree)
    document["baseImage"] = "archlinux:latest"
    info = tmp_path / "runtime-info.json"
    info.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RuntimeError, match="baseImage 必须固定 digest"):
        RuntimeIdentity.load(
            info,
            _release_manifest(tmp_path, document),
            expected_commit=commit,
            host_checkout=checkout,
        )
