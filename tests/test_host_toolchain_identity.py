from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.host_toolchain_identity import (
    declared_toolchain_identity,
    resolve_toolchain_identity,
)


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "rehearsal@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Rehearsal"], cwd=repository, check=True
    )
    (repository / "mise.toml").write_text(
        '[tools]\nnode="22.23.1"\nnpm="10.9.8"\npython="3.14.6"\n'
        'uv="0.12.3"\n"npm:@jackwener/opencli"="1.8.6"\n'
        'opencode="1.18.15"\n',
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repository, check=True)
    return repository


def _fake_mise(tmp_path: Path) -> Path:
    mise = tmp_path / "mise"
    versions = {
        "node": "v22.23.1",
        "npm": "10.9.8",
        "python": "Python 3.14.6",
        "uv": "uv 0.12.3",
        "opencli": "1.8.6",
        "opencode": "1.18.15",
    }
    mise.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"versions = {json.dumps(versions)}\n"
        "print(versions[sys.argv[3]])\n",
        encoding="utf-8",
    )
    mise.chmod(0o755)
    return mise


def test_resolve_toolchain_identity_is_commit_bound(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    identity = resolve_toolchain_identity(repository, _fake_mise(tmp_path))
    assert (
        identity["releaseCommit"]
        == subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    assert len(str(identity["toolchainDigest"])) == 64
    assert identity == declared_toolchain_identity(
        str(identity["releaseCommit"]), (repository / "mise.toml").read_bytes()
    )


def test_resolve_toolchain_identity_rejects_dirty_checkout(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    (repository / "dirty.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean"):
        resolve_toolchain_identity(repository, _fake_mise(tmp_path))
