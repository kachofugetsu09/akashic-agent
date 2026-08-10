from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.build_host_runtime_release import (
    _assert_release_paths_safe,
    _resolve_commit,
)


def _commit(tmp_path: Path, name: str) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repository, check=True)
    target = repository / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("credential material", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repository, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repository, commit


@pytest.mark.parametrize(
    "name",
    ["config.toml", "auth.json", ".env", "config.toml.settings-deadbeef.bak"],
)
def test_release_rejects_tracked_runtime_credentials(tmp_path: Path, name: str) -> None:
    repository, commit = _commit(tmp_path, name)
    with pytest.raises(RuntimeError, match="禁止发布"):
        _assert_release_paths_safe(repository, commit)


def test_release_accepts_example_configuration(tmp_path: Path) -> None:
    repository, commit = _commit(tmp_path, "config.example.toml")
    _assert_release_paths_safe(repository, commit)


def test_release_accepts_tracked_fixture_configuration(tmp_path: Path) -> None:
    repository, commit = _commit(tmp_path, "benchmark/config.toml")
    _assert_release_paths_safe(repository, commit)


def test_release_rejects_mutable_commit_reference(tmp_path: Path) -> None:
    repository, _ = _commit(tmp_path, "config.example.toml")
    with pytest.raises(RuntimeError, match="完整 40 位"):
        _resolve_commit(repository, "HEAD")
