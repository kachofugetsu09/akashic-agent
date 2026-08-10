from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.prepare_runtime_checkout import prepare_runtime_checkout


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_runtime_checkout_does_not_carry_deleted_secret_history(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q")
    _git(source, "config", "user.email", "test@example.invalid")
    _git(source, "config", "user.name", "Test")
    secret = source / "config.toml"
    secret.write_text("token='secret'\n", encoding="utf-8")
    _git(source, "add", ".")
    _git(source, "commit", "-qm", "secret parent")
    parent = _git(source, "rev-parse", "HEAD")
    secret.unlink()
    (source / "main.py").write_text("print('safe')\n", encoding="utf-8")
    _git(source, "add", "-A")
    _git(source, "commit", "-qm", "safe release")
    commit = _git(source, "rev-parse", "HEAD")

    target = prepare_runtime_checkout(
        source, commit, tmp_path / "runtime", "git@example.invalid:owner/repo.git"
    )

    assert _git(target, "rev-parse", "HEAD") == commit
    assert _git(target, "rev-list", "--all", "--count") == "1"
    missing_parent = subprocess.run(
        ["git", "cat-file", "-e", parent], cwd=target, capture_output=True
    )
    assert missing_parent.returncode != 0
    assert not (target / "config.toml").exists()
    assert _git(target, "remote", "get-url", "origin") == (
        "git@example.invalid:owner/repo.git"
    )
