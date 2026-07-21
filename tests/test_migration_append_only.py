from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.check_migrations_append_only import check_append_only


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    bundle = repo / "migrations" / "existing"
    bundle.mkdir(parents=True)
    _ = _git(repo, "init", "-b", "main")
    _ = _git(repo, "config", "user.email", "test@example.com")
    _ = _git(repo, "config", "user.name", "Test")
    (repo / "migrations" / ".root").write_text("a" * 40, encoding="ascii")
    (bundle / "migration.py").write_text("print('existing')\n", encoding="utf-8")
    _ = _git(repo, "add", "migrations")
    _ = _git(repo, "commit", "-m", "migration baseline")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_new_bundle_is_allowed(
    tmp_path: Path, monkeypatch,
) -> None:
    repo, base = _repository(tmp_path)
    bundle = repo / "migrations" / "new_bundle"
    bundle.mkdir()
    (bundle / "migration.py").write_text("print('new')\n", encoding="utf-8")
    _ = _git(repo, "add", "migrations")
    _ = _git(repo, "commit", "-m", "new migration")
    monkeypatch.chdir(repo)

    assert check_append_only(base) == []


def test_existing_bundle_cannot_change(
    tmp_path: Path, monkeypatch,
) -> None:
    repo, base = _repository(tmp_path)
    migration = repo / "migrations" / "existing" / "migration.py"
    migration.write_text("print('changed')\n", encoding="utf-8")
    _ = _git(repo, "add", "migrations")
    _ = _git(repo, "commit", "-m", "change old migration")
    monkeypatch.chdir(repo)

    violations = check_append_only(base)

    assert len(violations) == 1
    assert "migrations/existing/migration.py" in violations[0]


def test_existing_bundle_cannot_gain_helper(
    tmp_path: Path, monkeypatch,
) -> None:
    repo, base = _repository(tmp_path)
    helper = repo / "migrations" / "existing" / "helper.py"
    helper.write_text("value = 1\n", encoding="utf-8")
    _ = _git(repo, "add", "migrations")
    _ = _git(repo, "commit", "-m", "extend old migration")
    monkeypatch.chdir(repo)

    violations = check_append_only(base)

    assert len(violations) == 1
    assert "migrations/existing/helper.py" in violations[0]
