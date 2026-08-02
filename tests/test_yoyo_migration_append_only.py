from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import scripts.check_yoyo_migrations as checker


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
    migration = repo / "migrations/yoyo/20260802_01_origin.py"
    migration.parent.mkdir(parents=True)
    migration.write_text("steps = []\n", encoding="utf-8")
    _ = _git(repo, "init", "-b", "main")
    _ = _git(repo, "config", "user.email", "test@example.com")
    _ = _git(repo, "config", "user.name", "Test")
    _ = _git(repo, "add", ".")
    _ = _git(repo, "commit", "-m", "baseline")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_existing_yoyo_migration_cannot_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repository(tmp_path)
    migration = repo / "migrations/yoyo/20260802_01_origin.py"
    migration.write_text("steps = ['changed']\n", encoding="utf-8")
    monkeypatch.setattr(checker, "ROOT", repo)

    assert checker.violations(base) == [
        "registered Yoyo migration changed: migrations/yoyo/20260802_01_origin.py"
    ]


def test_new_yoyo_migration_is_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repository(tmp_path)
    (repo / "migrations/yoyo/20260803_01_next.py").write_text(
        "steps = []\n",
        encoding="utf-8",
    )
    _ = _git(repo, "add", "migrations/yoyo/20260803_01_next.py")
    monkeypatch.setattr(checker, "ROOT", repo)

    assert checker.violations(base) == []


def test_archived_migration_is_outside_the_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repository(tmp_path)
    archived = repo / "migrations/provider_runtimes_and_akasha/migration.py"
    archived.parent.mkdir(parents=True)
    archived.write_text("historical = True\n", encoding="utf-8")
    _ = _git(repo, "add", "migrations/provider_runtimes_and_akasha/migration.py")
    monkeypatch.setattr(checker, "ROOT", repo)

    assert checker.violations(base) == []


def test_existing_yoyo_migration_cannot_be_deleted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repository(tmp_path)
    (repo / "migrations/yoyo/20260802_01_origin.py").unlink()
    monkeypatch.setattr(checker, "ROOT", repo)

    assert checker.violations(base) == [
        "registered Yoyo migration changed: migrations/yoyo/20260802_01_origin.py"
    ]
