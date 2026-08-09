from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from infra.mobile_webui.auto_publish import auto_publish_webui
from infra.mobile_webui.store import MobileWebUiStore


def _run(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(tmp_path: Path, *, branch: str = "main") -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _run(repository, "init", "-q", "-b", branch)
    _run(repository, "config", "user.email", "test@example.invalid")
    _run(repository, "config", "user.name", "Test")
    (repository / "tracked.txt").write_text("source\n", encoding="utf-8")
    (repository / "scripts").mkdir()
    (repository / "scripts/publish-mobile-webui.py").write_text("pass\n", encoding="utf-8")
    _run(repository, "add", ".")
    _run(repository, "commit", "-qm", "initial")
    head = _run(repository, "rev-parse", "HEAD")
    _run(repository, "update-ref", "refs/remotes/origin/main", head)
    return repository


def test_feature_branch_does_not_reconcile(tmp_path: Path) -> None:
    repository = _repository(tmp_path, branch="feature")
    assert not auto_publish_webui(
        repository,
        tmp_path / "workspace",
        server_id="server-1",
    )


def test_unsynchronized_main_fails_loud(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    (repository / "tracked.txt").write_text("next\n", encoding="utf-8")
    _run(repository, "commit", "-am", "next", "-q")
    with pytest.raises(RuntimeError, match="拒绝未同步的 main"):
        auto_publish_webui(
            repository,
            tmp_path / "workspace",
            server_id="server-1",
        )


def test_dirty_main_does_not_reconcile(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    (repository / "tracked.txt").write_text("dirty\n", encoding="utf-8")

    assert not auto_publish_webui(
        repository,
        tmp_path / "workspace",
        server_id="server-1",
    )


def test_new_main_invokes_stable_publisher(tmp_path: Path) -> None:
    repository = _repository(tmp_path)

    assert auto_publish_webui(
        repository,
        tmp_path / "workspace",
        server_id="server-1",
    )


def test_applied_main_is_a_noop(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    head = _run(repository, "rev-parse", "HEAD")
    store = MobileWebUiStore(tmp_path / "workspace/mobile-webui", server_id="server-1")
    store._db.execute(
        "INSERT INTO webui_generations(generation_id, target_key, manifest_digest, manifest_json, created_at, source_repository, source_commit, source_tree) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("a" * 64, "b" * 64, "c" * 64, b"{}", "2026-08-08T00:00:00Z", "repo", head, "d" * 40),
    )
    store._db.execute(
        "INSERT INTO webui_publication_journal(sequence, generation_id, operation, release_epoch, stable, preview, actor, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (1, "a" * 64, "publish", store._lineage_epoch(), 1, 0, "test", "2026-08-08T00:00:00Z"),
    )
    store.close()

    assert not auto_publish_webui(
        repository,
        tmp_path / "workspace",
        server_id="server-1",
    )
