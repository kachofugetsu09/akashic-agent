from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from infra.mobile_webui.auto_publish import auto_publish_webui
from infra.mobile_webui.manifest import manifest_from_directory
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


def _advance_tracked_main(repository: Path) -> None:
    """推进 origin/main，同时保持本地 main 停在原提交。"""

    # 1. 在临时分支创建远端后继提交
    _run(repository, "switch", "-qc", "remote-main")
    (repository / "tracked.txt").write_text("remote\n", encoding="utf-8")
    _run(repository, "commit", "-am", "remote", "-q")
    remote_head = _run(repository, "rev-parse", "HEAD")

    # 2. 只推进 remote-tracking ref，再回到旧 main
    _run(repository, "update-ref", "refs/remotes/origin/main", remote_head)
    _run(repository, "switch", "-q", "main")


def test_feature_branch_does_not_reconcile(tmp_path: Path) -> None:
    repository = _repository(tmp_path, branch="feature")
    assert not auto_publish_webui(
        repository,
        tmp_path / "workspace",
        server_id="server-1",
    )


def test_unsynchronized_main_fails_loud(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _advance_tracked_main(repository)
    with pytest.raises(RuntimeError, match="拒绝未同步的 main"):
        auto_publish_webui(
            repository,
            tmp_path / "workspace",
            server_id="server-1",
        )


@pytest.mark.parametrize("stable_matches_head", [True, False])
def test_unsynchronized_main_with_existing_stable(
    tmp_path: Path,
    stable_matches_head: bool,
) -> None:
    repository = _repository(tmp_path)
    workspace = tmp_path / "workspace"
    head = _run(repository, "rev-parse", "HEAD")
    build = tmp_path / "stable-build"
    build.mkdir()
    (build / "mobile.html").write_text("<html>stable</html>\n", encoding="utf-8")
    manifest, contents = manifest_from_directory(
        build,
        source_repository=str(repository),
        source_commit=head if stable_matches_head else "e" * 40,
        source_tree=_run(repository, "rev-parse", "HEAD^{tree}"),
        input_digest="a" * 64,
        build_context_digest="b" * 64,
        dirty_provenance=None,
        reproducible=True,
        builder_identity={
            "node_version": "v22.23.1",
            "npm_version": "10.9.0",
            "package_lock_digest": "c" * 64,
            "build_script_digest": "d" * 64,
        },
    )
    store = MobileWebUiStore(workspace / "mobile-webui", server_id="server-1")
    before = store.publish(manifest, contents, stable=True, preview=False)
    store.close()
    _advance_tracked_main(repository)

    if stable_matches_head:
        assert not auto_publish_webui(repository, workspace, server_id="server-1")
    else:
        with pytest.raises(RuntimeError, match="拒绝未同步的 main"):
            auto_publish_webui(repository, workspace, server_id="server-1")

    store = MobileWebUiStore(workspace / "mobile-webui", server_id="server-1")
    try:
        assert store.get_release() == before
        assert store._db.execute(
            "SELECT COUNT(*) FROM webui_publication_journal"
        ).fetchone()[0] == 1
    finally:
        store.close()


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
        (1, "a" * 64, "publish", store._release_epoch(), 1, 0, "test", "2026-08-08T00:00:00Z"),
    )
    store.close()

    assert not auto_publish_webui(
        repository,
        tmp_path / "workspace",
        server_id="server-1",
    )
