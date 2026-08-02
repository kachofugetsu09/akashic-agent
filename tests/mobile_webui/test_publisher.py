from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


_PUBLISHER_PATH = Path(__file__).parents[2] / "scripts" / "publish-mobile-webui.py"
_SPEC = importlib.util.spec_from_file_location("publish_mobile_webui", _PUBLISHER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PUBLISHER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PUBLISHER)


def _run(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _repo(tmp_path: Path, *, lock: bool = True) -> Path:
    repo = tmp_path / "repo"
    (repo / "frontend/chat").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "frontend/chat/mobile.html").write_text("base", encoding="utf-8")
    (repo / "package.json").write_text('{"name":"test"}\n', encoding="utf-8")
    (repo / "scripts/package-mobile-web.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    if lock:
        (repo / "package-lock.json").write_text(
            '{"name":"test","lockfileVersion":3,"packages":{"":{"name":"test"}}}\n',
            encoding="utf-8",
        )
    _run(repo, "init", "-q")
    _run(repo, "config", "user.email", "test@example.invalid")
    _run(repo, "config", "user.name", "Test")
    _run(repo, "add", ".")
    _run(repo, "commit", "-qm", "initial")
    _run(repo, "remote", "add", "origin", "https://github.com/example/test.git")
    return repo


def test_stable_without_commit_lock_fails_before_build(tmp_path: Path) -> None:
    repo = _repo(tmp_path, lock=False)
    with pytest.raises(RuntimeError, match="package-lock"):
        _PUBLISHER._build(repo, None, allow_dirty=False, source_commit=None, stable=True)


def test_dirty_preview_overlay_is_frozen_with_untracked_inputs(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "frontend/chat/mobile.html").write_text("dirty", encoding="utf-8")
    (repo / "frontend/chat/extra.js").write_text("extra", encoding="utf-8")
    (repo / ".gitignore").write_text("frontend/chat/ignored.js\n", encoding="utf-8")
    _run(repo, "add", ".gitignore")
    _run(repo, "commit", "-qm", "ignore rule")
    (repo / "frontend/chat/ignored.js").write_text("ignored", encoding="utf-8")
    commit = _PUBLISHER._git(repo, "rev-parse", "HEAD")
    with _PUBLISHER._build_source(repo, commit, dirty=True) as snapshot:
        assert (snapshot / "frontend/chat/mobile.html").read_text(encoding="utf-8") == "dirty"
        assert (snapshot / "frontend/chat/extra.js").read_text(encoding="utf-8") == "extra"
        assert not (snapshot / "frontend/chat/ignored.js").exists()
        (repo / "frontend/chat/mobile.html").write_text("changed-after-freeze", encoding="utf-8")
        assert (snapshot / "frontend/chat/mobile.html").read_text(encoding="utf-8") == "dirty"


def test_dirty_overlay_rejects_symlink_and_preserves_tracked_deletion(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "frontend/chat/mobile.html").unlink()
    commit = _PUBLISHER._git(repo, "rev-parse", "HEAD")
    with _PUBLISHER._build_source(repo, commit, dirty=True) as snapshot:
        assert not (snapshot / "frontend/chat/mobile.html").exists()
    (repo / "frontend/chat/link.js").symlink_to(repo / "package.json")
    with pytest.raises(RuntimeError, match="symlink"):
        with _PUBLISHER._build_source(repo, commit, dirty=True):
            pass


def test_clean_sidecar_rejects_source_race(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    before = _PUBLISHER._capture_provenance(repo)
    output = tmp_path / "output"
    output.mkdir()
    (output / "mobile.html").write_text("base", encoding="utf-8")
    (tmp_path / "output.provenance.json").write_text(
        json.dumps({**before, "artifact_digest": _PUBLISHER._artifact_digest(output)}),
        encoding="utf-8",
    )
    _ = _PUBLISHER._manifest(repo, output, allow_dirty=False)
    (repo / "frontend/chat/mobile.html").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source"):
        _PUBLISHER._manifest(repo, output, allow_dirty=False)
