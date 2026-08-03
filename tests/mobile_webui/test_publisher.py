from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest
from infra.mobile_webui.store import MobileWebUiStore


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
    (repo / "package.json").write_text(
        json.dumps(
            {
                "name": "test",
                "scripts": {
                    "build:mobile-web": (
                        "node -e \"const fs=require('fs');"
                        "fs.mkdirSync(process.env.AKASHIC_MOBILE_WEB_OUT_DIR,{recursive:true});"
                        "fs.copyFileSync('frontend/chat/mobile.html',"
                        "process.env.AKASHIC_MOBILE_WEB_OUT_DIR+'/mobile.html')\""
                    )
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
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
    output = tmp_path / "output"
    output.mkdir()
    environment = _PUBLISHER._build_environment(output)
    before = _PUBLISHER._capture_provenance(repo, environment=environment, output_dir=output)
    (output / "mobile.html").write_text("base", encoding="utf-8")
    (tmp_path / "output.provenance.json").write_text(
        json.dumps({**before, "artifact_digest": _PUBLISHER._artifact_digest(output)}),
        encoding="utf-8",
    )
    _ = _PUBLISHER._manifest(repo, output, allow_dirty=False)
    (repo / "frontend/chat/mobile.html").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source"):
        _PUBLISHER._manifest(repo, output, allow_dirty=False)


def test_build_context_tracks_effective_env_and_normalizes_output_dir(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    environment = os.environ.copy()
    environment["VITE_PUBLIC_THEME"] = "dark"
    environment["VITE_PRIVATE_TOKEN"] = "do-not-write-this-value"
    first = _PUBLISHER._capture_provenance(
        repo,
        environment={**environment, "AKASHIC_MOBILE_WEB_OUT_DIR": "/tmp/mobile-web-a"},
        output_dir=Path("/tmp/mobile-web-a"),
    )
    second = _PUBLISHER._capture_provenance(
        repo,
        environment={**environment, "AKASHIC_MOBILE_WEB_OUT_DIR": "/tmp/mobile-web-b"},
        output_dir=Path("/tmp/mobile-web-b"),
    )
    assert first["build_context_digest"] == second["build_context_digest"]
    assert "do-not-write-this-value" not in repr(first)

    changed = _PUBLISHER._capture_provenance(
        repo,
        environment={
            **environment,
            "VITE_PUBLIC_THEME": "light",
            "AKASHIC_MOBILE_WEB_OUT_DIR": "/tmp/mobile-web-b",
        },
        output_dir=Path("/tmp/mobile-web-b"),
    )
    assert changed["build_context_digest"] != second["build_context_digest"]


def test_source_bound_provenance_ignores_inherited_pwd(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    environment = os.environ.copy()
    first = _PUBLISHER._capture_provenance(
        repo,
        environment={**environment, "PWD": str(tmp_path / "inherited-a")},
        output_dir=tmp_path / "output",
    )
    second = _PUBLISHER._capture_provenance(
        repo,
        environment={**environment, "PWD": str(tmp_path / "inherited-b")},
        output_dir=tmp_path / "output",
    )
    assert first["build_context_digest"] == second["build_context_digest"]


def test_run_build_binds_pwd_to_build_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path)
    calls: list[tuple[list[str], Path, dict[str, str]]] = []

    def fake_run(
        command: list[str], *, cwd: Path, env: dict[str, str], check: bool
    ) -> None:
        assert check is True
        calls.append((command, cwd, env))

    monkeypatch.setattr(_PUBLISHER.subprocess, "run", fake_run)
    _PUBLISHER._run_build(
        repo,
        environment={"PATH": os.environ["PATH"], "PWD": str(tmp_path / "inherited")},
        lock_available=True,
    )
    assert len(calls) == 2
    assert all(cwd == repo for _, cwd, _ in calls)
    assert all(env["PWD"] == str(repo.resolve()) for _, _, env in calls)


def test_build_environment_is_controlled_and_clean_publish_revalidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path)
    monkeypatch.setenv("UNTRACKED_BUILD_INPUT", "ignored")
    output = tmp_path / "build-output"
    built = _PUBLISHER._build(repo, output, allow_dirty=False, source_commit=None, stable=True)
    assert "UNTRACKED_BUILD_INPUT" not in _PUBLISHER._build_environment(output)
    manifest, contents = _PUBLISHER._manifest(repo, built, allow_dirty=False)
    store = MobileWebUiStore(tmp_path / "store", server_id="server-1")
    try:
        release = store.publish(manifest, contents, stable=True, preview=False)
        assert release.stable is not None
        assert release.stable.generation_id == manifest.generation_id
    finally:
        store.close()

    monkeypatch.setenv("VITE_PUBLIC_THEME", "light")
    changed_environment = _PUBLISHER._build_environment(output)
    original_environment = dict(changed_environment)
    original_environment.pop("VITE_PUBLIC_THEME", None)
    first = _PUBLISHER._capture_provenance(
        repo,
        environment=original_environment,
        output_dir=output,
    )
    second = _PUBLISHER._capture_provenance(
        repo,
        environment=changed_environment,
        output_dir=output,
    )
    assert first["build_context_digest"] != second["build_context_digest"]


def test_clean_publish_rejects_nondeterministic_artifact_rebuild(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    package = json.loads((repo / "package.json").read_text(encoding="utf-8"))
    package["scripts"]["build:mobile-web"] = (
        "node -e \"const fs=require('fs');"
        "fs.mkdirSync(process.env.AKASHIC_MOBILE_WEB_OUT_DIR,{recursive:true});"
        "fs.writeFileSync(process.env.AKASHIC_MOBILE_WEB_OUT_DIR+'/mobile.html',String(Math.random()))\""
    )
    (repo / "package.json").write_text(json.dumps(package) + "\n", encoding="utf-8")
    _run(repo, "add", "package.json")
    _run(repo, "commit", "-qm", "nondeterministic build")
    output = tmp_path / "nondeterministic-output"
    built = _PUBLISHER._build(repo, output, allow_dirty=False, source_commit=None, stable=True)
    with pytest.raises(RuntimeError, match="artifact 不可复现"):
        _PUBLISHER._manifest(repo, built, allow_dirty=False)


def test_internal_publish_cleans_build_sidecar_on_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _repo(tmp_path)
    workspace = tmp_path / "runtime-workspace"
    monkeypatch.chdir(repo)
    monkeypatch.setenv("PWD", str(repo))
    assert _PUBLISHER.main(
        [
            "publish",
            "--source-repository",
            str(repo),
            "--workspace",
            str(workspace),
            "--server-id",
            "server-1",
            "--stable",
        ]
    ) == 0
    _ = capsys.readouterr()
    assert not list(workspace.glob("mobile-webui-build-*.provenance.json"))


def test_internal_publish_cleans_build_sidecar_on_build_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path)
    workspace = tmp_path / "runtime-workspace"

    def fail_build(source: Path, output: Path, **_kwargs: object) -> Path:
        output.mkdir(parents=True, exist_ok=True)
        output.with_name(output.name + ".provenance.json").write_text("owned", encoding="utf-8")
        raise RuntimeError("synthetic build failure")

    monkeypatch.setattr(_PUBLISHER, "_build", fail_build)
    with pytest.raises(RuntimeError, match="synthetic build failure"):
        _PUBLISHER.main(
            [
                "publish",
                "--source-repository",
                str(repo),
                "--workspace",
                str(workspace),
                "--server-id",
                "server-1",
            ]
        )
    assert not list(workspace.glob("mobile-webui-build-*.provenance.json"))
