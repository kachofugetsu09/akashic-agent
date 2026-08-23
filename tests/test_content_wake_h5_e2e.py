from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from docker.debug.content_wake_h5_e2e import _load_manifest, run

ROOT = Path(__file__).resolve().parents[1]


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _plugin_repository(path: Path) -> tuple[Path, str]:
    path.mkdir()
    _ = _git(path, "init", "--quiet")
    _ = _git(path, "config", "user.email", "fixture@example.com")
    _ = _git(path, "config", "user.name", "Fixture")
    (path / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        'name = "h5-fixture"\n'
        'version = "1.0.0"\n'
        "api_version = 3\n"
        'entrypoint = "plugin.py"\n'
        "\n[[python]]\n"
        'requirements = "requirements.txt"\n',
        encoding="utf-8",
    )
    (path / "requirements.txt").write_text("requests==2.32.5\n", encoding="utf-8")
    (path / "plugin.py").write_text(
        "api_version = 3\n"
        "name = 'h5-fixture'\n"
        "async def apply(ctx, config):\n"
        "    del ctx, config\n",
        encoding="utf-8",
    )
    tests = path / "tests"
    tests.mkdir()
    (tests / "test_plugin.py").write_text(
        "import json\n"
        "import os\n"
        "import subprocess\n"
        "def test_installed_fixture():\n"
        "    python = os.environ['AKASHIC_PLUGIN_FIXTURE_PYTHON']\n"
        "    code = \"import json,requests;print(json.dumps({'version':requests.__version__,'path':requests.__file__}))\"\n"
        "    result = subprocess.run([python, '-c', code], check=True, capture_output=True, text=True)\n"
        "    receipt = json.loads(result.stdout)\n"
        "    assert receipt['version'] == '2.32.5'\n"
        "    assert '/.venv/' in receipt['path']\n",
        encoding="utf-8",
    )
    _ = _git(path, "add", ".")
    _ = _git(path, "commit", "--quiet", "-m", "test: H5 fixture")
    return path, _git(path, "rev-parse", "HEAD")


def _contracts(tmp_path: Path, repository: Path, revision: str) -> Path:
    lock = tmp_path / "interop-lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "core_contract": _git(ROOT, "rev-parse", "HEAD"),
                "core_cases": ["tests/test_content_source_interop_gate.py"],
                "coexistence": [],
                "cross_repo": [],
                "plugins": [
                    {
                        "id": "h5-fixture",
                        "repository": str(repository),
                        "branch": "main",
                        "resolved_sha": revision,
                        "pull_request": None,
                        "role": "fixture",
                        "atoms": ["content.source.v1"],
                        "test_cwd": ".",
                        "cases": ["tests/test_plugin.py"],
                    }
                ],
                "pending": [],
                "retired": [],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "interop_lock": str(lock),
                "suites": [
                    {
                        "id": "core_boundary",
                        "cases": [
                            "tests/test_content_source_interop_gate.py::test_content_logical_state_detects_empty_submission"
                        ],
                    }
                ],
                "real_provider": {
                    "status": "PENDING",
                    "reason": "fixture",
                    "command": ["provider", "<RUN_ROOT>"],
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_h5_runner_uses_trusted_receipt_paths_and_composes_real_reports(
    tmp_path: Path,
) -> None:
    repository, revision = _plugin_repository(tmp_path / "plugin-source")
    manifest = _contracts(tmp_path, repository, revision)
    protected = tmp_path / "protected"
    protected.mkdir()
    run_root = tmp_path / "run"

    index_path = run(
        run_root=run_root,
        protected_workspace=protected.resolve(),
        manifest_path=manifest,
        seed_protected_fixture=True,
    )

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["status"] == "deterministic_passed"
    assert payload["real_provider"]["status"] == "PENDING"
    assert payload["protected_workspace"]["status"] == "unchanged"
    protected_before = payload["protected_workspace"]["before"]
    assert protected_before == payload["protected_workspace"]["after"]
    assert set(protected_before["files"]) == {
        "PROACTIVE_CONTEXT.md",
        "drift/drift.db",
        "proactive.db",
        "proactive_pending.md",
        "proactive_quota.json",
        "sessions.db",
        "wake_proactive.db",
    }
    for item in protected_before["files"].values():
        assert item["inode"] > 0
        assert item["size"] > 0
        assert len(item["sha256"]) == 64
    for item in protected_before["sqlite"].values():
        assert item["integrity"] == "ok"
        assert item["quick_check"] == "ok"
        assert sum(item["rows"].values()) == 1
    assert len(payload["reports"]) == 5
    assert {item["status"] for item in payload["reports"]} == {"passed"}
    installed = payload["trusted_batch"]["installed"][0]
    assert installed["revision"] == revision
    assert Path(installed["installedPath"]).is_relative_to(run_root / "plugin-home")
    receipt = json.loads(
        (run_root / "reports" / "trusted-install.json").read_text(encoding="utf-8")
    )
    assert installed["installedPath"] == receipt["plugins"][0]["installedPath"]
    artifact = Path(installed["installedPath"])
    assert list(artifact.glob(".venv/lib/python*/site-packages/urllib3"))
    bindings = json.loads(
        (run_root / "reports" / "fixture-runtime-bindings.json").read_text(
            encoding="utf-8"
        )
    )
    runtime = bindings["runtimes"][0]
    assert Path(runtime["pytest_path"]).is_relative_to(
        run_root / "home" / "fixture-layer"
    )
    assert Path(runtime["artifact_dependency_path"]).is_relative_to(artifact)
    assert runtime["core_only_black"] == "unavailable"
    assert payload["core"]["head"] == _git(ROOT, "rev-parse", "HEAD")


def test_h5_runner_rejects_empty_protected_workspace(tmp_path: Path) -> None:
    repository, revision = _plugin_repository(tmp_path / "plugin-source")
    manifest = _contracts(tmp_path, repository, revision)
    protected = tmp_path / "protected"
    protected.mkdir()
    run_root = tmp_path / "run"

    with pytest.raises(RuntimeError, match="缺少非空fixture"):
        run(
            run_root=run_root,
            protected_workspace=protected.resolve(),
            manifest_path=manifest,
        )

    assert not any(protected.iterdir())
    assert not any((run_root / "plugin-home").iterdir())
    assert not (run_root / "reports" / "trusted-install.json").exists()


def test_manifest_requires_pending_real_provider_and_existing_cases(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "interop_lock": str(tmp_path / "missing.json"),
                "suites": [{"id": "missing", "cases": ["tests/not-there.py"]}],
                "real_provider": {"status": "passed"},
            }
        ),
        encoding="utf-8",
    )

    try:
        _load_manifest(manifest)
    except RuntimeError as error:
        assert "PENDING" in str(error)
    else:
        raise AssertionError("real provider status drift must be rejected")
