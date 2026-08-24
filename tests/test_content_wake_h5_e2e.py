from __future__ import annotations

from copy import deepcopy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from docker.debug.content_wake_h5_e2e import (
    PROTECTED_REQUIRED_FILES,
    PROTECTED_SQLITE_TABLES,
    H5Error,
    _load_manifest,
    _seed_protected_fixture,
    _validate_protected_snapshot,
    run,
)
from docker.debug.wake_v3_provider_e2e import snapshot_protected_workspace

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
    (path / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    (path / "plugin.py").write_text(
        "from .helper import VALUE\n"
        "api_version = 3\n"
        "name = 'h5-fixture'\n"
        "assert VALUE == 'package-import-ok'\n"
        "async def apply(ctx, config):\n"
        "    del ctx, config\n",
        encoding="utf-8",
    )
    (path / "helper.py").write_text("VALUE = 'package-import-ok'\n", encoding="utf-8")
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


def _complete_protected_snapshot(tmp_path: Path) -> dict[str, object]:
    protected = tmp_path / "protected-mutant"
    protected.mkdir()
    _seed_protected_fixture(protected)
    return snapshot_protected_workspace(protected)


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
    assert protected_before["sqlite"] == {
        "drift/drift.db": {
            "integrity": "ok",
            "quick_check": "ok",
            "rows": {"proposals": 1},
        },
        "proactive.db": {
            "integrity": "ok",
            "quick_check": "ok",
            "rows": {"deliveries": 1},
        },
        "sessions.db": {
            "integrity": "ok",
            "quick_check": "ok",
            "rows": {"messages": 1},
        },
        "wake_proactive.db": {
            "integrity": "ok",
            "quick_check": "ok",
            "rows": {"wake_runs": 1},
        },
    }
    assert len(payload["reports"]) == 6
    assert {item["status"] for item in payload["reports"]} == {"passed"}
    installed = payload["trusted_batch"]["installed"][0]
    assert installed["revision"] == revision
    assert Path(installed["installedPath"]).is_relative_to(run_root / "plugin-home")
    receipt = json.loads(
        (run_root / "reports" / "trusted-install.json").read_text(encoding="utf-8")
    )
    assert installed["installedPath"] == receipt["plugins"][0]["installedPath"]
    entrypoints = json.loads(
        (run_root / "reports" / "plugin-entrypoints.json").read_text(
            encoding="utf-8"
        )
    )
    assert entrypoints == {
        "plugins": [
            {
                "entrypoint": "plugin.py",
                "plugin_id": "h5-fixture",
                "status": "passed",
            }
        ],
        "status": "passed",
    }
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


@pytest.mark.parametrize("relative", sorted(PROTECTED_REQUIRED_FILES))
def test_protected_gate_rejects_each_missing_file(
    tmp_path: Path, relative: str
) -> None:
    snapshot = _complete_protected_snapshot(tmp_path)
    files = snapshot["files"]
    assert isinstance(files, dict)
    del files[relative]

    with pytest.raises(H5Error, match="缺少非空fixture"):
        _validate_protected_snapshot(snapshot)


@pytest.mark.parametrize("relative", sorted(PROTECTED_SQLITE_TABLES))
def test_protected_gate_rejects_each_missing_sqlite(
    tmp_path: Path, relative: str
) -> None:
    snapshot = _complete_protected_snapshot(tmp_path)
    sqlite_state = snapshot["sqlite"]
    assert isinstance(sqlite_state, dict)
    del sqlite_state[relative]

    with pytest.raises(H5Error, match="缺少SQLite"):
        _validate_protected_snapshot(snapshot)


@pytest.mark.parametrize("relative,table", tuple(PROTECTED_SQLITE_TABLES.items()))
@pytest.mark.parametrize("mutation", ("zero", "missing"))
def test_protected_gate_rejects_missing_or_empty_required_rows(
    tmp_path: Path, relative: str, table: str, mutation: str
) -> None:
    snapshot = deepcopy(_complete_protected_snapshot(tmp_path))
    sqlite_state = snapshot["sqlite"]
    assert isinstance(sqlite_state, dict)
    database = sqlite_state[relative]
    assert isinstance(database, dict)
    rows = database["rows"]
    assert isinstance(rows, dict)
    if mutation == "zero":
        rows[table] = 0
    else:
        del rows[table]

    with pytest.raises(H5Error, match="SQLite fixture 无效"):
        _validate_protected_snapshot(snapshot)


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
