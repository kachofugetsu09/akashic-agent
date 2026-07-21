from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agent.migrations.runner import MigrationRunner


_BUNDLE = '''\
from __future__ import annotations

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("action")
parser.add_argument("--config")
parser.add_argument("--workspace", type=Path, required=True)
parser.add_argument("--migration-commit")
parser.add_argument("--backup-dir", type=Path)
args = parser.parse_args()
marker = args.workspace / "bundle-applied"

if args.action == "assess":
    print(json.dumps({"status": "satisfied" if marker.exists() else "needed"}))
elif args.action == "apply":
    assert args.backup_dir is not None
    args.backup_dir.mkdir(parents=True)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("applied", encoding="utf-8")
elif args.action == "verify" and not marker.exists():
    raise SystemExit("marker missing")
'''


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _repository(tmp_path: Path, bundle: str = _BUNDLE) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _ = _git(repo, "init", "-b", "main")
    _ = _git(repo, "config", "user.email", "test@example.com")
    _ = _git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("baseline\n", encoding="utf-8")
    _ = _git(repo, "add", "README.md")
    _ = _git(repo, "commit", "-m", "baseline")
    baseline = _git(repo, "rev-parse", "HEAD")

    migration = repo / "migrations" / "example"
    migration.mkdir(parents=True)
    (repo / "migrations" / ".root").write_text(f"{baseline}\n", encoding="ascii")
    (migration / "migration.py").write_text(bundle, encoding="utf-8")
    _ = _git(repo, "add", "migrations")
    _ = _git(repo, "commit", "-m", "add migration")
    return repo, baseline, _git(repo, "rev-parse", "HEAD")


def _runner(repo: Path, root: Path) -> MigrationRunner:
    return MigrationRunner(
        repo_root=repo,
        config_path=root / "config.toml",
        workspace=root / "workspace",
    )


def test_legacy_config_starts_at_baseline_and_runs_bundle(tmp_path: Path) -> None:
    repo, baseline, head = _repository(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (state / "config.toml").write_text("legacy = true\n", encoding="utf-8")

    outcome = _runner(repo, state).run()

    assert outcome.state == "migrated"
    assert outcome.commits == (head,)
    assert (state / "workspace" / "bundle-applied").read_text() == "applied"
    cursor = state / "config.toml.migration-cursor"
    assert cursor.read_text(encoding="ascii").strip() == head
    assert baseline != head


def test_fresh_installation_skips_history_until_initialized(tmp_path: Path) -> None:
    repo, _, head = _repository(tmp_path)
    state = tmp_path / "fresh"

    outcome = _runner(repo, state).run()

    assert outcome.state == "fresh"
    assert outcome.head == head
    assert not (state / "config.toml.migration-cursor").exists()
    assert not (state / "workspace" / "bundle-applied").exists()


def test_durable_workspace_without_config_uses_legacy_adoption(tmp_path: Path) -> None:
    repo, _, head = _repository(tmp_path)
    state = tmp_path / "state"
    workspace = state / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "sessions.db").write_bytes(b"existing")

    outcome = _runner(repo, state).run()

    assert outcome.state == "migrated"
    assert (state / "config.toml.migration-cursor").read_text().strip() == head


def test_current_cursor_does_not_scan_migration_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _, head = _repository(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (state / "config.toml.migration-cursor").write_text(f"{head}\n", encoding="ascii")
    runner = _runner(repo, state)

    def fail(*_args: object) -> list[str]:
        raise AssertionError("stable path scanned migration history")

    monkeypatch.setattr(runner, "_migration_commits", fail)

    assert runner.run().state == "current"


def test_verify_failure_keeps_cursor_at_baseline(tmp_path: Path) -> None:
    failing = _BUNDLE.replace(
        'elif args.action == "verify" and not marker.exists():',
        'elif args.action == "verify":',
    ).replace(
        'raise SystemExit("marker missing")',
        'raise SystemExit("forced verify failure")',
    )
    repo, baseline, _ = _repository(tmp_path, failing)
    state = tmp_path / "state"
    state.mkdir()
    (state / "config.toml").write_text("legacy = true\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="forced verify failure"):
        _runner(repo, state).run()

    assert (state / "config.toml.migration-cursor").read_text().strip() == baseline


def test_code_only_commit_advances_cursor_without_running_bundle(tmp_path: Path) -> None:
    repo, _, migration_head = _repository(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (state / "config.toml").write_text("legacy = true\n", encoding="utf-8")
    runner = _runner(repo, state)
    _ = runner.run()
    marker = state / "workspace" / "bundle-applied"
    marker.write_text("unchanged", encoding="utf-8")

    (repo / "README.md").write_text("new code\n", encoding="utf-8")
    _ = _git(repo, "add", "README.md")
    _ = _git(repo, "commit", "-m", "code only")
    head = _git(repo, "rev-parse", "HEAD")

    outcome = runner.run()

    assert outcome.state == "migrated"
    assert outcome.commits == ()
    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert migration_head != head
    assert (state / "config.toml.migration-cursor").read_text().strip() == head


def test_merge_commit_discovers_bundle_from_first_parent(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _ = _git(repo, "init", "-b", "main")
    _ = _git(repo, "config", "user.email", "test@example.com")
    _ = _git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("baseline\n", encoding="utf-8")
    _ = _git(repo, "add", "README.md")
    _ = _git(repo, "commit", "-m", "baseline")
    baseline = _git(repo, "rev-parse", "HEAD")

    _ = _git(repo, "switch", "-c", "migration")
    migration = repo / "migrations" / "example"
    migration.mkdir(parents=True)
    (repo / "migrations" / ".root").write_text(f"{baseline}\n", encoding="ascii")
    (migration / "migration.py").write_text(_BUNDLE, encoding="utf-8")
    _ = _git(repo, "add", "migrations")
    _ = _git(repo, "commit", "-m", "add migration")
    _ = _git(repo, "switch", "main")
    _ = _git(repo, "merge", "--no-ff", "migration", "-m", "merge migration")
    head = _git(repo, "rev-parse", "HEAD")

    state = tmp_path / "state"
    state.mkdir()
    (state / "config.toml").write_text("legacy = true\n", encoding="utf-8")
    outcome = _runner(repo, state).run()

    assert outcome.commits == (head,)
    assert (state / "workspace" / "bundle-applied").read_text() == "applied"
