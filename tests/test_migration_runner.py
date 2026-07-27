from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agent.migrations.runner import MigrationRunner, _MigrationLock

_BUNDLE = """\
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
"""


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


def _ordered_bundle(name: str) -> str:
    return _BUNDLE.replace(
        'marker = args.workspace / "bundle-applied"',
        f'marker = args.workspace / "bundle-{name}-applied"\n'
        'order = args.workspace / "migration-order.log"',
    ).replace(
        'marker.write_text("applied", encoding="utf-8")',
        'marker.write_text("applied", encoding="utf-8")\n'
        f'    with order.open("a", encoding="utf-8") as stream: stream.write("{name}\\n")',
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


def test_veda_only_workspace_without_config_uses_legacy_adoption(
    tmp_path: Path,
) -> None:
    repo, _, head = _repository(tmp_path)
    state = tmp_path / "state"
    veda = state / "workspace/memory/veda.md"
    veda.parent.mkdir(parents=True)
    veda.write_text("custom veda\n", encoding="utf-8")

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


def test_current_state_without_cursor_verifies_without_rewriting(
    tmp_path: Path,
) -> None:
    repo, _, head = _repository(tmp_path)
    state = tmp_path / "state"
    workspace = state / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "bundle-applied").write_text("applied", encoding="utf-8")
    config = state / "config.toml"
    original = b"current = true\n"
    config.write_bytes(original)

    outcome = _runner(repo, state).run()

    assert outcome.state == "migrated"
    assert config.read_bytes() == original
    assert not (state / "config.toml.migration-backups").exists()
    assert (state / "config.toml.migration-cursor").read_text().strip() == head


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


def test_apply_failure_keeps_cursor_at_baseline(tmp_path: Path) -> None:
    failing = _BUNDLE.replace(
        "    assert args.backup_dir is not None\n"
        "    args.backup_dir.mkdir(parents=True)\n"
        "    marker.parent.mkdir(parents=True, exist_ok=True)\n"
        '    marker.write_text("applied", encoding="utf-8")',
        '    raise SystemExit("forced apply failure")',
    )
    repo, baseline, _ = _repository(tmp_path, failing)
    state = tmp_path / "state"
    state.mkdir()
    (state / "config.toml").write_text("legacy = true\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="forced apply failure"):
        _runner(repo, state).run()

    assert (state / "config.toml.migration-cursor").read_text().strip() == baseline
    assert not (state / "workspace/bundle-applied").exists()


def test_blocked_assessment_keeps_cursor_at_baseline(tmp_path: Path) -> None:
    blocked = _BUNDLE.replace(
        'print(json.dumps({"status": "satisfied" if marker.exists() else "needed"}))',
        'print(json.dumps({"status": "blocked", "reason": "unknown lineage"}))',
    )
    repo, baseline, _ = _repository(tmp_path, blocked)
    state = tmp_path / "state"
    state.mkdir()
    (state / "config.toml").write_text("legacy = true\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unknown lineage"):
        _runner(repo, state).run()

    assert (state / "config.toml.migration-cursor").read_text().strip() == baseline


def test_cursor_write_failure_retries_without_reapplying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, baseline, head = _repository(tmp_path, _ordered_bundle("once"))
    state = tmp_path / "state"
    state.mkdir()
    (state / "config.toml").write_text("legacy = true\n", encoding="utf-8")
    runner = _runner(repo, state)
    original_write = runner._write_cursor
    failed = False

    def fail_once(revision: str) -> None:
        nonlocal failed
        if revision == head and not failed:
            failed = True
            raise RuntimeError("forced cursor write failure")
        original_write(revision)

    monkeypatch.setattr(runner, "_write_cursor", fail_once)
    with pytest.raises(RuntimeError, match="forced cursor write failure"):
        runner.run()
    assert runner.cursor_path.read_text().strip() == baseline

    monkeypatch.setattr(runner, "_write_cursor", original_write)
    outcome = runner.run()

    assert outcome.state == "migrated"
    assert runner.cursor_path.read_text().strip() == head
    assert (state / "workspace/migration-order.log").read_text() == "once\n"


def test_code_only_commit_advances_cursor_without_running_bundle(
    tmp_path: Path,
) -> None:
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


def test_multiple_migration_commits_run_in_first_parent_order(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _ = _git(repo, "init", "-b", "main")
    _ = _git(repo, "config", "user.email", "test@example.com")
    _ = _git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("baseline\n", encoding="utf-8")
    _ = _git(repo, "add", "README.md")
    _ = _git(repo, "commit", "-m", "baseline")
    baseline = _git(repo, "rev-parse", "HEAD")

    commits: list[str] = []
    for name in ("first", "second"):
        bundle = repo / "migrations" / name
        bundle.mkdir(parents=True)
        if name == "first":
            (repo / "migrations" / ".root").write_text(
                f"{baseline}\n", encoding="ascii"
            )
        (bundle / "migration.py").write_text(_ordered_bundle(name), encoding="utf-8")
        _ = _git(repo, "add", "migrations")
        _ = _git(repo, "commit", "-m", f"add {name} migration")
        commits.append(_git(repo, "rev-parse", "HEAD"))

    state = tmp_path / "state"
    state.mkdir()
    (state / "config.toml").write_text("legacy = true\n", encoding="utf-8")

    outcome = _runner(repo, state).run()

    assert outcome.commits == tuple(commits)
    assert (state / "workspace/migration-order.log").read_text() == "first\nsecond\n"


def test_diverged_cursor_fails_without_changing_cursor(tmp_path: Path) -> None:
    repo, baseline, migration_head = _repository(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    cursor = state / "config.toml.migration-cursor"
    cursor.write_text(f"{migration_head}\n", encoding="ascii")
    _ = _git(repo, "switch", "-c", "diverged", baseline)
    (repo / "README.md").write_text("diverged\n", encoding="utf-8")
    _ = _git(repo, "add", "README.md")
    _ = _git(repo, "commit", "-m", "diverged change")

    with pytest.raises(RuntimeError, match="禁止自动降级或跨分支迁移"):
        _runner(repo, state).run()

    assert cursor.read_text(encoding="ascii").strip() == migration_head


def test_shallow_history_missing_baseline_fails_loud(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source, baseline, _ = _repository(source_root)
    shallow = tmp_path / "shallow"
    _ = subprocess.run(
        ["git", "clone", "--depth", "1", source.as_uri(), str(shallow)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    assert (
        subprocess.run(
            ["git", "-C", str(shallow), "cat-file", "-e", f"{baseline}^{{commit}}"],
            check=False,
        ).returncode
        != 0
    )
    state = tmp_path / "state"
    state.mkdir()
    (state / "config.toml").write_text("legacy = true\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Git 命令失败"):
        _runner(shallow, state).run()

    assert not (state / "workspace/bundle-applied").exists()
    assert not (state / "config.toml.migration-cursor").exists()


def test_concurrent_runner_cannot_bypass_config_lock(tmp_path: Path) -> None:
    repo, _, _ = _repository(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (state / "config.toml").write_text("legacy = true\n", encoding="utf-8")
    runner = _runner(repo, state)

    with _MigrationLock(runner.lock_path):
        with pytest.raises(RuntimeError, match="配置迁移已由其他进程执行"):
            runner.run()

    assert not runner.cursor_path.exists()
