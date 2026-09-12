from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import closing
from pathlib import Path

import pytest
import toml

from agent.plugins.artifacts import relative_artifact_pointer, write_pointers
from agent.migrations.bundles import MigrationBundleBlocked, MigrationBundleError
from agent.migrations.runner import MigrationRunner
from bootstrap.init_workspace import init_workspace
from bootstrap.workspace_lock import WorkspaceInstanceLock


_PROJECT_ROOT = Path(__file__).parents[1]
_ORIGIN_ID = "20260802_01_yoyo_origin"
_EMPTY_CATALOG = "schema_version = 1\nmigrations = []\n"


def _empty_repo(root: Path) -> Path:
    """Create the current Core shape: an empty Yoyo source and catalog."""

    repo = root / "repo"
    (repo / "migrations/core").mkdir(parents=True)
    (repo / "migrations/catalog.toml").write_text(
        _EMPTY_CATALOG,
        encoding="utf-8",
    )
    return repo


def _write_core_step(
    repo: Path,
    migration_id: str,
    *,
    depends: Sequence[str] = (),
) -> Path:
    """Add a minimal Core migration used only to establish an old source ID."""

    path = repo / "migrations/core" / f"{migration_id}.py"
    path.write_text(
        "from yoyo import step\n"
        f"__depends__ = {set(depends)!r}\n"
        "__transactional__ = False\n"
        "steps = [step('SELECT 1', 'SELECT 1')]\n",
        encoding="utf-8",
    )
    return path


def _write_core_marker_step(
    repo: Path,
    migration_id: str,
    dependency: str,
    marker: Path,
) -> Path:
    """Add a Core step that records execution after its bundle dependency."""

    path = repo / "migrations/core" / f"{migration_id}.py"
    path.write_text(
        "from pathlib import Path\n"
        "from yoyo import step\n"
        f"__depends__ = {{{dependency!r}}}\n"
        "__transactional__ = False\n"
        "\n"
        "def apply(connection):\n"
        "    _ = connection\n"
        f"    Path({str(marker)!r}).write_text('applied', encoding='utf-8')\n"
        "\n"
        "steps = [step(apply)]\n",
        encoding="utf-8",
    )
    return path


def _runner(
    root: Path,
    repo: Path,
    *,
    plugin_dirs: Sequence[Path] = (),
) -> MigrationRunner:
    """Build a runner whose source and cache are isolated from the checkout."""

    return MigrationRunner(
        repo_root=repo,
        config_path=root / "config.toml",
        workspace=root / "workspace",
        plugin_dirs=plugin_dirs,
        installed_cache_root=root / "plugin-cache",
        migration_catalog=repo / "migrations/catalog.toml",
    )


def _applied_ids(ledger: Path) -> tuple[str, ...]:
    if not ledger.exists():
        return ()
    with closing(sqlite3.connect(ledger)) as connection:
        try:
            rows = connection.execute(
                "SELECT migration_id FROM _yoyo_migration ORDER BY rowid"
            ).fetchall()
        except sqlite3.OperationalError as error:
            if "no such table" not in str(error).lower():
                raise
            return ()
    return tuple(str(row[0]) for row in rows)


def _ledger_rows(ledger: Path) -> tuple[tuple[object, ...], ...]:
    with closing(sqlite3.connect(ledger)) as connection:
        return tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT migration_hash, migration_id, applied_at_utc "
                "FROM _yoyo_migration ORDER BY rowid"
            ).fetchall()
        )


def _baseline_ids(ledger: Path) -> tuple[str, ...]:
    with closing(sqlite3.connect(ledger)) as connection:
        row = connection.execute(
            "SELECT baseline_ids FROM _akashic_workspace_origin"
        ).fetchone()
    assert row is not None
    values = json.loads(row[0])
    assert isinstance(values, list)
    return tuple(values)


def _migration_source(
    marker: Path,
    *,
    depends: Sequence[str] = (),
    fail_once: bool = False,
) -> str:
    """Build a tiny plugin-owned step with an observable retry marker."""

    return (
        "from pathlib import Path\n"
        "from yoyo import step\n"
        f"__depends__ = {set(depends)!r}\n"
        "__transactional__ = False\n"
        "\n"
        "def apply(connection):\n"
        "    _ = connection\n"
        f"    marker = Path({str(marker)!r})\n"
        "    if not marker.exists():\n"
        "        marker.write_text('attempted', encoding='utf-8')\n"
        + ("        raise RuntimeError('future bundle failure')\n" if fail_once else "")
        + "\n"
        "steps = [step(apply)]\n"
    )


def _write_bundle(
    parent: Path,
    migrations: Mapping[str, tuple[Sequence[str], str]],
    *,
    bundle_id: str = "future_plugin",
    package_name: str = "future_migrations",
    manifest_name: str | None = None,
) -> Path:
    """Write a complete synthetic v3 artifact accepted by the real loader."""

    artifact = parent / bundle_id
    migration_root = artifact / package_name
    migration_root.mkdir(parents=True, exist_ok=True)
    (artifact / "plugin.py").write_text(
        "# entrypoint is not needed by the migration runner\n",
        encoding="utf-8",
    )
    (migration_root / "__init__.py").write_text("\n", encoding="utf-8")
    for migration_id, (_depends, source) in migrations.items():
        (migration_root / f"{migration_id}.py").write_text(
            source,
            encoding="utf-8",
        )

    files = []
    for path in sorted(migration_root.rglob("*")):
        if not path.is_file() or path.suffix == ".pyc":
            continue
        files.append(
            {
                "path": path.relative_to(migration_root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    migration_rows = []
    for migration_id, (depends, _source) in migrations.items():
        path = migration_root / f"{migration_id}.py"
        migration_rows.append(
            {
                "id": migration_id,
                "path": path.name,
                "depends": list(depends),
                "transactional": False,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    catalog = {
        "schema_version": 1,
        "bundle_id": bundle_id,
        "version": "1.0.0",
        "migration_root": package_name,
        "package_name": package_name,
        "files": files,
        "migrations": migration_rows,
    }
    catalog_path = artifact / "migration.catalog.toml"
    catalog_path.write_text(toml.dumps(catalog), encoding="utf-8")
    (artifact / "akashic.plugin.toml").write_text(
        toml.dumps(
            {
                "schema_version": 1,
                "name": manifest_name or bundle_id,
                "version": "1.0.0",
                "api_version": 3,
                "entrypoint": "plugin.py",
                "migration": {
                    "catalog": "migration.catalog.toml",
                    "catalog_sha256": hashlib.sha256(
                        catalog_path.read_bytes()
                    ).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    return artifact


def test_empty_core_and_catalog_establish_current_baseline(tmp_path: Path) -> None:
    root = tmp_path / "state"
    repo = _empty_repo(tmp_path)
    runner = _runner(root, repo)

    first = runner.run()
    second = runner.run()

    assert (first.state, first.migrations) == ("current", ())
    assert (second.state, second.migrations) == ("current", ())
    assert _baseline_ids(runner.ledger_path) == ()
    assert _applied_ids(runner.ledger_path) == ()


def test_stable_installed_cache_pointer_is_loaded_without_plugin_dir(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    repo = _empty_repo(tmp_path)
    plugin_base = root / "plugin-cache/marketplace/installed_plugin"
    artifacts = plugin_base / ".artifacts"
    stable = _write_bundle(
        artifacts,
        {
            "stable_step": (
                (),
                _migration_source(tmp_path / "stable.marker"),
            )
        },
        bundle_id="stable_release",
        package_name="stable_migrations",
        manifest_name="installed_plugin",
    )
    candidate = _write_bundle(
        artifacts,
        {
            "candidate_step": (
                (),
                _migration_source(tmp_path / "candidate.marker"),
            )
        },
        bundle_id="candidate_release",
        package_name="candidate_migrations",
        manifest_name="installed_plugin",
    )
    write_pointers(
        plugin_base,
        stable=relative_artifact_pointer(plugin_base, stable),
        latest=relative_artifact_pointer(plugin_base, candidate),
    )

    outcome = _runner(root, repo).run()

    assert outcome.migrations == ("stable_step",)
    assert (tmp_path / "stable.marker").read_text(encoding="utf-8") == "attempted"
    assert not (tmp_path / "candidate.marker").exists()
    assert _runner(root, repo).run().state == "current"


def test_future_core_step_runs_after_installed_bundle_is_applied(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    repo = _empty_repo(tmp_path)
    plugin_base = root / "plugin-cache/marketplace/installed_plugin"
    artifacts = plugin_base / ".artifacts"
    bundle = _write_bundle(
        artifacts,
        {
            "installed_step": (
                (),
                _migration_source(tmp_path / "bundle.marker"),
            )
        },
        bundle_id="installed_release",
        package_name="installed_migrations",
        manifest_name="installed_plugin",
    )
    write_pointers(
        plugin_base,
        stable=relative_artifact_pointer(plugin_base, bundle),
        latest=relative_artifact_pointer(plugin_base, bundle),
    )
    runner = _runner(root, repo)
    assert runner.run().migrations == ("installed_step",)
    before_ledger = _ledger_rows(runner.ledger_path)
    bundle_marker = (tmp_path / "bundle.marker").read_bytes()
    core_marker = tmp_path / "core.marker"
    _write_core_marker_step(
        repo,
        "future_core_step",
        "installed_step",
        core_marker,
    )

    outcome = runner.run()

    assert outcome.migrations == ("future_core_step",)
    assert core_marker.read_text(encoding="utf-8") == "applied"
    assert (tmp_path / "bundle.marker").read_bytes() == bundle_marker
    after_ledger = _ledger_rows(runner.ledger_path)
    old_row = next(row for row in before_ledger if row[1] == "installed_step")
    assert next(row for row in after_ledger if row[1] == "installed_step") == old_row


def test_applied_bundle_with_retired_core_dependency_does_not_block(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    repo = _empty_repo(tmp_path)
    core_path = _write_core_step(repo, "retired_core_origin")
    plugin_root = tmp_path / "plugins"
    _write_bundle(
        plugin_root,
        {
            "retired_step": (
                ("retired_core_origin",),
                _migration_source(
                    tmp_path / "retired.marker",
                    depends=("retired_core_origin",),
                ),
            )
        },
        bundle_id="retired_plugin",
        package_name="retired_migrations",
    )
    runner = _runner(root, repo, plugin_dirs=(plugin_root,))
    first = runner.run()
    assert set(first.migrations) == {"retired_core_origin", "retired_step"}

    core_path.unlink()

    second = runner.run()

    assert second.state == "current"
    assert second.migrations == ()


def test_applied_bundle_still_requires_artifact_digest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    repo = _empty_repo(tmp_path)
    _write_core_step(repo, "retired_core_origin")
    plugin_root = tmp_path / "plugins"
    artifact = _write_bundle(
        plugin_root,
        {
            "retired_step": (
                ("retired_core_origin",),
                _migration_source(
                    tmp_path / "retired.marker",
                    depends=("retired_core_origin",),
                ),
            )
        },
        bundle_id="retired_plugin",
        package_name="retired_migrations",
    )
    runner = _runner(root, repo, plugin_dirs=(plugin_root,))
    runner.run()
    step_path = artifact / "retired_migrations/retired_step.py"
    step_path.write_text(
        step_path.read_text(encoding="utf-8") + "\n# artifact drift\n",
        encoding="utf-8",
    )

    with pytest.raises(MigrationBundleError, match="digest"):
        runner.run()


def test_pending_bundle_can_depend_on_applied_source_bundle(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    repo = _empty_repo(tmp_path)
    core_path = _write_core_step(repo, "retired_core_origin")
    old_plugins = tmp_path / "old-plugins"
    _write_bundle(
        old_plugins,
        {
            "old_step": (
                ("retired_core_origin",),
                _migration_source(
                    tmp_path / "old.marker",
                    depends=("retired_core_origin",),
                ),
            )
        },
        bundle_id="old_plugin",
        package_name="old_migrations",
    )
    runner = _runner(root, repo, plugin_dirs=(old_plugins,))
    assert runner.run().migrations == ("retired_core_origin", "old_step")
    core_path.unlink()

    new_plugins = tmp_path / "new-plugins"
    _write_bundle(
        new_plugins,
        {
            "new_step": (
                ("old_step",),
                _migration_source(
                    tmp_path / "new.marker",
                    depends=("old_step",),
                ),
            )
        },
        bundle_id="new_plugin",
        package_name="new_migrations",
    )

    outcome = _runner(
        root,
        repo,
        plugin_dirs=(old_plugins, new_plugins),
    ).run()

    assert outcome.migrations == ("new_step",)


def test_pending_bundle_dependency_only_in_ledger_is_blocked(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    repo = _empty_repo(tmp_path)
    _write_core_step(repo, "retired_core_origin")
    old_plugins = tmp_path / "old-plugins"
    _write_bundle(
        old_plugins,
        {
            "old_step": (
                ("retired_core_origin",),
                _migration_source(
                    tmp_path / "old.marker",
                    depends=("retired_core_origin",),
                ),
            )
        },
        bundle_id="old_plugin",
        package_name="old_migrations",
    )
    runner = _runner(root, repo, plugin_dirs=(old_plugins,))
    runner.run()
    before = _ledger_rows(runner.ledger_path)

    new_plugins = tmp_path / "new-plugins"
    _write_bundle(
        new_plugins,
        {
            "new_step": (
                ("old_step",),
                _migration_source(
                    tmp_path / "new.marker",
                    depends=("old_step",),
                ),
            )
        },
        bundle_id="new_plugin",
        package_name="new_migrations",
    )

    with pytest.raises(MigrationBundleBlocked) as raised:
        _runner(root, repo, plugin_dirs=(new_plugins,)).run()

    assert raised.value.bundle_id == "new_plugin"
    assert raised.value.migration_ids == ("new_step",)
    assert raised.value.missing_dependencies == ("old_step",)
    assert _ledger_rows(runner.ledger_path) == before


def test_pending_dependency_closure_requires_a_current_core_source(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    repo = _empty_repo(tmp_path)
    plugin_root = tmp_path / "plugins"
    _write_bundle(
        plugin_root,
        {
            "intermediate_step": (
                ("future_core_step",),
                _migration_source(
                    tmp_path / "intermediate.marker",
                    depends=("future_core_step",),
                ),
            ),
            "leaf_step": (
                ("intermediate_step",),
                _migration_source(
                    tmp_path / "leaf.marker",
                    depends=("intermediate_step",),
                ),
            ),
        },
        bundle_id="future_plugin",
        package_name="future_migrations",
    )

    with pytest.raises(MigrationBundleBlocked) as raised:
        _runner(root, repo, plugin_dirs=(plugin_root,)).run()

    assert raised.value.missing_dependencies == ("future_core_step",)
    assert not (root / "workspace/migrations.sqlite3").exists()


def test_existing_ledger_rows_for_deleted_sources_are_preserved(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    repo = _empty_repo(tmp_path)
    runner = _runner(root, repo)
    runner.run()

    with closing(sqlite3.connect(runner.ledger_path)) as connection:
        connection.execute(
            "INSERT INTO _yoyo_migration "
            "(migration_hash, migration_id, applied_at_utc) VALUES (?, ?, ?)",
            ("retired-hash", "retired_historical_id", "2026-09-01T00:00:00Z"),
        )
        connection.commit()
    before = _ledger_rows(runner.ledger_path)

    outcome = runner.run()

    assert outcome.state == "current"
    assert _ledger_rows(runner.ledger_path) == before
    assert _applied_ids(runner.ledger_path) == ("retired_historical_id",)


def test_future_bundle_failure_is_pending_and_retries(tmp_path: Path) -> None:
    root = tmp_path / "state"
    repo = _empty_repo(tmp_path)
    plugin_root = tmp_path / "plugins"
    marker = tmp_path / "failure.marker"
    _write_bundle(
        plugin_root,
        {"future_step": ((), _migration_source(marker, fail_once=True))},
    )
    runner = _runner(root, repo, plugin_dirs=(plugin_root,))

    with pytest.raises(RuntimeError, match="future bundle failure"):
        runner.run()
    assert marker.read_text(encoding="utf-8") == "attempted"
    assert _applied_ids(runner.ledger_path) == ()

    retry = runner.run()
    assert retry.migrations == ("future_step",)
    assert runner.run().state == "current"


def test_future_bundle_can_append_a_new_step_after_sibling_ran(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    repo = _empty_repo(tmp_path)
    plugin_root = tmp_path / "plugins"
    first_marker = tmp_path / "first.marker"
    second_marker = tmp_path / "second.marker"
    _write_bundle(
        plugin_root,
        {"future_first": ((), _migration_source(first_marker))},
    )
    runner = _runner(root, repo, plugin_dirs=(plugin_root,))
    assert runner.run().migrations == ("future_first",)

    _write_bundle(
        plugin_root,
        {
            "future_first": ((), _migration_source(first_marker)),
            "future_second": (
                ("future_first",),
                _migration_source(second_marker, depends=("future_first",)),
            ),
        },
    )
    assert runner.run().migrations == ("future_second",)
    assert _applied_ids(runner.ledger_path) == (
        "future_first",
        "future_second",
    )


def test_missing_historical_dependency_blocks_without_changing_old_ledger(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    repo = _empty_repo(tmp_path)
    runner = _runner(root, repo)
    runner.run()
    with closing(sqlite3.connect(runner.ledger_path)) as connection:
        connection.execute(
            "INSERT INTO _yoyo_migration "
            "(migration_hash, migration_id, applied_at_utc) VALUES (?, ?, ?)",
            ("retired-hash", "retired_historical_id", "2026-09-01T00:00:00Z"),
        )
        connection.commit()
    before = _ledger_rows(runner.ledger_path)

    plugin_root = tmp_path / "plugins"
    _write_bundle(
        plugin_root,
        {
            "future_step": (
                ("retired_historical_id",),
                _migration_source(
                    tmp_path / "should-not-run",
                    depends=("retired_historical_id",),
                ),
            )
        },
    )

    with pytest.raises(MigrationBundleBlocked) as raised:
        _runner(root, repo, plugin_dirs=(plugin_root,)).run()

    assert raised.value.missing_dependencies == ("retired_historical_id",)
    assert _ledger_rows(runner.ledger_path) == before


def test_checkout_plugin_is_ignored_without_explicit_plugin_dir(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    repo = _empty_repo(tmp_path)
    checkout_plugins = repo / "plugins"
    marker = tmp_path / "checkout.marker"
    _write_bundle(
        checkout_plugins,
        {"checkout_step": ((), _migration_source(marker))},
        bundle_id="checkout_plugin",
        package_name="checkout_migrations",
    )

    runner = _runner(root, repo)
    assert runner.run().state == "current"
    assert not marker.exists()

    explicit = _runner(root, repo, plugin_dirs=(checkout_plugins,))
    assert explicit.run().migrations == ("checkout_step",)
    assert marker.exists()


def test_runner_supplies_yoyo_identity_without_os_username(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in ("LOGNAME", "USER", "LNAME", "USERNAME"):
        monkeypatch.delenv(key, raising=False)

    def get_user() -> str:
        username = os.environ.get("USER")
        if not username:
            raise OSError("No username set in the environment")
        return username

    monkeypatch.setattr("yoyo.backends.base.getpass.getuser", get_user)
    repo = _empty_repo(tmp_path)

    outcome = _runner(tmp_path / "state", repo).run()

    assert outcome.state == "current"
    assert "USER" not in os.environ


def test_workspace_lock_prevents_runner_and_leaves_no_ledger(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    runner = _runner(root, _empty_repo(tmp_path))
    lock = WorkspaceInstanceLock(runner.workspace)
    lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="workspace 已由其他 runtime 占用"):
            runner.run()
    finally:
        lock.release()

    assert not runner.ledger_path.exists()


def test_ledger_uri_accepts_workspace_path_with_uri_characters(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state with # and ?"
    runner = _runner(root, _empty_repo(tmp_path))

    runner.run()

    assert "%23" in runner._ledger_uri()
    assert "%3F" in runner._ledger_uri()
    assert runner.ledger_path.is_file()


def test_runtime_data_created_after_baseline_does_not_replay_migrations(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    runner = _runner(root, _empty_repo(tmp_path))
    runner.run()
    sessions = runner.workspace / "sessions.db"
    sessions.write_bytes(b"runtime-data")

    outcome = runner.run()

    assert outcome.state == "current"
    assert outcome.migrations == ()
    assert sessions.read_bytes() == b"runtime-data"


def test_init_workspace_records_baseline_before_runtime_data(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"

    init_workspace(config_path=config, workspace=workspace)
    ledger = workspace / "migrations.sqlite3"
    baseline = _baseline_ids(ledger)
    assert not (workspace / "sessions.db").exists()

    sessions = workspace / "sessions.db"
    sessions.write_bytes(b"runtime-data")
    before_ledger = ledger.read_bytes()
    init_workspace(config_path=config, workspace=workspace)

    assert ledger.read_bytes() == before_ledger
    assert _baseline_ids(ledger) == baseline
    assert sessions.read_bytes() == b"runtime-data"


def test_core_only_cli_restarts_after_creating_runtime_data(
    tmp_path: Path,
) -> None:
    """CLI startup must not discover checkout plugins implicitly."""

    config = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"
    environment = dict(os.environ)
    environment["AKASHIC_PLUGIN_HOME"] = str(tmp_path / "plugin-home")
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(_PROJECT_ROOT / "sdk/python/src"), str(_PROJECT_ROOT))
    )
    arguments = ["--config", str(config), "--workspace", str(workspace)]
    commands = (
        ["init", *arguments],
        ["--inspect-modules", *arguments],
        ["--inspect-modules", *arguments],
    )
    for command in commands:
        result = subprocess.run(
            [sys.executable, str(_PROJECT_ROOT / "main.py"), *command],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=40,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    assert (workspace / "migrations.sqlite3").is_file()
    assert set(_applied_ids(workspace / "migrations.sqlite3")) <= {_ORIGIN_ID}
