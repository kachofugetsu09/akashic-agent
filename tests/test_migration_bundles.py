from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agent.migrations.bundles import (
    MigrationBundleBlocked,
    MigrationBundleError,
    discover_migration_bundles,
)
from agent.migrations.runner import MigrationRunner


def _write_core(repo: Path, migration_id: str) -> None:
    root = repo / "migrations/core"
    root.mkdir(parents=True)
    (root / f"{migration_id}.py").write_text(
        "from yoyo import step\n"
        "__depends__ = set()\n"
        "__transactional__ = False\n"
        "steps = [step('CREATE TABLE IF NOT EXISTS core_marker (value TEXT)', "
        "'DROP TABLE core_marker')]\n",
        encoding="utf-8",
    )


def _write_bundle(
    parent: Path,
    *,
    bundle_id: str = "legacy_upgrade",
    migration_id: str = "bundle_step",
    depends: tuple[str, ...] = (),
    source: str | None = None,
) -> Path:
    artifact = parent / bundle_id
    steps = artifact / "migration_steps"
    steps.mkdir(parents=True)
    (artifact / "plugin.py").write_text(
        f"name = {bundle_id!r}\nversion = '1.0.0'\napi_version = 3\n",
        encoding="utf-8",
    )
    step_path = steps / f"{migration_id}.py"
    step_path.write_text(
        source
        or (
            "from yoyo import step\n"
            "from agent.migrations.context import current_migration_context\n"
            "from pathlib import Path\n"
            "__depends__ = "
            + repr(set(depends))
            + "\n__transactional__ = False\n"
            "def apply(connection):\n"
            "    _ = connection\n"
            "    context = current_migration_context()\n"
            "    marker = context.workspace / 'bundle.marker'\n"
            "    marker.write_text('applied', encoding='utf-8')\n"
            "steps = [step(apply)]\n"
        ),
        encoding="utf-8",
    )
    source_hash = hashlib.sha256(step_path.read_bytes()).hexdigest()
    catalog = (
        "schema_version = 1\n"
        f"bundle_id = {bundle_id!r}\n"
        "version = '1.0.0'\n"
        "migration_root = 'migration_steps'\n\n"
        "[[migrations]]\n"
        f"id = {migration_id!r}\n"
        f"path = {step_path.name!r}\n"
        f"depends = {list(depends)!r}\n"
        "transactional = false\n"
        f"sha256 = {source_hash!r}\n"
    ).encode("utf-8")
    catalog_path = artifact / "migration.catalog.toml"
    catalog_path.write_bytes(catalog)
    catalog_hash = hashlib.sha256(catalog).hexdigest()
    (artifact / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        f"name = {bundle_id!r}\n"
        "version = '1.0.0'\n"
        "api_version = 3\n"
        "entrypoint = 'plugin.py'\n\n"
        "[migration]\n"
        "catalog = 'migration.catalog.toml'\n"
        f"catalog_sha256 = {catalog_hash!r}\n",
        encoding="utf-8",
    )
    return artifact


def _runner(root: Path, *, plugin_dirs: tuple[Path, ...] = ()) -> MigrationRunner:
    return MigrationRunner(
        repo_root=root / "repo",
        config_path=root / "config.toml",
        workspace=root / "workspace",
        plugin_dirs=plugin_dirs,
        installed_cache_root=root / "empty-plugin-cache",
    )


def test_external_bundle_is_combined_with_core_under_one_ledger(tmp_path: Path) -> None:
    root = tmp_path / "state"
    repo = root / "repo"
    core_id = "test_core_bundle_origin"
    _write_core(repo, core_id)
    plugin_parent = root / "plugins"
    artifact = _write_bundle(plugin_parent, depends=(core_id,))

    outcome = _runner(root, plugin_dirs=(plugin_parent,)).run()

    assert outcome.migrations == (core_id, "bundle_step")
    assert (root / "workspace/bundle.marker").read_text() == "applied"
    assert (root / "workspace/migrations.sqlite3").is_file()

    second = _runner(root, plugin_dirs=(plugin_parent,)).run()
    assert second.state == "current"
    assert second.migrations == ()


def test_core_does_not_scan_checkout_plugin_directory_implicitly(tmp_path: Path) -> None:
    root = tmp_path / "state"
    repo = root / "repo"
    _write_core(repo, "test_core_only_origin")
    _write_bundle(repo / "plugins")

    outcome = _runner(root).run()

    assert outcome.migrations == ("test_core_only_origin",)
    assert not (root / "workspace/bundle.marker").exists()


def test_missing_pending_bundle_is_typed_and_does_not_create_ledger(tmp_path: Path) -> None:
    root = tmp_path / "state"
    repo = root / "repo"
    _write_core(repo, "test_core_missing_owner_origin")
    catalog = repo / "migrations/catalog.toml"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(
        "schema_version = 1\n\n"
        "[[migrations]]\n"
        "id = 'test_missing_owner_step'\n"
        "bundle = 'legacy_upgrade'\n"
        "depends = ['test_core_missing_owner_origin']\n"
        "transactional = false\n",
        encoding="utf-8",
    )

    with pytest.raises(MigrationBundleBlocked) as raised:
        _runner(root).run()

    error = raised.value
    assert error.code == "migration_blocked"
    assert error.bundle_id == "legacy_upgrade"
    assert error.migration_ids == ("test_missing_owner_step",)
    assert not (root / "workspace/migrations.sqlite3").exists()


def test_bundle_source_digest_drift_fails_before_ledger_write(tmp_path: Path) -> None:
    root = tmp_path / "state"
    repo = root / "repo"
    _write_core(repo, "test_core_digest_origin")
    plugin_parent = root / "plugins"
    artifact = _write_bundle(plugin_parent, depends=("test_core_digest_origin",))
    step_path = artifact / "migration_steps/bundle_step.py"
    step_path.write_text(step_path.read_text(encoding="utf-8") + "\n# drift\n")

    with pytest.raises(MigrationBundleError, match="source digest"):
        _runner(root, plugin_dirs=(plugin_parent,)).run()

    assert not (root / "workspace/migrations.sqlite3").exists()


def test_bundle_rejects_current_plugin_namespace_import(tmp_path: Path) -> None:
    root = tmp_path / "state"
    plugin_parent = root / "plugins"
    _write_bundle(
        plugin_parent,
        source=(
            "from plugins.some_active_owner import migrate\n"
            "from yoyo import step\n"
            "steps = [step('SELECT 1', 'SELECT 1')]\n"
        ),
    )

    with pytest.raises(MigrationBundleError, match="runtime/插件 namespace"):
        discover_migration_bundles(
            plugin_dirs=(plugin_parent,),
            installed_cache_root=tmp_path / "empty-cache",
        )
