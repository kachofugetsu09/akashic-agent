"""Exercise real Models/SQLite opt-outs and the immutable Yoyo migration.

Run from the checkout: python -m scripts.model_enabled_sync_scenario
All data is temporary; the controlled driver makes no external requests.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from yoyo import get_backend, read_migrations

from agent.migrations.context import bind_migration_context
from agent.migrations.bundles import discover_migration_bundles
from agent.plugin_composition import (
    CapabilitySources,
    DiscoveredModel,
    ModelCapabilities,
    ModelKind,
)
from plugins.models.settings import SetDefaultModel, SetModelEnabled, SyncModels
from plugins.models.state import ModelUnavailableError
from plugins.models.store import ModelsStore, RevisionConflictError
from tests.support.material_models import MaterialModelDriver, material_models

CONNECTION = "material-fixture-connection"
MIGRATION = "20260930_01_model_user_disabled"
CHECKS: list[str] = []


async def model_scenario(workspace: Path) -> None:
    chat = DiscoveredModel(
        kind=ModelKind.CHAT,
        model="extra-chat",
        capabilities=ModelCapabilities(context_window=8192),
        capability_sources=CapabilitySources(context_window="fixture"),
    )
    embedding = DiscoveredModel(
        kind=ModelKind.EMBEDDING,
        model="extra-embedding",
        capabilities=ModelCapabilities(
            embedding_dimensions=3, embedding_normalization="unit"
        ),
        capability_sources=CapabilitySources(embedding_dimensions="probe"),
    )
    returned = [chat, embedding]
    original_definition = MaterialModelDriver.definition

    async def discover(*_args):
        return tuple(returned)

    def definition(driver):
        original = original_definition(driver)

        async def probe_embedding(descriptor, credential, model):
            if model == embedding.model:
                driver.embedding_bindings.append(model)
                return embedding
            return await original.probe_embedding(descriptor, credential, model)

        return replace(original, discover=discover, probe_embedding=probe_embedding)

    with patch.object(MaterialModelDriver, "definition", definition):
        async with material_models(workspace) as models:

            async def apply(command):
                return await models.settings.apply(command)

            def revision():
                return models.store.read_snapshot().revision

            async def sync():
                return await apply(SyncModels(revision(), CONNECTION))

            await sync()
            saved = {
                model.model: model
                for model in models.store.read_snapshot().models.values()
            }
            ids = [saved[item.model].model_id for item in returned]
            for model_id in ids:
                await apply(SetModelEnabled(revision(), model_id, False))
            disabled_revision = revision()
            for model_id in ids:
                await apply(SetModelEnabled(revision(), model_id, False))
            assert revision() == disabled_revision
            models.driver.reset_records()
            await sync()
            assert (
                revision() == disabled_revision
            ), "identical sync with opted-out models must be a no-op"
            assert (
                not models.driver.chat_requests and not models.driver.embedding_bindings
            )
            returned[0] = replace(
                chat, capabilities=ModelCapabilities(context_window=16384)
            )
            await sync()
            snapshot = models.store.read_snapshot()
            for model_id in ids:
                model = snapshot.models[model_id]
                assert (
                    model.user_disabled and not model.enabled and model.discovery_owned
                )
            assert snapshot.models[ids[0]].capabilities.context_window == 16384
            readonly = ModelsStore(
                models.store.path, workspace / "unused-backups", writable=False
            )
            assert readonly.read_snapshot() == snapshot
            CHECKS.append(
                "Chat and embedding opt-outs survive sync/readback; capability refresh and ownership remain intact"
            )

            # An older artifact can refresh the raw enabled bit without knowing
            # the additive opt-out flag. All new writer eligibility must agree
            # with the effective availability exposed by the current reader.
            with closing(sqlite3.connect(models.store.path)) as connection:
                for table, model_id in zip(
                    ("model_definitions", "embedding_models"), ids
                ):
                    connection.execute(
                        f"UPDATE {table} SET enabled=1 WHERE id=?", (model_id,)
                    )
                connection.commit()
            before_binding = models.store.read_snapshot()
            for role, model_id in zip(("agent", None), ids):
                try:
                    await apply(SetDefaultModel(revision(), role, model_id))
                except ValueError as error:
                    assert "unavailable" in str(error)
                else:
                    raise AssertionError(
                        "an opted-out model became a default after an old writer refresh"
                    )
                assert models.store.read_snapshot() == before_binding
            CHECKS.append(
                "Direct default commands reject retained opt-outs even after an old writer sets raw enabled=1; no revision or binding changes"
            )

            original_bind = MaterialModelDriver.bind_chat

            def reject(driver, descriptor, config):
                if descriptor.model == chat.model:
                    raise ModelUnavailableError("fixture rejection")
                return original_bind(driver, descriptor, config)

            failed_revision = revision()
            with patch.object(MaterialModelDriver, "bind_chat", reject):
                try:
                    await apply(SetModelEnabled(revision(), ids[0], True))
                except ModelUnavailableError:
                    pass
                else:
                    raise AssertionError("unverified reopening succeeded")
            assert revision() == failed_revision
            assert models.store.read_snapshot().models[ids[0]].user_disabled
            for model_id in ids:
                await apply(SetModelEnabled(revision(), model_id, True))
            assert len(models.driver.chat_requests) == 1
            assert models.driver.embedding_bindings == [embedding.model]
            try:
                await apply(SetModelEnabled(failed_revision, ids[0], False))
            except RevisionConflictError:
                pass
            else:
                raise AssertionError("stale user choice committed")
            CHECKS.append(
                "Reopening verifies each kind; failed verification and stale CAS preserve committed choices"
            )

            returned[:] = [embedding]
            await sync()
            removed = models.store.read_snapshot().models[ids[0]]
            assert not removed.enabled and not removed.user_disabled
            returned.insert(0, chat)
            await sync()
            assert models.store.read_snapshot().models[ids[0]].enabled
            returned[:] = [embedding]
            await sync()
            await apply(SetModelEnabled(revision(), ids[0], False))
            returned.insert(0, chat)
            await sync()
            assert not models.store.read_snapshot().models[ids[0]].enabled
            CHECKS.append(
                "Provider disappearance/reappearance auto-recovers unless an explicit opt-out is recorded while absent"
            )

            for model_id in ["material-fixture-chat", "material-fixture-embedding"]:
                try:
                    await apply(SetModelEnabled(revision(), model_id, False))
                except ValueError as error:
                    assert "still in use" in str(error)
                else:
                    raise AssertionError("bound model was disabled")
            CHECKS.append(
                "Chat role and default embedding references still reject disabling"
            )


def migrate(workspace: Path, *, fail_backup: bool = False):
    root = Path(__file__).resolve().parents[1] / "plugins/models/models_migrations"
    bundles = discover_migration_bundles(plugin_dirs=(root.parent,))
    assert [(bundle.bundle_id, bundle.migration_ids) for bundle in bundles] == [
        ("models", (MIGRATION,))
    ]
    backend = get_backend(f"sqlite:///{workspace / 'migration-ledger.sqlite3'}")
    migrations = read_migrations(str(root))
    migration = migrations[0]
    migration.load()
    with bind_migration_context(
        workspace=workspace, config_path=workspace / "config.toml"
    ):
        if fail_backup:
            with patch.object(
                migration.module,
                "_backup_registry",
                side_effect=RuntimeError("fixture backup failure"),
            ):
                backend.apply_migrations(backend.to_apply(migrations))
        else:
            backend.apply_migrations(backend.to_apply(migrations))
    return migration.module.add_user_disabled


def migration_scenario(workspace: Path) -> None:
    path = workspace / "model-registry.sqlite3"
    with closing(sqlite3.connect(path)) as connection:
        for table in ("model_definitions", "embedding_models"):
            connection.execute(f"ALTER TABLE {table} DROP COLUMN user_disabled")
        connection.commit()
        before = {
            table: connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            for table in (
                "model_registry_meta",
                "model_connections",
                "model_definitions",
                "embedding_models",
                "model_role_bindings",
            )
        }
    legacy = ModelsStore(path, workspace / "legacy-backups", writable=False)
    assert not any(
        model.user_disabled for model in legacy.read_snapshot().models.values()
    )
    writer = ModelsStore(path, workspace / "legacy-backups", writable=True)
    try:
        writer.initialize()
    except RuntimeError as error:
        assert MIGRATION in str(error)
    else:
        raise AssertionError("ordinary startup applied an unapproved expansion")
    finally:
        writer.close()
    try:
        migrate(workspace, fail_backup=True)
    except RuntimeError as error:
        assert "fixture backup failure" in str(error)
    else:
        raise AssertionError("migration proceeded without a recovery point")
    with closing(sqlite3.connect(path)) as connection:
        for table in ("model_definitions", "embedding_models"):
            assert "user_disabled" not in {
                row[1] for row in connection.execute(f"PRAGMA table_info({table})")
            }
    callback = migrate(workspace)
    backup_roots = list((workspace / "runtime/model-backups").glob(f"{MIGRATION}-*"))
    assert len(backup_roots) == 1
    root = backup_roots[0]
    manifest = json.loads((root / "manifest.json").read_text())
    backup = root / manifest["backup"]
    assert manifest["sha256"] == hashlib.sha256(backup.read_bytes()).hexdigest()
    assert backup.stat().st_mode & 0o777 == 0o600
    with closing(sqlite3.connect(backup)) as connection:
        for table, rows in before.items():
            assert (
                connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
                == rows
            )
    with closing(sqlite3.connect(path)) as connection:
        for table, rows in before.items():
            after = connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            if table in ("model_definitions", "embedding_models"):
                assert all(row[-1] == 0 for row in after)
                after = [row[:-1] for row in after]
            assert after == rows
    migrate(workspace)
    assert (
        list((workspace / "runtime/model-backups").glob(f"{MIGRATION}-*"))
        == backup_roots
    )
    current = ModelsStore(path, workspace / "current-backups", writable=True)
    current.initialize()
    current.close()
    CHECKS.append(
        "Yoyo migrates the prior schema once; private verified backup and all existing rows/revision are preserved"
    )
    with bind_migration_context(
        workspace=workspace, config_path=workspace / "config.toml"
    ):
        callback(
            None
        )  # Also verify callback idempotence, independently of Yoyo's ledger.
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("ALTER TABLE embedding_models DROP COLUMN user_disabled")
            connection.commit()
        try:
            callback(None)
        except RuntimeError as error:
            assert "partially installed" in str(error)
        else:
            raise AssertionError("partial schema was silently repaired")
    assert (
        list((workspace / "runtime/model-backups").glob(f"{MIGRATION}-*"))
        == backup_roots
    )
    CHECKS.append(
        "Backup failure leaves both tables unchanged; replay is idempotent and partial schema fails closed"
    )


def migration_boundary_scenario(root: Path) -> None:
    migration = read_migrations(
        str(Path(__file__).resolve().parents[1] / "plugins/models/models_migrations")
    )[0]
    migration.load()
    callback = migration.module.add_user_disabled

    def prior_registry(workspace: Path) -> Path:
        workspace.mkdir()
        path = workspace / "model-registry.sqlite3"
        store = ModelsStore(path, workspace / "unused-backups")
        store.initialize()
        store.close()
        with closing(sqlite3.connect(path)) as connection:
            for table in ("model_definitions", "embedding_models"):
                connection.execute(f"ALTER TABLE {table} DROP COLUMN user_disabled")
            connection.commit()
        return path

    def run(workspace: Path) -> None:
        with bind_migration_context(
            workspace=workspace, config_path=workspace / "config.toml"
        ):
            callback(None)

    for relative, destination_kind in [
        ("runtime", "outside"),
        ("runtime/model-backups", "outside"),
        ("runtime", "dangling"),
        ("runtime", "inside"),
    ]:
        case = root / f"symlink-{relative.replace('/', '-')}-{destination_kind}"
        case.mkdir()
        workspace = case / "workspace"
        path = prior_registry(workspace)
        before = path.read_bytes()
        destination = (
            workspace / "redirect" if destination_kind == "inside" else case / "outside"
        )
        if destination_kind != "dangling":
            destination.mkdir()
        link = workspace / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(destination, target_is_directory=True)
        before_entries = sorted(
            str(entry.relative_to(case)) for entry in case.rglob("*")
        )
        try:
            run(workspace)
        except ValueError as error:
            assert "symbolic-link" in str(error)
        else:
            raise AssertionError(f"migration followed {relative} -> {destination_kind}")
        assert path.read_bytes() == before
        assert (
            sorted(str(entry.relative_to(case)) for entry in case.rglob("*"))
            == before_entries
        )
    CHECKS.append(
        "Migration rejects runtime/backup-parent symlinks, including internal and dangling links, with unchanged registry and no new files inside or outside the workspace"
    )

    template = root / "schema-template"
    path = prior_registry(template)
    with closing(sqlite3.connect(path)) as connection:
        schema = ";\n".join(
            row[0]
            for row in connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table'"
            )
        )
    malformed = {
        "wrong-types": schema.replace(
            "id TEXT PRIMARY KEY", "id BLOB PRIMARY KEY"
        ).replace("enabled INTEGER", "enabled TEXT"),
        "wrong-meta": schema.replace(
            "revision INTEGER NOT NULL CHECK (revision >= 0)", "revision TEXT"
        ),
        "missing-check": schema.replace("CHECK (enabled IN (0, 1))", ""),
        "missing-foreign-key": schema.replace(
            "REFERENCES model_connections(id) ON DELETE RESTRICT", ""
        ),
        "missing-unique": schema.replace("UNIQUE(connection_id, model)", "CHECK (1)"),
        "unknown-conflict-policy": schema.replace(
            "revision INTEGER NOT NULL CHECK",
            "revision INTEGER NOT NULL ON CONFLICT IGNORE CHECK",
        ),
        "unknown-collation-policy": schema.replace(
            "name TEXT NOT NULL", "name TEXT COLLATE NOCASE NOT NULL"
        ),
        "unknown-deferral-policy": schema.replace(
            "REFERENCES model_connections(id) ON DELETE RESTRICT",
            "REFERENCES model_connections(id) ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED",
        ),
        "unknown-autoincrement-policy": schema.replace(
            "singleton INTEGER PRIMARY KEY CHECK",
            "singleton INTEGER PRIMARY KEY AUTOINCREMENT CHECK",
        ),
        "unknown-trigger": schema
        + "; CREATE TRIGGER unexpected_model_write AFTER INSERT ON model_definitions BEGIN UPDATE model_registry_meta SET revision=revision+1; END",
    }
    for name, definition in malformed.items():
        assert definition != schema
        workspace = root / name
        workspace.mkdir()
        path = workspace / "model-registry.sqlite3"
        with closing(sqlite3.connect(path)) as connection:
            connection.executescript(definition)
        before = path.read_bytes()
        try:
            run(workspace)
        except RuntimeError as error:
            assert "schema lineage" in str(error)
        else:
            raise AssertionError(f"migration accepted {name}")
        assert path.read_bytes() == before
        assert sorted(entry.name for entry in workspace.iterdir()) == [path.name]
    CHECKS.append(
        "Unknown registry types, metadata, CHECK/FK/UNIQUE constraints, unexposed constraint policies and triggers fail before any registry or backup writes"
    )

    additions = {
        "model_connections": ["driver_config_json TEXT NOT NULL DEFAULT '{}'"],
        "model_registry_meta": [
            "default_embedding_model_id TEXT DEFAULT NULL",
            "host_epoch INTEGER NOT NULL DEFAULT 0",
        ],
        "model_definitions": ["capabilities_json TEXT"],
        "embedding_models": ["capabilities_json TEXT"],
    }
    for append in (False, True):
        workspace = root / f"known-additive-{append}"
        path = prior_registry(workspace)
        with closing(sqlite3.connect(path)) as connection:
            for table, columns in additions.items():
                for definition in columns:
                    connection.execute(
                        f"ALTER TABLE {table} DROP COLUMN {definition.split()[0]}"
                    )
                    if append:
                        connection.execute(
                            f"ALTER TABLE {table} ADD COLUMN {definition}"
                        )
            connection.commit()
        run(workspace)
        with closing(sqlite3.connect(path)) as connection:
            for table in ("model_definitions", "embedding_models"):
                assert "user_disabled" in {
                    row[1] for row in connection.execute(f"PRAGMA table_info({table})")
                }
        run(workspace)
        assert len(list((workspace / "runtime/model-backups").iterdir())) == 1
    CHECKS.append(
        "Both known pre-additive and appended legacy schemas migrate and replay, including historical host_epoch default 0"
    )

    workspace = root / "equivalent-ddl"
    workspace.mkdir()
    with closing(sqlite3.connect(workspace / "model-registry.sqlite3")) as connection:
        connection.executescript(
            schema.replace(
                "CREATE TABLE model_definitions", 'create table "model_definitions"'
            )
            .replace("id TEXT PRIMARY KEY", '"id" text primary key')
            .replace("CHECK (", "check (  ")
        )
    run(workspace)
    CHECKS.append(
        "Equivalent DDL quoting, keyword case and whitespace preserve the recognized SQLite schema identity"
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="models-opt-out-scenario-") as temporary:
        workspace = Path(temporary)
        asyncio.run(model_scenario(workspace))
        migration_scenario(workspace)
        migration_boundary_scenario(workspace)
    print(
        json.dumps(
            {
                "passed": len(CHECKS),
                "boundary": "Real Models composition/SQLite/Yoyo, controlled non-network provider, temporary workspace",
                "checks": CHECKS,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
