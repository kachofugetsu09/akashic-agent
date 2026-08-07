from __future__ import annotations

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.model_runtime.store import ModelRegistryStore

__depends__ = {"20260807_01_model_registry_database"}


def add_embedding_model_registry(_connection: object) -> None:
    """Add first-class embedding models without changing existing model rows."""

    current = current_migration_context()
    store = ModelRegistryStore.for_workspace(current.workspace)
    store.initialize()
    store.integrity_check()


steps = [step(add_embedding_model_registry)]
