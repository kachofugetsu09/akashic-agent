from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.model_runtime.catalog.litellm_registry import (
    resolve_catalog_capabilities,
)
from agent.model_runtime.store import ModelRegistrySnapshot, ModelRegistryStore

__depends__ = {"20260807_02_embedding_model_registry"}


def restore_migrated_reasoning_efforts(_connection: object) -> None:
    """补回迁移时缺失的思考强度，同时保持模型身份不变。"""

    current = current_migration_context()
    store = ModelRegistryStore.for_workspace(current.workspace)
    before = store.read_snapshot()
    if before is None:
        return

    # 1. 只从固定本地目录推导旧 TOML 迁移遗漏的字段
    updates = _reasoning_effort_updates(before)
    if not updates:
        return

    # 2. 修改包含凭据的模型库前创建可验证备份
    backup = _backup_registry(current.workspace, store)
    try:
        with closing(sqlite3.connect(store.path)) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("BEGIN IMMEDIATE")
            for runtime_id, efforts in updates.items():
                cursor = connection.execute(
                    """
                    UPDATE model_definitions
                    SET supported_reasoning_efforts = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND supported_reasoning_efforts = '[]'
                    """,
                    (
                        json.dumps(efforts, ensure_ascii=False, separators=(",", ":")),
                        runtime_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(f"模型能力迁移遇到并发变化: {runtime_id}")
            connection.execute(
                "UPDATE model_registry_meta SET revision = revision + 1 WHERE singleton = 1"
            )
            connection.commit()

        # 3. 验证身份、角色和所有非目标字段均未变化
        store.integrity_check()
        after = store.read_snapshot()
        _validate_reconciled_snapshot(before, after, updates)
    except BaseException:
        store.restore_from(backup)
        raise


def _reasoning_effort_updates(
    snapshot: ModelRegistrySnapshot,
) -> dict[str, tuple[str, ...]]:
    """解析缺失的强度列表，并保留旧配置正在使用的强度。"""

    updates: dict[str, tuple[str, ...]] = {}
    for runtime_id, runtime in snapshot.runtimes.items():
        if runtime.supported_reasoning_efforts:
            continue
        capabilities = resolve_catalog_capabilities(
            runtime.catalog_provider_id or runtime.provider,
            runtime.model,
            base_url=runtime.base_url,
        )
        efforts = list(
            capabilities.supported_reasoning_efforts if capabilities is not None else ()
        )
        if runtime.reasoning_effort and runtime.reasoning_effort not in efforts:
            efforts.append(runtime.reasoning_effort)
        if efforts:
            updates[runtime_id] = tuple(efforts)
    return updates


def _backup_registry(workspace: Path, store: ModelRegistryStore) -> Path:
    backup = (
        workspace
        / "backups"
        / "model-registry-capabilities-v1"
        / uuid4().hex
        / "registry.before.sqlite3"
    )
    store.backup_to(backup)
    return backup


def _validate_reconciled_snapshot(
    before: ModelRegistrySnapshot,
    after: ModelRegistrySnapshot | None,
    updates: dict[str, tuple[str, ...]],
) -> None:
    if after is None:
        raise RuntimeError("模型能力迁移后注册库为空")
    if after.revision != before.revision + 1:
        raise RuntimeError("模型能力迁移没有提交唯一 revision")
    if after.roles != before.roles or after.runtimes.keys() != before.runtimes.keys():
        raise RuntimeError("模型能力迁移改变了模型身份或角色绑定")
    for runtime_id, old_runtime in before.runtimes.items():
        new_runtime = after.runtimes[runtime_id]
        expected_efforts = updates.get(
            runtime_id,
            old_runtime.supported_reasoning_efforts,
        )
        if new_runtime.supported_reasoning_efforts != expected_efforts:
            raise RuntimeError(f"模型能力迁移结果不完整: {runtime_id}")
        if (
            replace(
                new_runtime,
                supported_reasoning_efforts=old_runtime.supported_reasoning_efforts,
            )
            != old_runtime
        ):
            raise RuntimeError(f"模型能力迁移改写了非目标字段: {runtime_id}")


steps = [step(restore_migrated_reasoning_efforts)]
