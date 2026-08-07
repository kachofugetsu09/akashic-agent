from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.model_runtime.store import ModelRegistrySnapshot, ModelRegistryStore

__depends__ = {"20260808_01_restore_migrated_reasoning_efforts"}


_STALE_VARIANTS = ("low", "medium", "high")
_OPENCODE_GO_VARIANTS = {
    "deepseek-v4-flash": ("low", "high", "max"),
    "deepseek-v4-pro": ("high", "max"),
}


def correct_opencode_go_variants(_connection: object) -> None:
    """勘误被通用注册表写错的 OpenCode Go DeepSeek variant。"""

    current = current_migration_context()
    store = ModelRegistryStore.for_workspace(current.workspace)
    before = store.read_snapshot()
    if before is None:
        return

    # 1. 只修正上一条迁移产生的精确错误形状
    updates = _variant_updates(before)
    if not updates:
        return

    # 2. 修改模型库前创建恢复点并原子提交唯一 revision
    backup = _backup_registry(current.workspace, store)
    try:
        with closing(sqlite3.connect(store.path)) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("BEGIN IMMEDIATE")
            for runtime_id, variants in updates.items():
                cursor = connection.execute(
                    """
                    UPDATE model_definitions
                    SET supported_reasoning_efforts = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND supported_reasoning_efforts = ?
                    """,
                    (
                        _encode(variants),
                        runtime_id,
                        _encode(_STALE_VARIANTS),
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(f"OpenCode variant 勘误遇到并发变化: {runtime_id}")
            connection.execute(
                "UPDATE model_registry_meta SET revision = revision + 1 WHERE singleton = 1"
            )
            connection.commit()

        # 3. 验证除目标能力列表和 revision 外的权威状态逐项不变
        store.integrity_check()
        after = store.read_snapshot()
        _validate_corrected_snapshot(before, after, updates)
    except BaseException:
        store.restore_from(backup)
        raise


def _variant_updates(
    snapshot: ModelRegistrySnapshot,
) -> dict[str, tuple[str, ...]]:
    """定位由通用 LiteLLM fallback 写入的精确错误记录。"""

    updates: dict[str, tuple[str, ...]] = {}
    for runtime_id, runtime in snapshot.runtimes.items():
        expected = _OPENCODE_GO_VARIANTS.get(runtime.model)
        if (
            runtime.catalog_provider_id == "opencode-go"
            and runtime.supported_reasoning_efforts == _STALE_VARIANTS
            and expected is not None
        ):
            updates[runtime_id] = expected
    return updates


def _backup_registry(workspace: Path, store: ModelRegistryStore) -> Path:
    backup = (
        workspace
        / "backups"
        / "model-registry-opencode-variants-v1"
        / uuid4().hex
        / "registry.before.sqlite3"
    )
    store.backup_to(backup)
    return backup


def _validate_corrected_snapshot(
    before: ModelRegistrySnapshot,
    after: ModelRegistrySnapshot | None,
    updates: dict[str, tuple[str, ...]],
) -> None:
    if after is None:
        raise RuntimeError("OpenCode variant 勘误后模型注册库为空")
    if after.revision != before.revision + 1:
        raise RuntimeError("OpenCode variant 勘误没有提交唯一 revision")
    if after.roles != before.roles or after.runtimes.keys() != before.runtimes.keys():
        raise RuntimeError("OpenCode variant 勘误改变了模型身份或角色绑定")
    for runtime_id, old_runtime in before.runtimes.items():
        new_runtime = after.runtimes[runtime_id]
        expected = updates.get(runtime_id, old_runtime.supported_reasoning_efforts)
        if new_runtime.supported_reasoning_efforts != expected:
            raise RuntimeError(f"OpenCode variant 勘误结果不完整: {runtime_id}")
        if (
            replace(
                new_runtime,
                supported_reasoning_efforts=old_runtime.supported_reasoning_efforts,
            )
            != old_runtime
        ):
            raise RuntimeError(f"OpenCode variant 勘误改写了非目标字段: {runtime_id}")


def _encode(values: tuple[str, ...]) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


steps = [step(correct_opencode_go_variants)]
