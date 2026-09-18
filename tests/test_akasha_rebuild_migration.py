from __future__ import annotations

import importlib.util
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from types import ModuleType

from yoyo.migrations import StepCollector

from agent.migrations.bundles import discover_migration_bundles
from agent.migrations.context import bind_migration_context

REPO = Path(__file__).resolve().parents[1]
BUNDLE_ID = "akasha"
MIGRATION_ID = "20260918_02_correct_graph_replay_request"


def _step_module() -> ModuleType:
    """按文件名加载迁移 step；模块名以数字开头，不能用普通 import 语句。"""

    path = REPO / "plugins" / "akasha" / "akasha_migrations" / f"{MIGRATION_ID}.py"
    spec = importlib.util.spec_from_file_location(f"akasha_migration_{MIGRATION_ID}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # yoyo 的 step() 从调用帧的 globals 读取 collector；直接加载模块时必须提供它。
    module.__dict__["__yoyo_collector__"] = StepCollector(None)
    spec.loader.exec_module(module)
    return module


def _write_config(root: Path, *, engine: str = "akasha", enabled: bool = True) -> Path:
    config = root / "config.toml"
    config.write_text(
        "[memory]\n"
        f"enabled = {'true' if enabled else 'false'}\n"
        f"engine = {engine!r}\n",
        encoding="utf-8",
    )
    return config


def _write_memory(root: Path, *, version: int | None) -> Path:
    """写一个只有 metadata 的最小学习图；version=None 表示缺少消费状态。"""

    memory = root / "memory" / "akasha.db"
    memory.parent.mkdir(parents=True, exist_ok=True)
    memory.unlink(missing_ok=True)
    with closing(sqlite3.connect(memory)) as connection:
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("CREATE TABLE turn_nodes (node_id INTEGER PRIMARY KEY)")
        if version is not None:
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('consumer_state_json', ?)",
                (json.dumps({"version": version, "cutover_heads": [], "applied": []}),),
            )
        connection.commit()
    return memory


def test_akasha_bundle_is_discoverable_and_import_clean() -> None:
    bundles = discover_migration_bundles(plugin_dirs=(REPO / "plugins",))
    akasha = next(item for item in bundles if item.bundle_id == BUNDLE_ID)
    assert akasha.migration_ids == (
        "20260918_01_register_graph_replay",
        MIGRATION_ID,
    )
    assert akasha.plugin_name == "akasha"


def test_replay_request_is_registered_once_with_a_readable_recovery_point(tmp_path: Path) -> None:
    step_module = _step_module()

    workspace = tmp_path / "workspace"
    _write_memory(workspace, version=1)
    config = _write_config(tmp_path)
    data_root = workspace / "plugin-data" / "akasha-builtin"

    with bind_migration_context(
        config_path=config, workspace=workspace, bundle_data_roots={BUNDLE_ID: data_root},
    ):
        step_module.request_akasha_replay(None)

    request = workspace / "memory" / ".akasha-replay-request.json"
    payload = json.loads(request.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["memory_path"] == str(workspace / "memory" / "akasha.db")

    backup = Path(payload["backup_path"])
    assert backup.is_file()
    with closing(sqlite3.connect(f"file:{backup}?mode=ro", uri=True)) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert connection.execute("SELECT COUNT(*) FROM turn_nodes").fetchone() == (0,)

    # 第二次运行看到当前消费版本后清掉凭据，不再重复登记。
    _write_memory(workspace, version=2)
    with bind_migration_context(
        config_path=config, workspace=workspace, bundle_data_roots={BUNDLE_ID: data_root},
    ):
        step_module.request_akasha_replay(None)
    assert not request.exists()


def test_replay_request_skips_installations_that_do_not_use_akasha(tmp_path: Path) -> None:
    step_module = _step_module()

    workspace = tmp_path / "workspace"
    _write_memory(workspace, version=1)
    data_root = workspace / "plugin-data" / "akasha-builtin"
    with bind_migration_context(
        config_path=_write_config(tmp_path, engine="default"),
        workspace=workspace,
        bundle_data_roots={BUNDLE_ID: data_root},
    ):
        step_module.request_akasha_replay(None)
    assert not (workspace / "memory" / ".akasha-replay-request.json").exists()


def test_replay_request_is_absent_for_a_workspace_without_a_learned_graph(tmp_path: Path) -> None:
    step_module = _step_module()

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_root = workspace / "plugin-data" / "akasha-builtin"
    with bind_migration_context(
        config_path=_write_config(tmp_path),
        workspace=workspace,
        bundle_data_roots={BUNDLE_ID: data_root},
    ):
        step_module.request_akasha_replay(None)
    assert not (workspace / "memory" / ".akasha-replay-request.json").exists()
