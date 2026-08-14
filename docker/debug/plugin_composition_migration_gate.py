from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import RuntimeSnapshot
from agent.tools.registry import ToolRegistry
from bus.event_bus import EventBus
from bus.events_lifecycle import TurnCommitted
from session.manager import Session, SessionManager


DEFAULT_LOCK = ROOT / "docker/debug/plugin-composition-migration.lock.json"
DEFAULT_REPORT = (
    ROOT / "docker/debug/reports/plugin-composition-migration/gate.json"
)
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class PluginSource:
    id: str
    repository: str
    commit: str
    pull_request: str


@dataclass(frozen=True)
class MigrationLock:
    profile: str
    plugins: tuple[PluginSource, ...]


@dataclass(frozen=True)
class GateArguments:
    lock: Path
    report: Path
    require_clean_core: bool


class _Drainable(Protocol):
    async def drain(self) -> None: ...


class _RunnableModule(Protocol):
    async def run(self, frame: Any) -> None: ...


def main() -> None:
    """验证固定 Observe/status_commands 候选的组合所有权。"""

    # 1. 固定 Core 与外部候选身份
    args = _parse_args()
    core_status = _git_output(ROOT, "status", "--porcelain").splitlines()
    if args.require_clean_core and core_status:
        raise RuntimeError(f"Core Git worktree 不干净: {core_status}")
    lock_path = args.lock.resolve(strict=True)
    lock = _load_lock(lock_path)

    # 2. 在一次性目录运行真实混合 v2/v3 generation
    with tempfile.TemporaryDirectory(
        prefix="akashic-plugin-composition-migration-"
    ) as raw_temp:
        temp_root = Path(raw_temp)
        plugins_root = temp_root / "plugins"
        plugins_root.mkdir()
        source_evidence = [
            _checkout_source(source, plugins_root / source.id)
            for source in lock.plugins
        ]
        observations = asyncio.run(_exercise_profile(temp_root, plugins_root))

    # 3. 只发布可重建的组合回执
    report: dict[str, object] = {
        "status": "passed",
        "checked_at": datetime.now(UTC).isoformat(),
        "profile": lock.profile,
        "core_head": _git_output(ROOT, "rev-parse", "HEAD"),
        "core_tree": _git_output(ROOT, "rev-parse", "HEAD^{tree}"),
        "core_dirty_status": core_status,
        "lock": str(lock_path.relative_to(ROOT)),
        "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "plugins": source_evidence,
        "observations": observations,
    }
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _ = report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"plugin composition migration gate passed: {report_path}")


def _parse_args() -> GateArguments:
    parser = argparse.ArgumentParser(description="验证插件组合迁移候选")
    _ = parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    _ = parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    _ = parser.add_argument("--require-clean-core", action="store_true")
    raw = parser.parse_args()
    return GateArguments(
        lock=cast(Path, raw.lock),
        report=cast(Path, raw.report),
        require_clean_core=cast(bool, raw.require_clean_core),
    )


def _load_lock(path: Path) -> MigrationLock:
    """严格解析迁移锁，不接受浮动 ref 或额外字段。"""

    # 1. 根结构和 profile 必须唯一明确
    decoded = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(decoded, dict):
        raise ValueError("插件组合迁移锁根结构无效")
    raw = cast(dict[str, object], decoded)
    if set(raw) != {
        "schema_version",
        "profile",
        "plugins",
    }:
        raise ValueError("插件组合迁移锁根结构无效")
    if raw["schema_version"] != 1:
        raise ValueError(f"不支持的插件组合迁移锁版本: {raw['schema_version']}")
    profile = raw["profile"]
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError("插件组合迁移锁 profile 必须是非空字符串")

    # 2. 当前 profile 精确绑定两个能力 owner
    plugins = raw["plugins"]
    if not isinstance(plugins, list):
        raise ValueError("插件组合迁移锁 plugins 必须是数组")
    plugin_items = cast(list[object], plugins)
    parsed = tuple(_parse_source(item) for item in plugin_items)
    if tuple(item.id for item in parsed) != ("observe", "status_commands"):
        raise ValueError("当前迁移 profile 必须按 Observe、status_commands 排列")
    return MigrationLock(profile=profile, plugins=parsed)


def _parse_source(raw: object) -> PluginSource:
    expected = {"id", "repository", "commit", "pull_request"}
    if not isinstance(raw, dict):
        raise ValueError(f"插件组合迁移 source 字段无效: {raw}")
    item = cast(dict[str, object], raw)
    if set(item) != expected:
        raise ValueError(f"插件组合迁移 source 字段无效: {raw}")
    values = {name: _required_string(item, name) for name in expected}
    if not values["repository"].startswith("https://github.com/"):
        raise ValueError(f"插件 repository 必须是 GitHub HTTPS: {values['id']}")
    if _COMMIT_PATTERN.fullmatch(values["commit"]) is None:
        raise ValueError(f"插件 commit 必须是完整 SHA: {values['id']}")
    if not values["pull_request"].startswith("https://github.com/"):
        raise ValueError(f"插件 pull_request 必须是 GitHub HTTPS: {values['id']}")
    return PluginSource(**values)


def _required_string(item: dict[str, object], name: str) -> str:
    value = item[name]
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"插件组合迁移 source {name} 必须是非空字符串")
    return value


def _checkout_source(source: PluginSource, checkout: Path) -> dict[str, object]:
    """只获取锁定 commit 并返回源码身份。"""

    # 1. 不复用宿主 cache，检出后必须 clean
    _run(("git", "init", "--quiet", str(checkout)), cwd=ROOT)
    _run(("git", "remote", "add", "origin", source.repository), cwd=checkout)
    _run(
        ("git", "fetch", "--quiet", "--depth=1", "origin", source.commit),
        cwd=checkout,
    )
    _run(("git", "checkout", "--quiet", "--detach", "FETCH_HEAD"), cwd=checkout)
    if _git_output(checkout, "rev-parse", "HEAD") != source.commit:
        raise RuntimeError(f"插件检出身份漂移: {source.id}")
    if _git_output(checkout, "status", "--porcelain"):
        raise RuntimeError(f"插件检出后不干净: {source.id}")

    # 2. 报告同时绑定 commit、tree 和入口摘要
    plugin_path = checkout / "plugin.py"
    return {
        **asdict(source),
        "tree": _git_output(checkout, "rev-parse", "HEAD^{tree}"),
        "plugin_sha256": hashlib.sha256(plugin_path.read_bytes()).hexdigest(),
    }


async def _exercise_profile(
    temp_root: Path,
    plugins_root: Path,
) -> dict[str, object]:
    """运行真实 generation 并比较目录、结果、写集和清理。"""

    workspace = temp_root / "workspace"
    event_bus = EventBus()
    sessions = SessionManager(workspace)
    manager = PluginManager(
        plugin_dirs=[plugins_root],
        event_bus=event_bus,
        tool_registry=ToolRegistry(),
        workspace=workspace,
        session_manager=sessions,
        installed_cache_root=temp_root / "plugin-home/cache",
    )
    observations: dict[str, object] = {}
    try:
        session = _seed_session(sessions)
        await manager.load_all()
        snapshot = manager.current_snapshot
        if snapshot is None:
            raise RuntimeError("组合迁移候选没有发布 RuntimeSnapshot")
        observations = await _collect_observations(
            manager,
            snapshot,
            event_bus,
            session,
            workspace,
        )
    finally:
        await manager.terminate_all()
        sessions.close()
    if manager.loaded_count != 0 or manager.current_snapshot is not None:
        raise RuntimeError("组合迁移候选 terminate 后仍有 generation")
    observations["cleanup"] = {"loaded_count": 0, "snapshot": None}
    return observations


async def _collect_observations(
    manager: PluginManager,
    snapshot: RuntimeSnapshot,
    event_bus: EventBus,
    session: Session,
    workspace: Path,
) -> dict[str, object]:
    """收集组合身份、能力 owner 与写集证据。"""

    # 1. 在命令执行前固定 Core 权威数据库摘要
    _assert_generation_shape(manager, snapshot)
    before_sessions = _database_files(workspace / "sessions.db")

    # 2. 分别走 v3 命令和 Observe-owned legacy module
    status_reply = await _exercise_memory_status(snapshot, session.key)
    kvcache_reply, rows = await _exercise_kvcache(
        manager, snapshot, event_bus, session, workspace
    )
    if _database_files(workspace / "sessions.db") != before_sessions:
        raise RuntimeError("只读命令改变了 sessions.db write set")

    # 3. 回执保留可审阅结果，不复制临时数据库
    topology = snapshot.composition_topology
    return {
        "snapshot_id": snapshot.snapshot_id,
        "composition_identity": None if topology is None else topology.identity,
        "generations": sorted(snapshot.generations),
        "telegram_catalog": manager.telegram_bot_commands,
        "before_turn_slots": [
            str(getattr(module, "slot", ""))
            for module in snapshot.before_turn_modules
        ],
        "status_reply_first_line": status_reply.splitlines()[0],
        "kvcache_reply_first_line": kvcache_reply.splitlines()[0],
        "observe_rows": rows,
        "sessions_db": before_sessions,
    }


def _seed_session(sessions: SessionManager) -> Session:
    session = sessions.get_or_create("telegram:combo")
    session.messages = [
        {"role": "user", "content": "已整理的问题"},
        {"role": "assistant", "content": "旧回答"},
        {"role": "user", "content": "待整理的问题"},
    ]
    sessions.save(session)
    return session


def _assert_generation_shape(
    manager: PluginManager,
    snapshot: RuntimeSnapshot,
) -> None:
    if set(snapshot.generations) != {"observe", "status_commands"}:
        raise RuntimeError(f"插件 generation 集合漂移: {sorted(snapshot.generations)}")
    expected = [
        ("kvcache", "查看 KVCache 状态"),
        ("memorystatus", "查看记忆整理状态"),
    ]
    if manager.telegram_bot_commands != expected:
        raise RuntimeError(f"Telegram 命令目录漂移: {manager.telegram_bot_commands}")
    observe = snapshot.generations["observe"].contributions.mobile_ui_asset
    status = snapshot.generations["status_commands"].contributions.mobile_ui_asset
    if observe is None or observe.slots != ("turn.after_answer",):
        raise RuntimeError("Observe Mobile UI slot 漂移")
    if status is None or status.slots != ("drawer.panel",):
        raise RuntimeError("status_commands Mobile UI slot 漂移")


async def _exercise_memory_status(
    snapshot: RuntimeSnapshot,
    session_key: str,
) -> str:
    registry = snapshot.command_registry
    if registry is None:
        raise RuntimeError("status_commands 没有发布 v3 Command registry")
    execution = await registry.execute(
        "/memory_status",
        session_key=session_key,
        channel="telegram",
        chat_id="combo",
        sender="hua",
    )
    if execution is None or "记忆整理状态" not in execution.result.text:
        raise RuntimeError("status_commands v3 回复无效")
    if (
        await registry.execute(
            "/kvcache",
            session_key=session_key,
            channel="telegram",
            chat_id="combo",
            sender="hua",
        )
        is not None
    ):
        raise RuntimeError("status_commands 仍然认领 /kvcache")
    return execution.result.text


async def _exercise_kvcache(
    manager: PluginManager,
    snapshot: RuntimeSnapshot,
    event_bus: EventBus,
    session: Session,
    workspace: Path,
) -> tuple[str, list[tuple[object, ...]]]:
    """从 TurnCommitted 写入到 Observe 命令读取完整走一遍。"""

    # 1. 通过 Observe 既有事件入口建立真实插件数据
    await _seed_observe_turn(manager, event_bus, session.key)

    # 2. 只允许 Observe legacy module 认领 KVCache 命令
    reply = await _invoke_kvcache(snapshot, session)

    # 3. 写集只属于 Observe 数据库
    rows = _read_observe_rows(workspace)
    if rows != [(session.key, 300, 260)]:
        raise RuntimeError(f"Observe KVCache rows 漂移: {rows}")
    return reply, rows


async def _seed_observe_turn(
    manager: PluginManager,
    event_bus: EventBus,
    session_key: str,
) -> None:
    """通过公开事件入口写入并等待 Observe 队列落盘。"""

    await event_bus.observe(
        TurnCommitted(
            session_key=session_key,
            channel="telegram",
            chat_id="combo",
            input_message="cache?",
            persisted_user_message="cache?",
            assistant_response="cache reply",
            tools_used=[],
            react_stats={
                "cache_prompt_tokens": 300,
                "cache_hit_tokens": 260,
            },
        )
    )
    generation = manager.generation("observe")
    if generation is None:
        raise RuntimeError("Observe generation 缺失")
    writer = cast(_Drainable, getattr(generation.instance, "_writer"))
    await writer.drain()


async def _invoke_kvcache(
    snapshot: RuntimeSnapshot,
    session: Session,
) -> str:
    """调用唯一 Observe KVCache module 并返回回复。"""

    modules = [
        module
        for module in snapshot.before_turn_modules
        if getattr(module, "slot", "") == "observe.kvcache_command"
    ]
    if len(modules) != 1:
        raise RuntimeError(f"Observe KVCache module 数量漂移: {len(modules)}")
    state = SimpleNamespace(
        session_key=session.key,
        msg=SimpleNamespace(
            content="/kvcache",
            channel="telegram",
            chat_id="combo",
            timestamp=datetime.now(UTC),
        ),
    )
    frame = SimpleNamespace(
        input=state,
        slots={"session:session": session},
    )
    await cast(_RunnableModule, modules[0]).run(frame)
    reply = str(frame.slots["session:ctx"].abort_reply)
    if "260 / 300" not in reply:
        raise RuntimeError("Observe KVCache 回复与真实 row 不一致")
    return reply


def _read_observe_rows(workspace: Path) -> list[tuple[object, ...]]:
    """读取 Gate 临时 Observe 数据库中的最小行为证据。"""

    with sqlite3.connect(workspace / "observe/observe.db") as connection:
        return cast(
            list[tuple[object, ...]],
            connection.execute(
                """
                SELECT session_key, react_cache_prompt_tokens,
                       react_cache_hit_tokens
                FROM turns
                """
            ).fetchall(),
        )


def _database_files(db_path: Path) -> dict[str, dict[str, object]]:
    return {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(db_path.parent.glob(f"{db_path.name}*"))
        if path.is_file()
    }


def _run(command: tuple[str, ...], *, cwd: Path) -> None:
    _ = subprocess.run(command, cwd=cwd, check=True)


def _git_output(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


if __name__ == "__main__":
    main()
