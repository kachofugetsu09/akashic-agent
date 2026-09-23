#!/usr/bin/env python3
"""Explicitly roll back one unlinked plugin install while the runtime is stopped."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from agent.plugin_composition.archive import encode_tree, sync_directory, tree_entries
from agent.plugins.artifacts import ArtifactPointer, ArtifactPointers, pointer_state_path, read_pointers, resolve_pointer
from agent.plugins.input_preparation import _source_revision
from agent.plugins.manifest import load_plugin_manifest, manifest_path
from agent.plugins.python_environment import ENVIRONMENT_FILE
from agent.plugins.reload_journal import JournalPreflight, ReloadJournal
from agent.plugins.selection import PluginSelection, SelectionConflictError
from agent.plugins.static_manifest import load_static_plugin_manifest
from agent.plugins.update_rollback import UpdateRollback, pointer_value
from bootstrap.workspace_lock import PluginPublicationLock, WorkspaceMaintenanceLock

_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CODE_EXCLUDE = frozenset({".venv", "node_modules", ENVIRONMENT_FILE})


def _directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"需要实际目录: {path}")


def _file_bytes(path: Path) -> bytes | None:
    """Read a regular file without following a replacement symlink."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"需要普通文件: {path}")
        with os.fdopen(os.dup(fd), "rb") as stream:
            return stream.read()
    finally:
        os.close(fd)


def _save(path: Path, content: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("xb") as stream:
        path.chmod(0o600)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    if path.read_bytes() != content:
        raise RuntimeError(f"恢复点写入后不一致: {path}")


def _code_identity(path: Path) -> tuple[str, str]:
    """Read the exact code identity used by the archive owner, without writing it."""
    code = hashlib.sha256(encode_tree(tree_entries(path, exclude=_CODE_EXCLUDE))).hexdigest()
    return code, _source_revision(path)


def _selected_code(selection: PluginSelection, root_ref: str | None, plugin_id: str) -> tuple[str, str] | None:
    """Check every selected code closure and return only the target's code facts."""
    if root_ref is None:
        return None
    root = selection.archive.read_descriptor(root_ref)
    components = root.get("components")
    if root.get("version") != 1 or not isinstance(components, tuple):
        raise ValueError("stable 完整记录无效")
    found: set[str] = set()
    selected = None
    for ref in components:
        if not isinstance(ref, str):
            raise ValueError("stable component ref 无效")
        descriptor = selection.archive.read_descriptor(ref)
        name = descriptor.get("plugin_id")
        code_ref = descriptor.get("code")
        revision = descriptor.get("source_revision")
        if descriptor.get("version") != 4 or not isinstance(name, str) or not isinstance(code_ref, str) or not isinstance(revision, str):
            raise ValueError("selected descriptor 无效")
        if name in found:
            raise ValueError(f"stable 重复插件: {name}")
        found.add(name)
        code = selection.archive.open(code_ref)
        identity = load_static_plugin_manifest(code)
        if identity.name != name.split("@", 1)[0] or _source_revision(code) != revision:
            raise ValueError(f"selected code/source 身份不一致: {name}")
        if name == plugin_id:
            if descriptor.get("source_type") != "installed":
                raise ValueError("目标 selected 输入不是 installed source")
            selected = (code_ref, revision)
    return selected


def _row_base(update: UpdateRollback, plugins_home: Path) -> Path:
    name, separator, marketplace = update.plugin_id.rpartition("@")
    if not separator or _SEGMENT.fullmatch(name) is None or _SEGMENT.fullmatch(marketplace) is None:
        raise ValueError("插件更新身份无效")
    base = plugins_home / "cache" / marketplace / name
    for directory in (plugins_home / "cache", base.parent, base):
        _directory(directory)
    if update.plugin_base != base or base.resolve() != base:
        raise RuntimeError("插件更新恢复点不属于当前插件目录")
    candidate = update.candidate.path
    if candidate is None or re.fullmatch(r"\.artifacts/[A-Za-z0-9][A-Za-z0-9._-]*", candidate) is None:
        raise ValueError("插件 candidate pointer 无效")
    return base


def _previous_identity(update: UpdateRollback, base: Path) -> tuple[tuple[str, str] | None, tuple[str, str] | None]:
    """Verify every old target before a pointer can be written back to it."""
    if update.previous is None:
        return None, None
    name = update.plugin_id.rpartition("@")[0]
    result = []
    for pointer in (update.previous.stable, update.previous.latest):
        target = resolve_pointer(base, pointer)
        if target is None:
            result.append(None)
            continue
        if load_static_plugin_manifest(target).name != name:
            raise ValueError("旧 artifact 静态身份与恢复点不符")
        result.append(_code_identity(target))
    return result[0], result[1]


def _current_inputs(
    update: UpdateRollback, selection: PluginSelection, root_ref: str | None, plugins_home: Path,
    *, terminal: bool,
) -> tuple[ArtifactPointers | None, dict[str, bool], tuple[tuple[str, str] | None, tuple[str, str] | None]]:
    """Check selection, old artifacts, pointer shape, and manifest before a write."""
    base = _row_base(update, plugins_home)
    previous_identity = _previous_identity(update, base)
    selected = _selected_code(selection, root_ref, update.plugin_id)
    if update.previous is None:
        if selected is not None:
            raise RuntimeError("首装回退不能改变已选插件的安装输入")
    elif selected is not None and selected != previous_identity[0]:
        raise RuntimeError("selected code/source 不是恢复点的旧 stable artifact")
    pointers = read_pointers(base)
    if pointers is not None:
        plugin_name = update.plugin_id.rpartition("@")[0]
        for pointer in (pointers.stable, pointers.latest):
            target = resolve_pointer(base, pointer)
            if target is not None and load_static_plugin_manifest(target).name != plugin_name:
                raise ValueError("当前 artifact 静态身份与恢复点不符")
    previous = update.previous
    staged = ArtifactPointers(
        ArtifactPointer(None if previous is None else previous.stable.path), update.candidate,
    )
    collapsed = ArtifactPointers(update.candidate, update.candidate)
    allowed = (previous,) if terminal else (previous, staged, collapsed)
    if pointers not in allowed:
        raise RuntimeError("插件指针已被其他操作改变，不能覆盖")
    _file_bytes(manifest_path(plugins_home))
    entries = load_plugin_manifest(plugins_home)
    enabled = entries.get(update.plugin_id)
    if terminal:
        if enabled != update.previous_enabled:
            raise RuntimeError("已回退记录的 manifest 后置状态已漂移")
    elif enabled not in (update.previous_enabled, True):
        raise RuntimeError("插件启用状态已被其他操作改变，不能覆盖")
    return pointers, entries, previous_identity


def _backup_dir(backup_dir: Path, workspace: Path, plugins_home: Path) -> Path:
    backup_dir = backup_dir.absolute()
    if backup_dir.is_relative_to(workspace) or backup_dir.is_relative_to(plugins_home):
        raise ValueError("恢复点必须在 workspace 和 plugin-home 外")
    _directory(backup_dir.parent)
    backup_dir = backup_dir.parent.resolve() / backup_dir.name
    if backup_dir.is_relative_to(workspace) or backup_dir.is_relative_to(plugins_home):
        raise ValueError("恢复点父目录不能链接到运行数据内")
    if backup_dir.exists() or backup_dir.is_symlink():
        raise FileExistsError(f"恢复点已存在: {backup_dir}")
    return backup_dir


def _sources(workspace: Path, plugins_home: Path, pointer: Path, selection: PluginSelection) -> dict[Path, str]:
    journal = workspace / "runtime/plugin-reloads.sqlite3"
    return {
        selection.path: "workspace/runtime/plugin-stable.json",
        manifest_path(plugins_home): "plugins/manifest.toml",
        pointer: "plugins/target/.pointers.json",
        journal: "workspace/runtime/plugin-reloads.sqlite3.raw",
        Path(f"{journal}-wal"): "workspace/runtime/plugin-reloads.sqlite3-wal.raw",
        Path(f"{journal}-shm"): "workspace/runtime/plugin-reloads.sqlite3-shm.raw",
    }


def _read_sources(sources: dict[Path, str]) -> dict[Path, bytes | None]:
    return {source: _file_bytes(source) for source in sources}


def _backup(
    backup_dir: Path, sources: dict[Path, str], observed: dict[Path, bytes | None],
    *, journal: JournalPreflight, workspace: Path, plugins_home: Path,
    root_ref: str | None, update: UpdateRollback,
) -> None:
    """Save file bytes and the WAL-aware logical DB before the first rollback write."""
    backup_dir.mkdir(mode=0o700)
    sync_directory(backup_dir.parent)
    files: list[dict[str, object]] = []
    for source, relative in sources.items():
        content = observed[source]
        if content is None:
            files.append({"source": str(source), "backup": None})
            continue
        target = backup_dir / relative
        _save(target, content)
        files.append({"source": str(source), "backup": relative,
                      "sha256": hashlib.sha256(content).hexdigest()})
    logical = backup_dir / "workspace/runtime/plugin-reloads.sqlite3"
    logical.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    journal.backup_to(logical)
    logical.chmod(0o600)
    with logical.open("rb") as stream:
        os.fsync(stream.fileno())
    files.append({"source": str(workspace / "runtime/plugin-reloads.sqlite3"),
                  "backup": "workspace/runtime/plugin-reloads.sqlite3",
                  "sha256": hashlib.sha256(logical.read_bytes()).hexdigest()})
    _save(backup_dir / "recovery.json", json.dumps({
        "purpose": "stopped-exact-install-rollback; restore SQLite from logical backup, not raw WAL/SHM",
        "workspace": str(workspace), "plugins_home": str(plugins_home),
        "update_id": update.update_id, "plugin_id": update.plugin_id,
        "root_ref": root_ref, "phase": update.phase, "previous": pointer_value(update.previous),
        "candidate": update.candidate.path, "previous_enabled": update.previous_enabled,
        "reload_tx_id": update.reload_tx_id, "input_ref": update.input_ref,
        "prior_error": update.error, "files": files,
    }, ensure_ascii=False, indent=2).encode())
    for current, _, _ in os.walk(backup_dir, topdown=False):
        sync_directory(Path(current))
    if _read_sources(sources) != observed:
        raise RuntimeError("恢复点制作期间源文件改变")


@contextmanager
def _stage(name: str) -> Iterator[None]:
    """Keep the original exception and record which step failed."""
    try:
        yield
    except Exception as error:
        error.add_note(f"rollback_stage={name}")
        raise


def rollback_plugin_install(
    *, workspace: Path, plugins_home: Path, update_id: str,
    expected_root_ref: str | None, backup_dir: Path,
) -> dict[str, object]:
    """Roll back one exact isolated install under stopped workspace and home owners."""
    if not update_id or update_id.strip() != update_id:
        raise ValueError("update_id 必须是非空且无首尾空白的字符串")
    if expected_root_ref is not None and _SHA256.fullmatch(expected_root_ref) is None:
        raise ValueError("expected_root_ref 必须是完整 SHA-256 或 null")
    _directory(workspace)
    _directory(plugins_home)
    workspace, plugins_home = workspace.resolve(), plugins_home.resolve()
    maintenance = WorkspaceMaintenanceLock(workspace)
    maintenance.acquire()
    try:
        publication = PluginPublicationLock(plugins_home)
        publication.acquire()
        try:
            selection = PluginSelection(workspace)
            root_ref = selection.read()
            if root_ref != expected_root_ref:
                raise SelectionConflictError("stable 基线与 expected_root_ref 不一致")
            with ReloadJournal.inspect_existing(workspace) as preflight:
                if preflight.pending_recovery:
                    raise RuntimeError("pending reload 须先由原 owner 结算")
                update = preflight.update(update_id)
                if update.reload_tx_id is not None or update.input_ref is not None:
                    raise RuntimeError("指定记录不是孤立安装")
                if update.phase not in ("armed", "rolled_back"):
                    raise RuntimeError("指定记录已提交，不能回退")
                terminal = update.phase == "rolled_back"
                pointer = pointer_state_path(_row_base(update, plugins_home))
                before_pointer, before_manifest, previous_identity = _current_inputs(
                    update, selection, root_ref, plugins_home, terminal=terminal,
                )
                selected_before = _selected_code(selection, root_ref, update.plugin_id)
                if terminal:
                    historical_pair = update.previous is not None and update.previous.stable != update.previous.latest
                    remaining = tuple(item.update_id for item in preflight.armed_updates)
                    return {"status": "already_rolled_back", "update_id": update_id,
                            "plugin_id": update.plugin_id, "root_ref": root_ref,
                            "backup_dir": None,
                            "remaining_armed": remaining,
                            "further_recovery_required": bool(remaining or historical_pair),
                            "runtime_started": False}
                backup_dir = _backup_dir(backup_dir, workspace, plugins_home)
                sources = _sources(workspace, plugins_home, pointer, selection)
                observed = _read_sources(sources)
                if observed[selection.path] is None or observed[workspace / "runtime/plugin-reloads.sqlite3"] is None:
                    raise RuntimeError("stable 或 reload journal 缺失")
                with _stage("backup"):
                    _backup(backup_dir, sources, observed, journal=preflight, workspace=workspace,
                            plugins_home=plugins_home, root_ref=root_ref, update=update)

            # The snapshot closes before the real SQLite writer opens. Recheck every input.
            with _stage("recheck"):
                with ReloadJournal.inspect_existing(workspace) as rechecked:
                    if rechecked.pending_recovery or rechecked.update(update_id) != update:
                        raise RuntimeError("插件安装恢复点在备份后改变")
                    if _read_sources(sources) != observed or selection.read() != root_ref:
                        raise RuntimeError("正式输入在备份后改变")
                    pointer_now, manifest_now, identity_now = _current_inputs(
                        update, selection, root_ref, plugins_home, terminal=False,
                    )
                    if (pointer_now, manifest_now, identity_now) != (before_pointer, before_manifest, previous_identity):
                        raise RuntimeError("插件安装输入在备份后改变")

            error = "stopped explicit rollback" + (f"; previous error: {update.error}" if update.error else "")
            with _stage("rollback_write"):
                ReloadJournal(workspace).rollback_install_update(plugins_home, expected=update, error=error)
            with _stage("postcheck"):
                with ReloadJournal.inspect_existing(workspace) as after:
                    settled = after.update(update_id)
                    if settled.phase != "rolled_back" or settled.error != error:
                        raise RuntimeError("指定插件安装回退未写入终态")
                    remaining = tuple(item.update_id for item in after.armed_updates)
                    if tuple(item for item in after.armed_updates if item.update_id != update_id) != tuple(
                        item for item in preflight.armed_updates if item.update_id != update_id
                    ):
                        raise RuntimeError("其他 armed 安装记录已改变")
                if selection.read() != root_ref or _selected_code(selection, root_ref, update.plugin_id) != selected_before:
                    raise RuntimeError("stable Root 在回退期间改变")
                pointer_after, manifest_after, identity_after = _current_inputs(
                    settled, selection, root_ref, plugins_home, terminal=True,
                )
                if pointer_after != update.previous or identity_after != previous_identity:
                    raise RuntimeError("旧完整指针对未恢复")
                if {key: value for key, value in manifest_after.items() if key != update.plugin_id} != {
                    key: value for key, value in before_manifest.items() if key != update.plugin_id
                }:
                    raise RuntimeError("其他插件 manifest 条目改变")
            return {"status": "rolled_back", "update_id": update_id,
                    "plugin_id": update.plugin_id, "root_ref": root_ref,
                    "backup_dir": str(backup_dir), "remaining_armed": remaining,
                    "further_recovery_required": bool(
                        remaining or (update.previous is not None and update.previous.stable != update.previous.latest)
                    ), "runtime_started": False}
        finally:
            publication.release()
    finally:
        maintenance.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--plugins-home", type=Path, required=True)
    parser.add_argument("--update-id", required=True)
    parser.add_argument("--expected-root-ref", required=True, help="完整 SHA-256 或字面量 null")
    parser.add_argument("--backup-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = rollback_plugin_install(
            workspace=args.workspace, plugins_home=args.plugins_home, update_id=args.update_id,
            expected_root_ref=None if args.expected_root_ref == "null" else args.expected_root_ref,
            backup_dir=args.backup_dir,
        )
    except Exception as error:
        notes = vars(error).get("__notes__", ())
        stage = next((note.partition("=")[2] for note in notes if note.startswith("rollback_stage=")), "preflight")
        print(json.dumps({"status": "failed", "update_id": args.update_id,
                          "backup_dir": str(args.backup_dir), "error_type": type(error).__name__,
                          "error": str(error), "stage": stage}, ensure_ascii=False))
        parser.exit(1)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
