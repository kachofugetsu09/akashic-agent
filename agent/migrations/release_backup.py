"""Save one stopped release state before any data migration or selection write."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
from contextlib import closing
from pathlib import Path

_SQLITE = b"SQLite format 3\0"
_RUNTIME_FILES = {
    Path("workspace/.instance.lock"),
    Path("workspace/.supervisor.lock"),
    Path("workspace/.supervisor.pid"),
    Path("workspace/.runtime-ready.json"),
    Path("workspace/akashic.sock"),
    Path("plugin-home/.publication.lock"),
}


def _sqlite_sidecar(source: Path) -> bool:
    """Only a regular SQLite base makes a suffix file forensic evidence."""

    base = source.with_name(source.name[:-4])
    if not (base.exists() or base.is_symlink()):
        return False
    if not stat.S_ISREG(base.lstat().st_mode):
        return False
    with base.open("rb") as stream:
        return stream.read(len(_SQLITE)) == _SQLITE


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _copy_db(source: Path, target: Path) -> None:
    """Include committed WAL pages in a checked logical SQLite backup."""

    with closing(sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True)) as original:
        if original.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise RuntimeError(f"源 SQLite 损坏: {source}")
        with closing(sqlite3.connect(target)) as saved:
            original.backup(saved)
            if saved.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise RuntimeError(f"备份 SQLite 损坏: {target}")
    shutil.copymode(source, target)
    _sync(target)


def backup_release_state(state: Path, backup: Path) -> dict[str, object]:
    """Copy regular state and logical DBs into a new external recovery point."""

    state = state.absolute()
    backup = backup.absolute()
    if (state.is_symlink() or not state.is_dir() or state != state.resolve(strict=True)
        or backup.is_relative_to(state)):
        raise ValueError("release state 或恢复点路径无效")
    if backup.exists() or backup.is_symlink():
        raise FileExistsError(f"release 恢复点已存在: {backup}")
    if (backup.parent.is_symlink() or not backup.parent.is_dir()
        or backup.parent != backup.parent.resolve(strict=True)):
        raise ValueError("release 恢复点父目录无效")
    backup.mkdir(mode=0o700)
    records: list[dict[str, object]] = []
    sidecars: list[dict[str, object]] = []
    omitted: list[str] = []
    snapshot = backup / "state"
    snapshot.mkdir(mode=0o700)

    # 1. Physical owners are stopped and their maintenance locks are held.
    for current, dirs, files in os.walk(state, topdown=True, followlinks=False):
        root = Path(current)
        relative_root = root.relative_to(state)
        target_root = snapshot / relative_root
        target_root.mkdir(parents=True, exist_ok=True)
        for name in sorted(dirs):
            source = root / name
            if source.is_symlink():
                target = target_root / name
                target.symlink_to(os.readlink(source), target_is_directory=True)
                records.append({"path": (relative_root / name).as_posix(),
                                "kind": "symlink", "target": os.readlink(source)})
                dirs.remove(name)
        for name in sorted(files):
            source = root / name
            relative = relative_root / name
            target = target_root / name
            if relative in _RUNTIME_FILES:
                omitted.append(relative.as_posix())
                continue
            mode = source.lstat().st_mode
            if stat.S_ISLNK(mode):
                target.symlink_to(os.readlink(source))
                records.append({"path": relative.as_posix(), "kind": "symlink",
                                "target": os.readlink(source)})
                continue
            if not stat.S_ISREG(mode):
                omitted.append(relative.as_posix())
                continue
            if name.endswith(("-wal", "-shm")) and _sqlite_sidecar(source):
                forensic = backup / "sidecars" / relative
                forensic.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, forensic)
                _sync(forensic)
                sidecars.append({"path": relative.as_posix(), "sha256": _sha(forensic),
                                 "restore": False})
                continue
            before = source.stat()
            with source.open("rb") as stream:
                header = stream.read(len(_SQLITE))
            if header == _SQLITE:
                _copy_db(source, target)
                kind = "sqlite_logical"
            else:
                shutil.copy2(source, target)
                _sync(target)
                if _sha(source) != _sha(target):
                    raise RuntimeError(f"release 备份字节漂移: {source}")
                kind = "file"
            after = source.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino, after.st_size, after.st_mtime_ns
            ):
                raise RuntimeError(f"release 备份期间源改变: {source}")
            records.append({"path": relative.as_posix(), "kind": kind,
                            "sha256": _sha(target), "size": target.stat().st_size})

    # 2. Publish the manifest only after every copy and DB readback succeeded.
    manifest: dict[str, object] = {
        "version": 1, "source": str(state), "backup": str(snapshot),
        "files": records, "forensic_sidecars": sidecars, "omitted_runtime_paths": omitted,
        "restore": "Restore state/ while stopped; never replay forensic sidecars alone.",
    }
    manifest_path = backup / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    for current, _, _ in os.walk(backup, topdown=False):
        _sync_dir(Path(current))
    _sync_dir(backup.parent)
    return manifest
