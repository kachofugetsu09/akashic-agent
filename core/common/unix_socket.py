"""为实际 owner 提供有界 Unix socket 路径与残留节点校验。"""
from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path


def socket_alias_path(directory: Path, name: str) -> Path:
    """实际节点仍在 owner 目录中，短别名避免超过 Unix 路径上限。"""

    # 1. 先固定实际目录，再计算同一目录的稳定短别名。
    directory.mkdir(parents=True, exist_ok=True)
    resolved_runtime = directory.resolve()

    # 2. 已有别名必须指向同一 owner，不覆盖其它节点。
    digest = hashlib.sha256(str(resolved_runtime).encode("utf-8")).hexdigest()[:20]
    alias_root = Path("/tmp") / f"akashic-web-{os.getuid()}"
    alias_root.mkdir(mode=0o700, exist_ok=True)
    alias = alias_root / digest
    if alias.is_symlink():
        if alias.resolve() != resolved_runtime:
            raise RuntimeError(f"运行时 socket 别名指向错误目录: {alias}")
    elif alias.exists():
        raise RuntimeError(f"运行时 socket 别名已被非链接占用: {alias}")
    else:
        try:
            alias.symlink_to(resolved_runtime, target_is_directory=True)
        except FileExistsError:
            if not alias.is_symlink() or alias.resolve() != resolved_runtime:
                raise RuntimeError(f"运行时 socket 别名发布冲突: {alias}") from None
    return alias / name


def prepare_unix_socket(path: Path) -> None:
    """调用者已取得目录 ownership；只移除旧 Unix socket，不覆盖其它节点。"""

    # 1. 核对旧节点类型，实际 listener 的生命周期由调用者负责。
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        mode = path.lstat().st_mode
        if not stat.S_ISSOCK(mode):
            raise RuntimeError(f"运行时 socket 路径已被非 socket 占用: {path}")
        path.unlink()
