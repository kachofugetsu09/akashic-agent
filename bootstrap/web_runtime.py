from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path


def chat_socket_path(workspace: Path) -> Path:
    return _socket_alias_directory(workspace) / "web-chat.sock"


def dashboard_socket_path(workspace: Path) -> Path:
    return _socket_alias_directory(workspace) / "dashboard.sock"


def _socket_alias_directory(workspace: Path) -> Path:
    """Expose the workspace runtime directory through a bounded Unix path."""

    # 1. Keep the socket node physically owned by the workspace runtime directory.
    runtime_directory = workspace / "runtime"
    runtime_directory.mkdir(parents=True, exist_ok=True)
    resolved_runtime = runtime_directory.resolve()

    # 2. Derive a stable short alias so deep workspaces remain valid AF_UNIX owners.
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
    return alias


def prepare_runtime_socket(path: Path) -> str:
    """Create the runtime directory and remove only a stale Unix socket."""

    # 1. Resolve the owner directory before touching an existing filesystem node.
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        mode = path.lstat().st_mode
        if not stat.S_ISSOCK(mode):
            raise RuntimeError(f"运行时 socket 路径已被非 socket 占用: {path}")
        path.unlink()

    # 2. Uvicorn owns creation and lifecycle after this boundary.
    return str(path)
