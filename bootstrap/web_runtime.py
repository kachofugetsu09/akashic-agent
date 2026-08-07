from __future__ import annotations

import stat
from pathlib import Path


def chat_socket_path(workspace: Path) -> Path:
    return workspace / "runtime" / "web-chat.sock"


def dashboard_socket_path(workspace: Path) -> Path:
    return workspace / "runtime" / "dashboard.sock"


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
