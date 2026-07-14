from __future__ import annotations

import os
import secrets
import stat
from pathlib import Path


def ensure_workspace_token(workspace: Path) -> str:
    """读取或创建仅供当前 workspace loopback 控制面使用的 token。"""

    path = workspace / ".app-server-token"
    workspace.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return read_workspace_token(workspace)
    token = secrets.token_urlsafe(32)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(token + "\n")
    return token


def read_workspace_token(workspace: Path) -> str:
    """读取已由 gateway 创建的 workspace token。"""

    path = workspace / ".app-server-token"
    if os.name != "nt" and stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise PermissionError(f"workspace token 权限必须为 0600: {path}")
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"workspace token 文件为空: {path}")
    return token
