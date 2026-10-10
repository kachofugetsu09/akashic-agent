from pathlib import Path
from core.common.unix_socket import socket_alias_path


def chat_socket_path(workspace: Path) -> Path:
    return socket_alias_path(workspace / "runtime", "web-chat.sock")


def dashboard_socket_path(workspace: Path) -> Path:
    return socket_alias_path(workspace / "runtime", "dashboard.sock")
