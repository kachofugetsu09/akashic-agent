"""Directory validation uses the public physical filesystem contract."""

from plugins.host_execution.contract import PathInfo

def check_directory(info: PathInfo) -> str:
    """Reject unavailable targets before any persistent directory change."""
    if info.status != "available":
        raise ValueError(f"目录不可用 ({info.status}): {info.path}; {info.error or ''}")
    if info.kind != "directory":
        raise ValueError(f"路径不是目录: {info.path}")
    return info.path
