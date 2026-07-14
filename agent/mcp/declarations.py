from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True)
class WorkspaceMcpDeclarations:
    specs: dict[str, dict[str, Any]]
    revision: str
    watched_files: tuple[Path, ...]


def load_workspace_mcp_declarations(
    root: Path,
    *,
    mcp_root: Path | None = None,
) -> WorkspaceMcpDeclarations:
    """严格解析 workspace MCP 声明并计算内容 revision。"""

    # 1. 解析并校验每个独立 server 声明
    root = root.resolve(strict=False)
    safe_root = (mcp_root or root.parent).resolve(strict=False)
    if not root.is_relative_to(safe_root):
        raise ValueError(f"MCP 声明目录越界: {root}")
    specs: dict[str, dict[str, Any]] = {}
    watched: set[Path] = set()
    normalized: list[dict[str, object]] = []
    for path in sorted(root.glob("*.toml")) if root.is_dir() else ():
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        if set(raw) - {"schema_version", "name", "command", "cwd", "env", "watch_paths"}:
            raise ValueError(f"MCP 声明包含未知字段: {path}")
        name = raw.get("name")
        schema_version = raw.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or schema_version != 1
            or not isinstance(name, str)
            or name != path.stem
        ):
            raise ValueError(f"MCP 声明 schema_version/name 无效: {path}")
        if name in specs:
            raise ValueError(f"MCP server 名称重复: {name}")
        command = raw.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(item, str) and item for item in command
        ):
            raise ValueError(f"MCP command 无效: {path}")
        env = raw.get("env", {})
        if not isinstance(env, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in env.items()
        ):
            raise ValueError(f"MCP env 无效: {path}")
        base = path.parent.resolve()
        cwd = _resolve_inside(
            base,
            safe_root,
            raw.get("cwd"),
            field="cwd",
            path=path,
        )
        watch_raw = raw.get("watch_paths", [])
        if not isinstance(watch_raw, list) or not all(
            isinstance(item, str) and item for item in watch_raw
        ):
            raise ValueError(f"MCP watch_paths 无效: {path}")
        watch_paths_list: list[Path] = []
        for item in cast(list[str], watch_raw):
            resolved_watch = cast(
                Path,
                _resolve_inside(
                    base,
                    safe_root,
                    item,
                    field="watch_paths",
                    path=path,
                ),
            )
            watch_paths_list.append(resolved_watch)
        watch_paths = tuple(watch_paths_list)
        watched.update(watch_paths)
        spec: dict[str, Any] = {"command": list(command), "env": dict(env)}
        if cwd is not None:
            spec["cwd"] = str(cwd)
        specs[cast(str, name)] = spec
        normalized.append(
            {"name": name, **spec, "watch_paths": [str(p) for p in watch_paths]}
        )

    # 2. revision 只取规范化声明和 watch 文件内容
    digest = hashlib.sha256(
        json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode()
    )
    for path in sorted(watched):
        _hash_watch_path(digest, path)
    return WorkspaceMcpDeclarations(specs, digest.hexdigest(), tuple(sorted(watched)))


def declarations_input_revision(
    root: Path,
    *,
    mcp_root: Path | None = None,
) -> str:
    """计算声明文件与当前有效 watch paths 的轮询输入指纹。"""

    digest = hashlib.sha256()
    root = root.resolve(strict=False)
    for path in sorted(root.glob("*.toml")) if root.is_dir() else ():
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    try:
        desired = load_workspace_mcp_declarations(root, mcp_root=mcp_root)
    except (OSError, ValueError):
        return digest.hexdigest()
    for path in desired.watched_files:
        _hash_watch_path(digest, path)
    return digest.hexdigest()


def _hash_watch_path(digest: Any, path: Path) -> None:
    if path.is_file():
        _hash_watch_entry(digest, b"file", str(path).encode(), path.read_bytes())
        return
    if path.is_dir():
        _hash_watch_entry(digest, b"directory", str(path).encode(), b"")
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            _hash_watch_entry(
                digest,
                b"child",
                str(child.relative_to(path)).encode(),
                child.read_bytes(),
            )
        return
    _hash_watch_entry(digest, b"missing", str(path).encode(), b"")


def _hash_watch_entry(
    digest: Any,
    entry_type: bytes,
    entry_path: bytes,
    content: bytes,
) -> None:
    """以长度分帧编码 watch entry，避免路径与内容拼接碰撞。"""

    digest.update(len(entry_type).to_bytes(4, "big"))
    digest.update(entry_type)
    digest.update(len(entry_path).to_bytes(8, "big"))
    digest.update(entry_path)
    digest.update(len(content).to_bytes(8, "big"))
    digest.update(content)


def _resolve_inside(
    base: Path,
    safe_root: Path,
    raw: object,
    *,
    field: str,
    path: Path,
) -> Path | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"MCP {field} 无效: {path}")
    resolved = (base / raw).resolve(strict=False)
    if not resolved.is_relative_to(safe_root):
        raise ValueError(f"MCP {field} 越界: {path}")
    return resolved
