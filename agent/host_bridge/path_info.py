"""Bounded, read-only path operations on the actual execution host."""

from __future__ import annotations

import heapq
import json
import os
from pathlib import Path
import stat
from typing import Literal, cast
from pydantic import BaseModel, ConfigDict
from agent.host_bridge.client import HostBridgeRpcError
from agent.host_bridge.factory import build_file_bridge

from core.common.file_io import run_file_io


class PathInfoOperation:
    async def execute(
        self, action: str, path: str, *, base_dir: str | None = None,
        after: str | None = None, limit: int = 100, max_bytes: int = 32768,
    ) -> str:
        """Validate the request once and run all disk work in a bounded worker."""
        if action not in {"resolve", "inspect", "browse", "read_text", "agents_chain"}:
            raise ValueError("Unknown path operation")
        if not isinstance(path, str) or not path or "\x00" in path:
            raise ValueError("Path must be nonempty text")
        if base_dir is not None and not Path(base_dir).is_absolute():
            raise ValueError("base_dir must be absolute")
        if type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError("Directory page limit must be 1..200")
        if type(max_bytes) is not int or not 1 <= max_bytes <= 32768:
            raise ValueError("Read limit must be 1..32768 bytes")
        if after is not None and (not isinstance(after, str) or "/" in after):
            raise ValueError("Directory cursor must be a file name")
        return await run_file_io(lambda: self._read(
            action, path, base_dir, after, limit, max_bytes,
        ))

    def _read(
        self, action: str, path: str, base_dir: str | None,
        after: str | None, limit: int, max_bytes: int,
    ) -> str:
        """Resolve on this host and return explicit availability or bounded data."""
        # 1. Resolve relative paths and home on the execution host, never in Core.
        target = Path(path).expanduser()
        if not target.is_absolute() and base_dir is not None:
            target = Path(base_dir) / target
        try:
            target = target.resolve(strict=action != "resolve")
            result: dict[str, object] = {"path": str(target), "status": "available"}
            if action == "resolve":
                return json.dumps(result, ensure_ascii=False)
            mode = target.stat().st_mode
            kind = "directory" if stat.S_ISDIR(mode) else "file" if stat.S_ISREG(mode) else "other"
            result["kind"] = kind
            access = os.R_OK | (os.X_OK if kind == "directory" else 0)
            if not os.access(target, access):
                raise PermissionError(f"Cannot access {target}")
            # 2. Only browse direct directories; pagination never walks a tree.
            if action == "browse":
                if kind != "directory":
                    raise NotADirectoryError(str(target))
                with os.scandir(target) as entries:
                    names = heapq.nsmallest(limit + 1, (
                        entry.name for entry in entries
                        if (after is None or entry.name > after) and entry.is_dir()
                    ))
                result["items"] = [{"name": name, "path": str(target / name)} for name in names[:limit]]
                result["after"] = names[limit - 1] if len(names) > limit else None
                result["parent"] = str(target.parent)
            # 3. Never read a device or FIFO; over-limit files have no partial text.
            if action == "read_text":
                if kind != "file":
                    return json.dumps({**result, "status": "not_file"}, ensure_ascii=False)
                with target.open("rb") as stream:
                    raw = stream.read(max_bytes + 1)
                if len(raw) > max_bytes:
                    return json.dumps({**result, "status": "too_large"}, ensure_ascii=False)
                result.update(text=raw.decode("utf-8-sig"), bytes=len(raw))
            # 4. AGENTS 规则链在一次磁盘工作内完成全部探测，避免逐层往返。
            if action == "agents_chain" and kind == "directory":
                result["chain"] = _agents_chain_memoized(target, max_bytes)
        except FileNotFoundError as error:
            result = {"path": path, "status": "not_found", "error": str(error)}
        except NotADirectoryError as error:
            result = {"path": path, "status": "not_directory", "error": str(error)}
        except PermissionError as error:
            result = {"path": path, "status": "permission_denied", "error": str(error)}
        except UnicodeDecodeError as error:
            result = {"path": path, "status": "invalid_text", "error": str(error)}
        except OSError as error:
            result = {"path": path, "status": "io_error", "error": str(error)}
        return json.dumps(result, ensure_ascii=False)


def _probe_marker(path: Path) -> tuple[str, str | None]:
    """与 inspect 相同的可用性判定；只关心路径是否存在且可进入。"""
    try:
        mode = path.stat().st_mode
        access = os.R_OK | (os.X_OK if stat.S_ISDIR(mode) else 0)
        if not os.access(path, access):
            raise PermissionError(f"Cannot access {path}")
        return "available", None
    except FileNotFoundError:
        return "not_found", None
    except NotADirectoryError:
        return "not_directory", None
    except PermissionError as error:
        return "permission_denied", str(error)
    except OSError as error:
        return "io_error", str(error)


def _file_stamp(path: Path) -> tuple[int, int, int] | None:
    """规则文件内容身份：mtime+size+mode 不变即视为同一文本（与技能目录签名同标准）。"""
    try:
        info = path.stat()
    except OSError:
        return None
    return (info.st_mtime_ns, info.st_size, info.st_mode)


# agents_chain 签名备忘：逐层 .git 探测与规则文件 stat 不变时复用上次读取，
# 每轮材料准备不再重读相同规则文本；签名变化（含 .git 增删）即失效。
_agents_chain_memo: tuple[tuple[object, ...], dict[str, object]] | None = None


def _agents_signature(current: Path) -> tuple[object, ...]:
    """沿目录链收集探测签名；走到第一个有 .git 的层为止，与 _agents_chain 同链。"""
    layers: list[tuple[object, ...]] = []
    for parent in (current, *current.parents):
        layers.append((
            str(parent),
            _file_stamp(parent / ".git"),
            _file_stamp(parent / "AGENTS.override.md"),
            _file_stamp(parent / "AGENTS.md"),
        ))
        if layers[-1][1] is not None:
            break
    return tuple(layers)


def _read_rule(path: Path, budget: int) -> dict[str, object]:
    """与 read_text 相同的有界读取；not_found 表示本层没有该规则文件。"""
    try:
        target = path.resolve(strict=True)
        if not stat.S_ISREG(target.stat().st_mode):
            return {"path": str(target), "status": "not_file"}
        if not os.access(target, os.R_OK):
            raise PermissionError(f"Cannot access {target}")
        with target.open("rb") as stream:
            raw = stream.read(budget + 1)
        if len(raw) > budget:
            return {"path": str(target), "status": "too_large"}
        return {
            "path": str(target), "status": "available",
            "text": raw.decode("utf-8-sig"), "bytes": len(raw),
        }
    except FileNotFoundError:
        return {"path": str(path), "status": "not_found"}
    except NotADirectoryError as error:
        return {"path": str(path), "status": "not_directory", "error": str(error)}
    except PermissionError as error:
        return {"path": str(path), "status": "permission_denied", "error": str(error)}
    except UnicodeDecodeError as error:
        return {"path": str(path), "status": "invalid_text", "error": str(error)}
    except OSError as error:
        return {"path": str(path), "status": "io_error", "error": str(error)}


def _agents_chain_memoized(current: Path, max_bytes: int) -> dict[str, object]:
    """签名未变复用上次的链读取；memo 只覆盖 chain，路径与目录状态仍逐次实测。"""
    global _agents_chain_memo
    signature = (str(current), max_bytes, _agents_signature(current))
    if _agents_chain_memo is not None and _agents_chain_memo[0] == signature:
        return _agents_chain_memo[1]
    chain = _agents_chain(current, max_bytes)
    _agents_chain_memo = (signature, chain)
    return chain


def _agents_chain(current: Path, max_bytes: int) -> dict[str, object]:
    """在最近 Git root 到当前目录的链上读取 AGENTS 规则，语义与逐层 read_text 一致。"""
    root = current
    for parent in (current, *current.parents):
        marker = parent / ".git"
        status, error = _probe_marker(marker)
        if status == "available":
            root = parent
            break
        if status != "not_found":
            return {"root": None, "files": [], "failure": {
                "kind": "probe", "path": str(marker), "status": status, "error": error,
            }}
    layers = [root]
    for part in current.relative_to(root).parts:
        layers.append(layers[-1] / part)
    files: list[dict[str, object]] = []
    used = 0
    for layer in layers:
        for name in ("AGENTS.override.md", "AGENTS.md"):
            entry = _read_rule(layer / name, max(1, max_bytes - used))
            if entry["status"] == "not_found":
                continue
            if entry["status"] != "available":
                return {"root": str(root), "files": files, "failure": {
                    "kind": "read", "path": entry["path"],
                    "status": entry["status"], "error": entry.get("error"),
                }}
            used += cast(int, entry["bytes"])
            if used > max_bytes:
                return {"root": str(root), "files": files,
                        "failure": {"kind": "budget", "path": None, "status": None, "error": None}}
            files.append({"path": entry["path"], "text": entry["text"], "bytes": entry["bytes"]})
            break
    return {"root": str(root), "files": files, "failure": None}


PathStatus = Literal[
    "available", "not_found", "not_directory", "permission_denied",
    "not_file", "too_large", "invalid_text", "io_error", "offline",
]


class DirectoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str
    path: str


class AgentsChainFile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str
    text: str
    bytes: int


class AgentsChainFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["probe", "read", "budget"]
    path: str | None = None
    status: str | None = None
    error: str | None = None


class AgentsChain(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    root: str | None = None
    files: list[AgentsChainFile]
    failure: AgentsChainFailure | None = None


class PathInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str
    status: PathStatus
    kind: Literal["file", "directory", "other"] | None = None
    error: str | None = None
    text: str | None = None
    bytes: int | None = None
    items: list[DirectoryEntry] | None = None
    after: str | None = None
    parent: str | None = None
    chain: AgentsChain | None = None


class PathAccess:
    def __init__(self):
        self._bridge = build_file_bridge()

    async def __aenter__(self) -> PathAccess:
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._bridge is not None:
            try:
                _ = await self._bridge.shutdown()
            except HostBridgeRpcError as error:
                if not error.transient:
                    raise
                # This manager only reads files; the host lease owns its expiry.
                await self._bridge.close_transport()

    async def read(
        self, action: str, path: str, *, base_dir: str | None = None,
        after: str | None = None, limit: int = 100, max_bytes: int = 32768,
    ) -> PathInfo:
        """Validate backend data; transport loss remains distinct from a missing path."""
        if self._bridge is None:
            raw = await PathInfoOperation().execute(
                action, path, base_dir=base_dir, after=after,
                limit=limit, max_bytes=max_bytes,
            )
        else:
            try:
                raw = await self._bridge.execute_file_tool(
                    "path_info", allowed_dir=None,
                    arguments={"action": action, "path": path,
                               "base_dir": base_dir, "after": after,
                               "limit": limit, "max_bytes": max_bytes},
                )
            except HostBridgeRpcError as error:
                if not error.transient:
                    raise
                return PathInfo(path=path, status="offline", error=str(error))
        if not isinstance(raw, str):
            raise TypeError("Path backend must return JSON text")
        return PathInfo.model_validate_json(raw)
