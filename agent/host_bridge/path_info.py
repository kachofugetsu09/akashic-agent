"""Bounded, read-only path operations on the actual execution host."""

from __future__ import annotations

import heapq
import json
import os
from pathlib import Path
import stat
from typing import Literal
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
        if action not in {"resolve", "inspect", "browse", "read_text"}:
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


PathStatus = Literal[
    "available", "not_found", "not_directory", "permission_denied",
    "not_file", "too_large", "invalid_text", "io_error", "offline",
]


class DirectoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str
    path: str


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
