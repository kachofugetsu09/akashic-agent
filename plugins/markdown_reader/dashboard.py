"""Authenticated dashboard routes; paths refer to the execution host, not Core."""
from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import PurePosixPath
from urllib.parse import quote

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, Response, StreamingResponse

from agent.plugin_composition import DashboardContext
from plugins.host_execution.contract import FILE_IMPORT_BYTES, FILES, FileError, PathReader
from plugins.working_directory.contract import WORKING_DIRECTORY

inject = (FILES, WORKING_DIRECTORY)
PREVIEW_MAX_BYTES = 256 * 1024


def _detect_image_mime(head: bytes) -> str | None:
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    return None


def _workspace_candidate(context: DashboardContext) -> str | None:
    parent = context.data_root.parent
    return str(parent.parent) if parent.name == "plugin-data" else None


def register(app: FastAPI, context: DashboardContext) -> None:
    workspace_hint = _workspace_candidate(context)
    cached_static: dict[str, str | None] = {}

    async def static_roots(reader: PathReader) -> tuple[str | None, str | None, str | None]:
        if "ready" not in cached_static:
            ws: str | None = None
            mem: str | None = None
            home: str | None = None
            offline = False
            if workspace_hint:
                ws_info = await reader.read("inspect", workspace_hint)
                offline = offline or ws_info.status == "offline"
                if ws_info.status == "available" and ws_info.kind == "directory":
                    ws = ws_info.path
                    mem_info = await reader.read("inspect", f"{ws}/memory")
                    offline = offline or mem_info.status == "offline"
                    if mem_info.status == "available" and mem_info.kind == "directory":
                        mem = mem_info.path
            home_info = await reader.read("inspect", "~")
            offline = offline or home_info.status == "offline"
            if home_info.status == "available" and home_info.kind == "directory":
                home = home_info.path
            if offline:
                return ws, mem, home
            cached_static.update(ready="1", workspace=ws, memory=mem, home=home)
        return cached_static.get("workspace"), cached_static.get("memory"), cached_static.get("home")

    async def build_roots(reader: PathReader, session: str) -> tuple[str | None, str | None, list[dict[str, str]]]:
        directories = context.require(WORKING_DIRECTORY)
        workdir = directories.snapshot(session).path if session else None
        ws, mem, home = await static_roots(reader)
        roots: list[dict[str, str]] = []
        seen: set[str] = set()
        for root_id, label, candidate in (
            ("workdir", "会话目录", workdir),
            ("workspace", "工作区", ws),
            ("memory", "记忆", mem),
            ("home", "主目录", home),
        ):
            if candidate and candidate not in seen:
                seen.add(candidate)
                roots.append({"id": root_id, "label": label, "path": candidate})
        return workdir, (workdir or ws or home), roots

    async def resolve_path(reader: PathReader, session: str, raw_path: str, base: str | None) -> tuple[str | None, dict[str, object] | None]:
        directories = context.require(WORKING_DIRECTORY)
        if not raw_path:
            return base or "/", None
        if session and directories.snapshot(session).path and not raw_path.startswith(("/", "~")):
            try:
                target, _ = await directories.resolve_target(session, raw_path)
                return target, None
            except ValueError as error:
                return None, {"status": "invalid_path", "message": str(error)}
        resolved = await reader.read("resolve", raw_path, base_dir=base)
        if resolved.status != "available":
            return None, resolved.model_dump(exclude_none=True)
        return resolved.path, None

    @app.get("/api/dashboard/markdown_reader/files")
    async def files(
        session: str = Query(default="", max_length=512),
        path: str = Query(default="", max_length=4096),
        after: str | None = Query(default=None, max_length=512),
        hidden: bool = Query(default=False),
    ) -> dict[str, object]:
        async with context.require(FILES).paths() as reader:
            workdir, base, roots = await build_roots(reader, session)
            target, err = await resolve_path(reader, session, path.strip(), base)
            if err is not None or target is None:
                return {**(err or {"status": "invalid_path"}), "root": workdir, "roots": roots}
            action = "file_list_hidden" if hidden else "file_list"
            result = await reader.read(action, target, after=after, limit=200)
            if result.status == "not_directory" and after is None:
                inspected = await reader.read("inspect", target)
                if inspected.status == "available" and inspected.kind == "file":
                    parent_dir = str(PurePosixPath(inspected.path).parent)
                    listed = await reader.read(action, parent_dir, limit=200)
                    return {**listed.model_dump(exclude_none=True), "root": workdir, "roots": roots, "open_file": inspected.path}
            return {**result.model_dump(exclude_none=True), "root": workdir, "roots": roots}

    @app.get("/api/dashboard/markdown_reader/document")
    async def document(
        session: str = Query(default="", max_length=512),
        path: str = Query(min_length=1, max_length=4096),
        tail: bool = Query(default=False),
    ) -> dict[str, object]:
        async with context.require(FILES).paths() as reader:
            _, base, _ = await build_roots(reader, session)
            target, err = await resolve_path(reader, session, path.strip(), base)
            if err is not None or target is None:
                return err or {"status": "invalid_path"}
            result = await reader.read("file_tail" if tail else "file_read", target, max_bytes=PREVIEW_MAX_BYTES)
            return result.model_dump(exclude_none=True)

    @app.get("/api/dashboard/markdown_reader/raw")
    async def raw(
        session: str = Query(default="", max_length=512),
        path: str = Query(min_length=1, max_length=4096),
        download: bool = Query(default=False),
    ) -> Response:
        files_service = context.require(FILES)
        async with files_service.paths() as reader:
            _, base, _ = await build_roots(reader, session)
            target, err = await resolve_path(reader, session, path.strip(), base)
        if err is not None or target is None:
            return JSONResponse({"status": "invalid_path", **(err or {})}, status_code=400)
        async with files_service.bytes() as byte_reader:
            first = await byte_reader.read(target, offset=0)
            if isinstance(first, FileError):
                status_code = 413 if f"{FILE_IMPORT_BYTES // (1024 * 1024)} MiB" in first.text else 400
                return JSONResponse({"status": "error", "message": first.text}, status_code=status_code)
            image_mime = None if download else _detect_image_mime(first.data)
            media_type = image_mime or "application/octet-stream"
            mode = "inline" if image_mime else "attachment"
            headers = {
                "Content-Disposition": f"{mode}; filename*=UTF-8''{quote(first.filename)}",
                "Content-Length": str(first.size),
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "default-src 'none'",
                "Cache-Control": "private, no-store",
            }
            if len(first.data) >= first.size:
                return Response(content=first.data, media_type=media_type, headers=headers)

        async def stream() -> AsyncIterator[bytes]:
            yield first.data
            offset = len(first.data)
            async with files_service.bytes() as rest_reader:
                while offset < first.size:
                    chunk = await rest_reader.read(target, offset=offset, version=first.version)
                    if isinstance(chunk, FileError) or not chunk.data:
                        break
                    yield chunk.data
                    offset += len(chunk.data)

        return StreamingResponse(stream(), media_type=media_type, headers=headers)
