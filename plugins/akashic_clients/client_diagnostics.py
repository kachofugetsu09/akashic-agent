from __future__ import annotations

import json
import logging
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, Field

from core.common.file_io import run_file_io

from .static import current_runtime_commit

# 浏览器端打点：慢交互逐条落盘给夜间巡检算 p75 与回归；前端报错进运行日志，走现有报错巡检。
logger = logging.getLogger("akashic.web.client")

RETAIN_DAYS = 14
DAY_FILE_LIMIT = 32 * 1024 * 1024
_SHELL_UA = re.compile(r"AkashicShell/([\w.+-]{1,32})")
_CHROME_UA = re.compile(r"Chrome/(\d+)")

Ms = Annotated[float, Field(ge=0, le=600_000)]


class InteractionRecord(BaseModel):
    kind: Literal["interaction"]
    at: float
    action: str = Field(max_length=120)
    event: str = Field(max_length=32)
    duration: Ms
    input_delay: Ms
    processing: Ms
    presentation: Ms
    script: Ms | None = None
    style_layout: Ms | None = None
    render: Ms | None = None
    blocking: Ms | None = None
    script_source: str | None = Field(default=None, max_length=240)
    dom: int = Field(ge=0, le=10_000_000)
    width: int = Field(ge=0, le=20_000)
    touch: bool
    page_release: str | None = Field(default=None, max_length=64)


class ErrorRecord(BaseModel):
    kind: Literal["error"]
    at: float
    message: str = Field(max_length=1000)
    stack: str = Field(default="", max_length=6000)
    source: str = Field(default="", max_length=240)
    count: int = Field(default=1, ge=1, le=100_000)
    page_release: str | None = Field(default=None, max_length=64)


class DiagnosticsBatch(BaseModel):
    records: list[Annotated[InteractionRecord | ErrorRecord, Field(discriminator="kind")]] = Field(max_length=200)


# 识别客户端：Android 壳在 UA 末尾带 AkashicShell/<版本>，其余按浏览器网页算。
def client_label(user_agent: str) -> str:
    shell = _SHELL_UA.search(user_agent)
    if shell is not None:
        return f"shell/{shell.group(1)}"
    chrome = _CHROME_UA.search(user_agent)
    form = "mobile" if "Mobile" in user_agent else "desktop"
    return f"web-{form}/chrome{chrome.group(1)}" if chrome else f"web-{form}"


class InteractionLog:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._pruned: date | None = None

    # 按天追加 JSONL；跨天时顺手删掉超出保留期的文件，单日文件超限后丢弃新记录。
    def append(self, lines: list[str]) -> None:
        today = datetime.now(UTC).date()
        self.directory.mkdir(parents=True, exist_ok=True)
        if self._pruned != today:
            self._prune(today)
            self._pruned = today
        path = self.directory / f"{today.isoformat()}.jsonl"
        if path.exists() and path.stat().st_size > DAY_FILE_LIMIT:
            return
        with path.open("a", encoding="utf-8") as handle:
            _ = handle.write("".join(f"{line}\n" for line in lines))

    def _prune(self, today: date) -> None:
        cutoff = today - timedelta(days=RETAIN_DAYS)
        for path in self.directory.glob("*.jsonl"):
            try:
                day = date.fromisoformat(path.stem)
            except ValueError:
                continue
            if day < cutoff:
                path.unlink(missing_ok=True)


def register_client_diagnostics(app: FastAPI, workspace: Path) -> None:
    log = InteractionLog(workspace / "diagnostics" / "client")

    # 页面打开时取一次：记录里带的是页面代码的版本，部署后还开着的旧页面不会算到新版本头上。
    @app.get("/api/client/diagnostics")
    async def client_diagnostics_info() -> dict[str, str]:
        return {"release": await current_runtime_commit() or ""}

    @app.post("/api/client/diagnostics", status_code=204)
    async def client_diagnostics(batch: DiagnosticsBatch, request: Request) -> Response:
        # 1. 交互落盘，附服务端时间、客户端与服务端版本
        # 2. 前端报错按 error 写运行日志，夜间报错巡检照常按指纹聚合
        client = client_label(request.headers.get("user-agent", ""))
        release = await current_runtime_commit() or ""
        received = datetime.now(UTC).isoformat(timespec="milliseconds")
        lines: list[str] = []
        for record in batch.records:
            if isinstance(record, ErrorRecord):
                logger.error(
                    "web.client_error client=%s release=%s count=%d source=%s: %s\n%s",
                    client, record.page_release or release, record.count, record.source,
                    record.message, record.stack,
                )
                continue
            document: dict[str, object] = {"ts": received, "client": client, "server_release": release, **record.model_dump(exclude_none=True)}
            lines.append(json.dumps(document, ensure_ascii=False, separators=(",", ":")))
        if lines:
            await run_file_io(lambda: log.append(lines))
        return Response(status_code=204)
