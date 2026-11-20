from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, suppress
import json
import os
from pathlib import Path
import shutil
from time import monotonic
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.plugin_composition import Context
from plugins.ledger.contract import ContentPart, json_value
from plugins.tools.contract import (
    TOOLS,
    BoundTool,
    CallSource,
    NestedToolScope,
    Result,
    current_nested_tool_scope,
)

api_version = 3
name = "codemode"
version = "1.0.0"
desc = "在隔离 JS 沙箱中一次编写并执行组合多个已获授工具的脚本（pi Code Mode）"
inject = (TOOLS,)

_RUNNER_PATH = Path(__file__).resolve().with_name("runner.mjs")
_MAX_OUTPUT_CHARS = 40_000
_SUBPROCESS_LINE_LIMIT = 4 * 1024 * 1024


class CodemodeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    code: str = Field(
        min_length=1,
        max_length=65_536,
        description=(
            "在隔离 JS 沙箱中执行的 async 函数体代码。"
            "支持 `await tools.<tool_name>(args)`（返回 `{ ok, outcome, text, artifacts }`）、"
            "`tools.list()`、`tools.describe(name)`、`sleep(ms)`、`console.log(...)`、`collectArtifact(id)` 与 `return`。"
        ),
    )
    timeout_ms: int = Field(
        default=60_000,
        ge=1_000,
        le=300_000,
        description="整段脚本的硬超时时间（毫秒），默认 60000，上限 300000。",
    )
    max_calls: int = Field(
        default=40,
        ge=1,
        le=100,
        description="本次脚本最多允许发起的嵌套工具调用次数，默认 40。",
    )


def _summarize_nested_result(result: Result) -> dict[str, object]:
    text_parts: list[str] = []
    artifacts: list[str] = []
    for part in result.parts:
        if part.kind == "text":
            text_parts.append(str(part.value))
        elif part.kind == "artifact_ref":
            artifacts.append(str(part.value))
    return {
        "ok": result.outcome == "success",
        "outcome": result.outcome,
        "text": "\n".join(text_parts),
        "artifacts": artifacts,
    }


def _truncate_text(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head - 80
    return f"{text[:head]}\n\n... [已截断 {len(text) - head - tail} 字符] ...\n\n{text[-tail:]}"


async def _run_js_sandbox(
    scope: NestedToolScope,
    args: CodemodeInput,
) -> Result:
    search_path = os.environ.get("PATH") or "/usr/local/bin:/usr/bin:/bin"
    node_bin = shutil.which("node", path=search_path)
    if not node_bin:
        return Result("error", (ContentPart("text", "codemode 需要运行环境安装 Node.js (`node`)"),))

    minimal_env = {
        "PATH": search_path,
        "LANG": os.environ.get("LANG") or "C.UTF-8",
    }
    proc = await asyncio.create_subprocess_exec(
        node_bin,
        "--no-warnings",
        "--disallow-code-generation-from-strings",
        str(_RUNNER_PATH),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=_SUBPROCESS_LINE_LIMIT,
        env=minimal_env,
    )
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None

    tools_meta: list[dict[str, object]] = [
        {
            "name": ref.name,
            "owner": ref.owner,
            "description": str(ref.description.get("description") or ""),
            "parameters": json_value(ref.description.get("parameters") or {}),
        }
        for ref in sorted(scope.tools.values(), key=lambda item: item.name)
    ]
    init_payload = json.dumps(
        {
            "code": args.code,
            "toolsMeta": tools_meta,
            "maxCalls": args.max_calls,
        },
        ensure_ascii=False,
    ) + "\n"
    proc.stdin.write(init_payload.encode("utf-8"))
    await proc.stdin.drain()

    started = monotonic()
    allowed_artifacts: set[str] = set()
    done_frame: dict[str, object] | None = None
    host_calls = 0
    write_lock = asyncio.Lock()
    in_flight: set[asyncio.Task[None]] = set()

    async def write_reply(reply_frame: dict[str, object]) -> None:
        async with write_lock:
            if proc.stdin is not None and not proc.stdin.is_closing():
                proc.stdin.write((json.dumps(reply_frame, ensure_ascii=False) + "\n").encode("utf-8"))
                await proc.stdin.drain()

    async def dispatch_call(
        call_id: object,
        tool_name: str,
        tool_args: Mapping[str, object],
    ) -> None:
        try:
            res = await scope.call(tool_name, tool_args)
            summary = _summarize_nested_result(res)
            for aid in cast(list[str], summary["artifacts"]):
                allowed_artifacts.add(aid)
            reply_frame: dict[str, object] = {"id": call_id, "ok": True, "result": summary}
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reply_frame = {"id": call_id, "ok": False, "error": str(exc)}
        with suppress(BrokenPipeError, ConnectionResetError):
            await write_reply(reply_frame)

    async def cancel_in_flight() -> None:
        for task in list(in_flight):
            _ = task.cancel()
        if in_flight:
            _ = await asyncio.gather(*in_flight, return_exceptions=True)
            in_flight.clear()

    try:
        async with asyncio.timeout(args.timeout_ms / 1000.0):
            while True:
                raw_line = await proc.stdout.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    frame = cast(dict[str, object], json.loads(line))
                except json.JSONDecodeError:
                    continue
                op = frame.get("op")
                if op == "call":
                    call_id = frame.get("id")
                    if host_calls >= args.max_calls:
                        with suppress(BrokenPipeError, ConnectionResetError):
                            await write_reply({
                                "id": call_id,
                                "ok": False,
                                "error": f"Exceeded max_calls limit ({args.max_calls})",
                            })
                        continue
                    host_calls += 1
                    tool_name = str(frame.get("name") or "")
                    raw_args = frame.get("arguments")
                    tool_args: Mapping[str, object] = (
                        cast(Mapping[str, object], raw_args)
                        if isinstance(raw_args, Mapping)
                        else {}
                    )
                    task = asyncio.create_task(dispatch_call(call_id, tool_name, tool_args))
                    in_flight.add(task)
                    task.add_done_callback(in_flight.discard)
                elif op == "done":
                    done_frame = frame
                    break
            await cancel_in_flight()
    except TimeoutError:
        await cancel_in_flight()
        with suppress(ProcessLookupError):
            proc.kill()
        _ = await proc.wait()
        return Result("error", (ContentPart("text", f"codemode 脚本执行超时 ({args.timeout_ms}ms)"),))
    except asyncio.CancelledError:
        await cancel_in_flight()
        with suppress(ProcessLookupError):
            proc.kill()
        _ = await proc.wait()
        raise
    finally:
        await cancel_in_flight()
        if proc.returncode is None:
            with suppress(ProcessLookupError):
                proc.terminate()
            try:
                _ = await asyncio.wait_for(proc.wait(), timeout=2.0)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    proc.kill()
                _ = await proc.wait()

    stderr_bytes = await proc.stderr.read()
    stderr_text = stderr_bytes.decode("utf-8", "replace").strip()
    elapsed_ms = int((monotonic() - started) * 1000)

    if done_frame is None:
        detail = stderr_text or f"沙箱意外退出 (exit={proc.returncode})"
        return Result("error", (ContentPart("text", f"codemode 执行失败 ({elapsed_ms}ms): {detail}"),))

    ok = bool(done_frame.get("ok"))
    logs = cast(list[str], done_frame.get("logs") or [])
    ret_val = done_frame.get("result")
    error_msg = done_frame.get("error")
    collected = [
        str(aid)
        for aid in cast(list[object], done_frame.get("collectedArtifacts") or [])
        if isinstance(aid, str) and aid in allowed_artifacts
    ]

    sections = [f"codemode {'ok' if ok else 'error'} | calls={host_calls} | elapsed={elapsed_ms}ms"]
    if logs:
        sections.append("Logs:\n" + "\n".join(str(x) for x in logs))
    if ret_val is not None:
        sections.append("Return:\n" + json.dumps(ret_val, ensure_ascii=False, indent=2))
    if not ok and error_msg:
        sections.append(f"Error:\n{error_msg}")
    if stderr_text and not ok:
        sections.append(f"Stderr:\n{stderr_text}")

    parts: list[ContentPart] = [ContentPart("text", _truncate_text("\n\n".join(sections)))]
    for aid in collected:
        parts.append(ContentPart("artifact_ref", aid))
    return Result("success" if ok else "error", tuple(parts))


class CodemodeTool:
    idempotent = False

    async def prepare(
        self, arguments: Mapping[str, object], source: CallSource | None = None
    ) -> Mapping[str, object] | str:
        _ = source
        try:
            return CodemodeInput.model_validate(json_value(arguments)).model_dump(mode="json")
        except ValidationError as error:
            return str(error)

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        _ = key
        scope = current_nested_tool_scope()
        if scope is None:
            return Result("error", (ContentPart("text", "codemode 只能由当前会话工具菜单内调用"),))
        parsed = CodemodeInput.model_validate(json_value(arguments))
        return await _run_js_sandbox(scope, parsed)

    async def query(self, key: str) -> Result | None:
        _ = key
        return None


async def apply(ctx: Context) -> None:
    catalog = ctx.require(TOOLS)
    _ = await catalog.declare_group(ctx, always_on=True, description=desc)

    @asynccontextmanager
    async def open_tool(_state: Mapping[str, object]) -> AsyncIterator[BoundTool]:
        yield CodemodeTool()

    _ = await catalog.register(
        ctx,
        name="codemode",
        description=(
            "在隔离 JS 沙箱中一次编写并执行组合多个已获授工具的脚本，减少多轮模型往返。"
            "脚本内支持 `await tools.<tool_name>(args)`（返回 `{ ok, outcome, text, artifacts }`）、"
            "`tools.list()`、`tools.describe(name)`、`sleep(ms)`、`console.log(...)`、`collectArtifact(id)` 与 `return`。"
            "适合多步手机操控（解析 UI 树 `snapshot_ref`/`node_id` 连续操作）、批量查询或数据整理。"
        ),
        parameters=CodemodeInput.model_json_schema(),
        open=open_tool,
        idempotent=False,
        parallel=False,
        icon="code-2",
    )
