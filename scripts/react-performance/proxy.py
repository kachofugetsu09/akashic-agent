"""同机边界时刻与原始流证据；不改模型请求及响应字节。"""

import argparse, asyncio, json, time, hashlib, os
from pathlib import Path
from aiohttp import web, ClientSession, ClientTimeout, TraceConfig, ClientError

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--endpoint", required=True, help="完整的上游 chat/completions URL")
parser.add_argument("--key-file", type=Path, required=True)
parser.add_argument("--directory", type=Path, required=True)
parser.add_argument("--port", type=int, default=2311)
parser.add_argument("--read-timeout", type=float, default=300)
args = parser.parse_args()
os.umask(0o077)
ROOT = args.directory.resolve()
ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
KEY = args.key_file.read_text().strip()
COUNTS = {}


def record(event, label, n, **fields):
    row = {
        "at": time.time(),
        "mono_ns": time.monotonic_ns(),
        "event": event,
        "label": label,
        "n": n,
        **fields,
    }
    with (ROOT / "wire.jsonl").open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def sent(session, ctx, params):
    label, n = ctx.trace_request_ctx
    record("upstream.sent", label, n, bytes=len(params.chunk))


async def forward(request):
    label = request.match_info["label"]
    folder = ROOT / label
    folder.mkdir(exist_ok=True)
    n = COUNTS.get(label, len(list(folder.glob("request-*.json")))) + 1
    COUNTS[label] = n
    raw = await request.read()
    body = json.loads(raw)
    record(
        "request.received",
        label,
        n,
        bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        messages=len(body["messages"]),
        reasoning_rows=sum(bool(m.get("reasoning_content")) for m in body["messages"]),
    )
    (folder / f"request-{n:03}.json").write_bytes(raw)
    downstream = web.StreamResponse()
    collected = bytearray()
    pending = b""
    first = True
    effective = False
    done = False
    try:
        async with request.app["client"].post(
            args.endpoint,
            data=raw,
            headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"},
            trace_request_ctx=(label, n),
        ) as upstream:
            record("upstream.headers", label, n, status=upstream.status)
            downstream.set_status(upstream.status)
            downstream.headers["Content-Type"] = upstream.headers.get("Content-Type", "application/json")
            await downstream.prepare(request)
            async for chunk in upstream.content.iter_any():
                if first:
                    record("provider.first_byte", label, n)
                    first = False
                collected.extend(chunk)
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        done = True
                        record("provider.done", label, n)
                        continue
                    try:
                        item = json.loads(data)
                    except json.JSONDecodeError:
                        record("provider.invalid_json", label, n)
                        continue
                    if item.get("usage"):
                        record("provider.usage", label, n, usage=item["usage"])
                    if not effective and any(
                        any(
                            c.get("delta", {}).get(k)
                            for k in ("content", "reasoning_content", "reasoning", "tool_calls")
                        )
                        for c in item.get("choices", [])
                    ):
                        effective = True
                        record("provider.first_delta", label, n)
                await downstream.write(chunk)
            await downstream.write_eof()
            record("response.closed", label, n, done=done)
    except asyncio.CancelledError:
        record("transport.closed" if done else "client.disconnected", label, n, done=done)
        raise
    except (ConnectionError, ClientError, asyncio.TimeoutError) as error:
        record(
            "transport.closed" if done else "transport.failed",
            label,
            n,
            error=type(error).__name__,
            done=done,
        )
        if not done:
            raise
    finally:
        (folder / f"response-{n:03}.sse").write_bytes(collected)
    return downstream


async def life(app):
    trace = TraceConfig()
    trace.on_request_chunk_sent.append(sent)
    async with ClientSession(
        timeout=ClientTimeout(total=600, connect=20, sock_read=args.read_timeout),
        trust_env=True,
        trace_configs=[trace],
    ) as client:
        app["client"] = client
        yield


app = web.Application(client_max_size=30 * 1024**2)
app.cleanup_ctx.append(life)
app.router.add_post(r"/{label:[a-zA-Z0-9_-]+}/v1/chat/completions", forward)
web.run_app(app, host="127.0.0.1", port=args.port, print=None, handler_cancellation=True)
