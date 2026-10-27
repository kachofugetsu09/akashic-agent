"""TTFT 测量用的本机模拟模型：记录请求到达时刻与请求体中最新的测量标记，立即结束回复。

用法：python fixture_model.py <port> <log>
- /v1/chat/completions：流式或非流式，回复固定文本 "模拟回复：TTFT"；
- /v1/embeddings：按文本哈希返回确定性单位向量，维度取请求的 dimensions（默认 1024）。
日志每行一个 JSON：{"arrived": epoch 秒, "marker": "TTFT-<毫秒>"|null, "messages": n, "bytes": n}。
"""

import hashlib
import json
import re
import struct
import sys
import time

from aiohttp import web

_MARKER = re.compile(r"TTFT-(\d+)")


def _log(path: str, record: dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as log:
        log.write(json.dumps(record, ensure_ascii=False) + "\n")


async def models(_request: web.Request) -> web.Response:
    return web.json_response({"data": [{"id": "fixture", "object": "model"}]})


async def chat(request: web.Request) -> web.StreamResponse:
    arrived = time.time()
    raw = await request.read()
    body = json.loads(raw)
    # 召回材料可能带着更早轮次的标记，取数值最大的才是本轮输入。
    markers = [int(value) for value in _MARKER.findall(raw.decode("utf-8", "replace"))]
    _log(request.app["log"], {
        "arrived": arrived, "marker": f"TTFT-{max(markers)}" if markers else None,
        "messages": len(body.get("messages", ())), "bytes": len(raw),
    })
    usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    message = {"role": "assistant", "content": "模拟回复：TTFT"}
    if not body.get("stream"):
        return web.json_response({
            "id": "fixture", "object": "chat.completion", "usage": usage,
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        })
    response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)
    chunk = {"choices": [{"index": 0, "delta": {"content": message["content"]}, "finish_reason": "stop"}], "usage": usage}
    await response.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode())
    await response.write_eof()
    return response


async def embeddings(request: web.Request) -> web.Response:
    body = await request.json()
    texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
    dimension = int(body.get("dimensions") or 1024)
    data = []
    for index, text in enumerate(texts):
        seed = hashlib.sha256(text.encode()).digest()
        raw = (seed * (dimension * 4 // len(seed) + 1))[: dimension * 4]
        vector = [value / 2**31 for value in struct.unpack(f"<{dimension}i", raw)]
        norm = sum(value * value for value in vector) ** 0.5
        data.append({"object": "embedding", "index": index, "embedding": [value / norm for value in vector]})
    return web.json_response({"object": "list", "data": data, "model": body.get("model"),
                              "usage": {"prompt_tokens": 1, "total_tokens": 1}})


def main() -> None:
    app = web.Application(client_max_size=512 * 1024 * 1024)
    app["log"] = sys.argv[2]
    # 连接地址带不带 /v1 都能用。
    app.add_routes([
        route
        for prefix in ("/v1", "")
        for route in (web.get(f"{prefix}/models", models), web.post(f"{prefix}/chat/completions", chat),
                      web.post(f"{prefix}/embeddings", embeddings))
    ])
    web.run_app(app, host="127.0.0.1", port=int(sys.argv[1]))


if __name__ == "__main__":
    main()
