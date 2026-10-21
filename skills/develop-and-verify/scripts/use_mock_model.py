"""给开发实例接上 mock_model.py：新建连接和模型并设为默认。只依赖标准库。

    python3 use_mock_model.py --state <start.py 的 --state> --url http://127.0.0.1:<端口>

用数据目录里的本机门禁密钥鉴权；dashboard 路由还要带 bootstrap 给出的模块身份请求头，否则一律 404。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast
from urllib.request import ProxyHandler, Request, build_opener

_OPENER = build_opener(ProxyHandler({}))


def _call(url: str, headers: dict[str, str], body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode()
    request = Request(url, data=data, method="GET" if body is None else "POST",
                      headers={"content-type": "application/json", **headers})
    with _OPENER.open(request, timeout=30) as response:
        return cast(dict[str, Any], json.load(response))


# 1. bootstrap 取 models 模块身份；2. 读修订号；3. 加连接、加模型、设默认，每步带上一步的修订号。
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--mock", default="http://127.0.0.1:44099")
    args = parser.parse_args()
    secret = (args.state.expanduser() / "workspace/web-auth/secret").read_text(encoding="utf-8").strip()
    base, auth = args.url.rstrip("/"), {"Authorization": f"Bearer {secret}"}
    boot = _call(f"{base}/api/chat/web-ui/bootstrap", auth)
    module = next(item for item in boot["modules"] if str(item["pluginId"]).startswith("models@"))
    headers = {**auth, "x-akashic-web-snapshot": str(boot["snapshotId"]), "x-akashic-web-catalog": str(boot["catalogId"]),
               "x-akashic-web-module": str(module["pluginId"]), "x-akashic-web-generation": str(module["generationId"])}
    api = f"{base}/api/dashboard/models"
    revision = int(_call(f"{api}/catalog", headers)["revision"])
    changes: list[dict[str, Any]] = [
        {"type": "add_connection", "connection_id": "mock", "name": "Mock", "driver_id": "openai-compatible",
         "endpoint": args.mock, "auth_identity": "mock", "credential": {"api_key": "mock"},
         "driver_config": {"allow_unverified_manual": True}},
        *({"type": "add_model", "model_id": name, "connection_id": "mock", "kind": "chat", "model": name,
           "capabilities": {"context_window": 32000, "max_output_tokens": 4096, "supports_tool_calls": True},
           "capability_sources": {}} for name in ("mock-main", "mock-mini")),
        {"type": "set_default", "role": "default", "model_id": "mock-main"},
    ]
    for change in changes:
        revision = int(_call(f"{api}/command", headers, {**change, "expected_revision": revision})["revision"])
    print(f"已接上模拟模型 mock-main / mock-mini，默认 mock-main（修订 {revision}）")


if __name__ == "__main__":
    main()
