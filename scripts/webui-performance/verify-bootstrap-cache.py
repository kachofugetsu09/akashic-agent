"""通过真实聊天 ASGI 路由检查 bootstrap 的版本校验与缓存失效。"""

import asyncio
import json
from pathlib import Path
import tempfile

import httpx

from plugins.akashic_clients.chat_api import create_chat_app
from plugins.akashic_clients.web_chat import WebChatChannel


class CatalogFixture:
    """提供可切换 snapshot 的完整响应，复现发布与暂时不可用。"""

    snapshot = "first"
    available = True

    async def bootstrap(self) -> bytes:
        if not self.available:
            raise RuntimeError("catalog unavailable")
        return json.dumps({"snapshotId": self.snapshot, "modules": []}).encode()

    async def state(self) -> dict[str, str]:
        return {"snapshotId": self.snapshot, "catalogId": "fixture"}


async def verify(workspace: Path) -> None:
    """验证首次下载、相同字节、新 snapshot 和故障四个边界。"""
    catalog = CatalogFixture()
    app = create_chat_app(workspace=workspace, channel=WebChatChannel(), web_ui_provider=catalog)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
        # 1. 第一次收到完整响应，后续仍必须向服务器重新校验。
        first = await client.get("/api/chat/web-ui/bootstrap")
        assert first.status_code == 200
        assert first.headers["cache-control"] == "private, no-cache"
        etag = first.headers["etag"]
        same = await client.get("/api/chat/web-ui/bootstrap", headers={"If-None-Match": etag})
        assert same.status_code == 304 and same.content == b""
        assert same.headers["etag"] == etag
        compressed = await client.get("/api/chat/web-ui/bootstrap", headers={"If-None-Match": f'"older", W/{etag}'})
        assert compressed.status_code == 304 and compressed.content == b""

        # 2. snapshot 变化不能复用旧字节；不可用不能伪装成缓存命中。
        catalog.snapshot = "second"
        changed = await client.get("/api/chat/web-ui/bootstrap", headers={"If-None-Match": etag})
        assert changed.status_code == 200
        assert changed.json()["snapshotId"] == "second"
        assert changed.headers["etag"] != etag
        catalog.available = False
        unavailable = await client.get("/api/chat/web-ui/bootstrap", headers={"If-None-Match": changed.headers["etag"]})
        assert unavailable.status_code == 503
    print("bootstrap 首次下载、304、snapshot 失效与 503 场景通过")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="akashic-bootstrap-cache-") as directory:
        asyncio.run(verify(Path(directory)))
