"""通过真实 ASGI 路由验证聊天静态资源的缓存合同。"""

import asyncio
from pathlib import Path
import runpy
import shutil
import tempfile
import sys

import httpx
from fastapi import FastAPI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


async def verify(root: Path) -> None:
    """在一次性目录中加载当前服务端源码，检查实际客户端入口、资源和缓存。"""
    # 1. 只替换资源所在目录，路由和中间件使用当前仓库的完整实现。
    source = Path(__file__).resolve().parents[2] / "plugins/akashic_clients/static.py"
    module = root / "plugins/akashic_clients/static.py"
    module.parent.mkdir(parents=True)
    shutil.copyfile(source, module)
    assets = root / "static/chat"
    assets.mkdir(parents=True)
    (assets / "chat-app.js").write_text("export {};", encoding="utf-8")
    (assets / "chat-app-mount.js").write_text("export {};", encoding="utf-8")
    (assets / "index-12345678.js").write_text("export {};", encoding="utf-8")
    (assets / "theme.json").write_text("{}", encoding="utf-8")
    large_body = "/* compression scenario */" + "x" * 10000
    (assets / "large-12345678.js").write_text(large_body, encoding="utf-8")
    app = FastAPI()
    runpy.run_path(str(module))["register_chat_assets"](app, assets)

    # 2. 实际经过 FileResponse、StaticFiles 和响应头中间件。
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://cache-scenario.local"
    ) as client:
        # 聊天并入外壳后 /chat 只把旧链接送回外壳，重定向本身不缓存。
        for path in ("/chat", "/chat/"):
            response = await client.get(path + "?session=s")
            assert response.status_code == 308 and response.headers["location"] == "/?session=s", path
            assert response.headers["cache-control"] == "no-store", path
        for path, cache in (
            ("/assets/chat-app.js", "no-cache"),
            ("/assets/chat-app-mount.js", "no-cache"),
            ("/assets/index-12345678.js", "public, max-age=31536000, immutable"),
            ("/assets/theme.json", "no-store"),
        ):
            response = await client.get(path)
            assert response.status_code == 200, (path, response.status_code)
            assert response.headers["cache-control"] == cache, path
            assert response.headers.get("pragma") == ("no-cache" if cache == "no-store" else None), path

        response = await client.get("/settings")
        assert response.status_code == 308
        assert response.headers["location"] == "/#models"
        assert response.headers["cache-control"] == "no-store"

        # 3. 编码变体与条件请求必须保持一致，gzip 的校验器不能续传原文字节。
        path = "/assets/large-12345678.js"
        plain = await client.get(path, headers={"Accept-Encoding": "identity"})
        zipped = await client.get(path, headers={"Accept-Encoding": "gzip"})
        assert plain.text == zipped.text == large_body
        assert "content-encoding" not in plain.headers
        assert zipped.headers["content-encoding"] == "gzip"
        assert plain.headers["etag"] != zipped.headers["etag"]
        for response in (plain, zipped):
            assert response.headers["vary"] == "Accept-Encoding"
        for encoding, original in (("identity", plain), ("gzip", zipped)):
            same = await client.get(path, headers={"Accept-Encoding": encoding,
                                                  "If-None-Match": original.headers["etag"]})
            assert same.status_code == 304 and not same.content
            assert same.headers["vary"] == "Accept-Encoding"
        changed_encoding = await client.get(path, headers={"Accept-Encoding": "gzip",
                                                          "If-None-Match": plain.headers["etag"]})
        assert changed_encoding.status_code == 200 and changed_encoding.text == large_body
        for quality in ("0", "bad", "", "NaN", "2"):
            refused = await client.get(path, headers={"Accept-Encoding": f"gzip;q={quality}"})
            assert refused.status_code == 200 and "content-encoding" not in refused.headers
        head = await client.head(path, headers={"Accept-Encoding": "gzip"})
        assert head.status_code == 200 and not head.content
        assert head.headers["etag"] == zipped.headers["etag"]
        assert head.headers["content-length"] == zipped.headers["content-length"]
        for validator, expected in ((plain.headers["etag"], 206), (zipped.headers["etag"], 200)):
            resumed = await client.get(path, headers={"Accept-Encoding": "gzip", "Range": "bytes=0-15",
                                                     "If-Range": validator})
            assert resumed.status_code == expected
            assert "content-encoding" not in resumed.headers
            assert resumed.text == (large_body[:16] if expected == 206 else large_body)
    print("真实 ASGI 静态缓存场景通过")


async def verify_dashboard(root: Path) -> None:
    """外壳(dashboard)的静态资产与聊天同一缓存合同:哈希产物永久缓存,稳定 URL 按 ETag 校验。"""
    source = Path(__file__).resolve().parents[2] / "plugins/ui/dashboard_app.py"
    module = root / "plugins/ui/dashboard_app.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, module)
    static_dir = root / "static/dashboard"
    (static_dir / "sdk").mkdir(parents=True)
    (static_dir / "index.html").write_text("<html></html>", encoding="utf-8")
    (static_dir / "index-12345678.js").write_text("export {};", encoding="utf-8")
    (static_dir / "sdk/react.js").write_text("export {};", encoding="utf-8")
    app = runpy.run_path(str(module))["create_dashboard_app"](static_dir)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://cache-scenario.local"
    ) as client:
        # 带内容哈希的产物永久缓存;稳定 URL(sdk、入口 HTML)不设显式头部,
        # 保持既有启发式缓存行为(强制校验会让模块图每个稳定分块等一次 304)。
        for path, cache in (
            ("/dashboard/assets/index-12345678.js", "public, max-age=31536000, immutable"),
            ("/assets/index-12345678.js", "public, max-age=31536000, immutable"),
        ):
            response = await client.get(path)
            assert response.status_code == 200, (path, response.status_code)
            assert response.headers["cache-control"] == cache, (path, response.headers.get("cache-control"))
        for path in ("/", "/dashboard", "/dashboard/assets/sdk/react.js"):
            response = await client.get(path)
            assert response.status_code == 200, (path, response.status_code)
            assert "cache-control" not in response.headers, (path, response.headers.get("cache-control"))
    print("真实 ASGI 外壳静态缓存场景通过")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="akashic-static-cache-") as directory:
        asyncio.run(verify(Path(directory)))
    with tempfile.TemporaryDirectory(prefix="akashic-dashboard-cache-") as directory:
        asyncio.run(verify_dashboard(Path(directory)))
