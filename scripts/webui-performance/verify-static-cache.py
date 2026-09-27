"""通过真实 ASGI 路由验证聊天静态资源的缓存合同。"""

import asyncio
from pathlib import Path
import runpy
import shutil
import tempfile

import httpx


async def verify(root: Path) -> None:
    """在一次性目录中加载当前服务端源码，检查入口、资源与重定向。"""
    # 1. 只替换资源所在目录，路由和中间件使用当前仓库的完整实现。
    source = Path(__file__).resolve().parents[2] / "bootstrap/settings_api.py"
    module = root / "bootstrap/settings_api.py"
    module.parent.mkdir()
    shutil.copyfile(source, module)
    assets = root / "static/chat"
    assets.mkdir(parents=True)
    (assets / "index.html").write_text("<html>chat</html>", encoding="utf-8")
    (assets / "index-12345678.js").write_text("export {};", encoding="utf-8")
    (assets / "theme.json").write_text("{}", encoding="utf-8")
    app = runpy.run_path(str(module))["create_settings_app"]()

    # 2. 实际经过 FileResponse、StaticFiles 和响应头中间件。
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://cache-scenario.local"
    ) as client:
        for path, cache in (
            ("/chat", "no-cache"),
            ("/chat/", "no-cache"),
            ("/assets/index.html", "no-cache"),
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
    print("真实 ASGI 静态缓存场景通过")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="akashic-static-cache-") as directory:
        asyncio.run(verify(Path(directory)))
