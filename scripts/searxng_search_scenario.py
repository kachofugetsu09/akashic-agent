"""Exercise installed search providers against real HTTP responses or a live instance."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.plugin_composition.config_input import save_config
from plugins.standard_web.contract import WEB_PROVIDERS, SearchRequest
from plugins.web_search_searxng.plugin import SETTINGS
from tests.test_default_reply import application


async def run(directory: Path, endpoint: str | None) -> None:
    """Load real plugins, check selection, then send searches through their registry."""
    os.environ.update(HOME=str(directory / "home"), AKASHIC_PLUGIN_HOME=str(directory / "home"),
                      AKASHIC_PLUGIN_DISTRIBUTION="", AKASHIC_EXECUTION_MODE="local", AKASHIC_WORKLOAD_SOCKET="")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(name, None)

    async def serve(reader, writer):
        """Serve SearXNG-shaped replies and explicit upstream failures over a socket."""
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            method, target, _ = header.split(b"\r\n", 1)[0].decode().split(" ")
            url = urlsplit(target)
            args = parse_qs(url.query)
            assert method == "GET" and url.path == "/instance/search" and args["format"] == ["json"]
            query = args["q"][0]
            results = [{"title": "First", "url": "https://example.org/1", "content": "Summary"},
                       {"title": "Second", "url": "https://example.org/2"}]
            data = {"results": [] if query in {"empty", "failed"} else results,
                    "unresponsive_engines": [["bing", "timeout"]] if query in {"partial", "failed"} else []}
            payload = b"<html>JSON disabled</html>" if query == "html" else json.dumps(data).encode()
            if query == "missing":
                payload = b'{}'
            status = "403 Forbidden" if query == "forbidden" else "200 OK"
            writer.write(f"HTTP/1.1 {status}\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode() + payload)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    async with server:
        address = endpoint or f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/instance/"
        # 1. Without an address, Exa remains the sole registered search provider.
        for configured in (False, True):
            case = directory / str(configured)

            def sources(path: Path):
                for name in ("standard_web", "web_search_exa", "web_search_searxng"):
                    shutil.copytree(ROOT / "plugins" / name, path / name,
                                    ignore=shutil.ignore_patterns("__pycache__"))
                if configured:
                    save_config(case / "workspace/plugin-data/web_search_searxng-builtin", {"endpoint": address})
                    save_config(case / "workspace/plugin-data/standard_web-builtin", {"search_provider": "web_search_searxng"})

            async with application(case, replying=False, extra_sources=sources) as (_, host):
                root = host.live_root
                assert root is not None
                providers = root.context.require(WEB_PROVIDERS)
                settings = await root.context.require(SETTINGS).read()
                assert settings["ready"] is configured, settings
                choices = providers.choices("search")
                assert {row["value"] for row in choices} == ({"web_search_exa", "web_search_searxng"} if configured else {"web_search_exa"}), choices
                if not configured:
                    continue
                # 2. Use the configured provider, with both providers registered.
                result = await providers.search(SearchRequest("python documentation" if endpoint else "中文 & Python", 1))
                assert "error" not in result and len(result["results"]) == 1, result
                print(json.dumps({"configured_search": result}, ensure_ascii=False))
                if endpoint:
                    continue
                # 3. Empty results, partial failures, and bad replies stay distinguishable.
                empty = await providers.search(SearchRequest("empty", 2))
                assert empty["results"] == [] and "error" not in empty, empty
                partial = await providers.search(SearchRequest("partial", 1))
                assert len(partial["results"]) == 1 and partial["unresponsive_engines"] and "error" not in partial, partial
                for query in ("failed", "html", "missing", "forbidden"):
                    failed = await providers.search(SearchRequest(query, 2))
                    assert failed["error"], failed
                # 4. Saving through the settings owner withdraws the provider after activation.
                owner = root.context.require(SETTINGS).ctx.runtime.plugin_id
                await root.context.require(SETTINGS).save("clear-searxng", settings["input_ref"], {"endpoint": ""})
                assert host._operation is not None
                await host._operation.task
                assert host.read_config_update(owner, "clear-searxng")["state"] == "active"
                updated = host.live_root.context.require(WEB_PROVIDERS)
                assert {row["value"] for row in updated.choices("search")} == {"web_search_exa"}
                assert "error" in await updated.search(SearchRequest("no fallback", 1))
                assert (await host.live_root.context.require(SETTINGS).read())["values"] == {}
                print("selection, subpath, query encoding, count, empty, partial and error scenarios passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", help="Optional live SearXNG root address")
    options = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="searxng-scenario-") as temporary:
        asyncio.run(run(Path(temporary), options.endpoint))
