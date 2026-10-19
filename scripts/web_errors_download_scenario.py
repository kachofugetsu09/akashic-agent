"""Check real HTTP failures and Shell downloads in an isolated installed runtime."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from pydantic import HttpUrl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plugins.channels.contract import CHANNEL_INPUT_V2 as CHANNEL_INPUT, ChannelInboundMessage
from plugins.ledger.contract import Output, ToolResult
from plugins.shell.shell_security import validate_command
from plugins.standard_web.contract import FetchRequest, SearchRequest
from plugins.web_fetch_http.plugin import ScopedFetch
from plugins.web_search_exa.search import ExaConfig, ExaSearch
from tests.test_default_reply import application


async def run(directory: Path) -> dict[str, object]:
    """Use real sockets, HTTP providers, installed Shell and persisted tool results."""
    os.environ.update(HOME=str(directory / "home"), AKASHIC_PLUGIN_HOME=str(directory / "home"),
                      AKASHIC_PLUGIN_DISTRIBUTION="", AKASHIC_EXECUTION_MODE="local", AKASHIC_WORKLOAD_SOCKET="")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(name, None)
    payload = b"downloaded evidence\n"
    key = "scenario-private-key"

    async def read_key():
        return key

    # 1. A bound but non-listening socket gives an actual connection refusal.
    with socket.socket() as refused:
        refused.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{refused.getsockname()[1]}/"
        private_url = url.replace("http://", "http://scenario-user:scenario-password@") + "?token=" + key
        fetch = await ScopedFetch().fetch(FetchRequest(private_url, "text", 2))
        search = await ExaSearch(ExaConfig(endpoint=HttpUrl(url)), read_key).search(SearchRequest("evidence", 1))
    for result in (fetch, search):
        assert "ConnectError" in str(result) and "connect" in str(result).lower(), result
        assert key not in str(result), result
        assert "scenario-password" not in str(result) and "scenario-user" not in str(result), result

    async def plain_server(_reader, writer):
        writer.write(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    # A TLS request to an HTTP listener must retain the SSL reason, not just ConnectError.
    tls_server = await asyncio.start_server(plain_server, "127.0.0.1", 0)
    async with tls_server:
        tls_url = f"https://127.0.0.1:{tls_server.sockets[0].getsockname()[1]}/"
        tls_error = await ScopedFetch().fetch(FetchRequest(tls_url, "text", 2))
        assert "SSL" in str(tls_error) and "ConnectError" in str(tls_error), tls_error

    async def serve(reader, writer):
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            failed = b"/fail" in header.split(b"\r\n", 1)[0]
            status = "503 Service Unavailable" if failed else "200 OK"
            writer.write(f"HTTP/1.1 {status}\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode() + payload)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    async with server:
        port = server.sockets[0].getsockname()[1]
        # 2. A real HTTP status exception includes the request URL; the key must not escape.
        error = await ExaSearch(ExaConfig(endpoint=HttpUrl(f"http://127.0.0.1:{port}/fail")), read_key).search(SearchRequest("evidence", 1))
        assert "503" in str(error) and "HTTPStatusError" in str(error), error
        assert key not in str(error) and "exaApiKey" not in str(error), error

        # 3. Exercise curl output forms through installed Shell and verify the actual files.
        download = directory / "downloads"
        download.mkdir()
        commands = [
            f"curl --noproxy '*' --resolve example.com:{port}:127.0.0.1 http://example.com:{port}/remote.txt {flag}"
            for flag in ("-o short.txt", "--output=long.txt", "-O")
        ]

        def configure(sources: Path):
            provider = sources / "test_provider/plugin.py"
            text = provider.read_text().replace(
                'ToolCall("provider-call", "write_evidence", {})',
                'ToolCall("download", "shell", ' + repr({"command": "; ".join(commands),
                    "description": "download evidence", "cwd": str(download), "login": False}) + ')')
            provider.write_text(text)

        async with application(directory, replying=True, shell=True, extra_sources=configure) as (log, host):
            root = host.live_root
            assert root is not None
            await root.context.require(CHANNEL_INPUT)("test:download", "input", ChannelInboundMessage(
                "test", "user", "download", "download the evidence", datetime.now(UTC), {}))
            async with asyncio.timeout(10):
                async for _ in log.catalog().follow():
                    rows = log.reader("test:download").snapshot()
                    if any(isinstance(row.body, Output) and row.body.finish == "complete" for row in rows):
                        break
            results = [row.body for row in rows if isinstance(row.body, ToolResult)]
            assert len(results) == 1 and results[0].outcome == "success", results
            for name in ("short.txt", "long.txt", "remote.txt"):
                assert (download / name).read_bytes() == payload

        # 4. Existing upload and directory rules still reject their original operations.
        for flag in ("-T private.txt", "--upload-file=private.txt", "-F file=@private.txt"):
            assert validate_command("curl https://example.com " + flag, allow_network=True, restricted_dir=None)
        for flag in ("-o ../outside", "--output=../outside", "-o../outside"):
            assert validate_command("curl https://example.com " + flag, allow_network=True,
                                    restricted_dir=download, cwd=download)
    return {"fetch_error": fetch["error"], "search_error": search["error"], "status_error": error["error"],
            "tls_error": tls_error["error"], "download_files": 3,
            "upload_and_directory_rules": True, "keys_redacted": True}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="akashic-web-errors-") as directory:
        print(json.dumps(asyncio.run(run(Path(directory))), ensure_ascii=False))
