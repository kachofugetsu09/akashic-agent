from __future__ import annotations

import argparse
import asyncio
import uuid
from pathlib import Path

from agent.host_bridge.client import HostBridgeShellProcessManager


async def _probe(socket_path: Path, token: str) -> None:
    manager = HostBridgeShellProcessManager(
        socket_path,
        f"preflight-{uuid.uuid4().hex}",
        token,
    )
    probe_succeeded = False
    try:
        response = await manager.probe()
        required = {
            "exec", "pty", "stdin", "stop", "lease", "file-tools", "raw-bytes"
        }
        capabilities = set(response["capabilities"])
        missing = required - capabilities
        if missing:
            raise RuntimeError(f"Host Bridge 缺少能力: {sorted(missing)}")
        probe_succeeded = True
    finally:
        if probe_succeeded:
            _ = await manager.shutdown()
        else:
            await manager.close_transport()


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the Akashic Host Bridge")
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    asyncio.run(_probe(args.socket, args.token))


if __name__ == "__main__":
    main()
