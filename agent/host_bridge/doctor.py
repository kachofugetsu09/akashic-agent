from __future__ import annotations

import argparse
import asyncio
import uuid
from pathlib import Path

from agent.host_bridge.client import HostBridgeShellProcessManager


async def _probe(
    socket_path: Path,
    token: str,
    expected_release_commit: str,
    expected_toolchain_digest: str,
) -> None:
    manager = HostBridgeShellProcessManager(
        socket_path,
        f"preflight-{uuid.uuid4().hex}",
        token,
        expected_release_commit,
        expected_toolchain_digest,
    )
    try:
        response = await manager.inspect()
        required = {
            "boot-fencing",
            "exec",
            "pty",
            "stdin",
            "stop",
            "lease",
            "file-tools",
            "raw-bytes",
            "skill-requirements",
        }
        capabilities = set(response["capabilities"])
        missing = required - capabilities
        if missing:
            raise RuntimeError(f"Host Bridge 缺少能力: {sorted(missing)}")
    finally:
        await manager.close_transport()


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the Akashic Host Bridge")
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--expected-release-commit", required=True)
    parser.add_argument("--expected-toolchain-digest", required=True)
    args = parser.parse_args()
    asyncio.run(
        _probe(
            args.socket,
            args.token,
            args.expected_release_commit,
            args.expected_toolchain_digest,
        )
    )


if __name__ == "__main__":
    main()
