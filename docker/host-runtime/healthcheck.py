from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path


def main() -> None:
    """Verify the in-band runtime identity and the real Web Chat health route."""

    # 1. Require the current boot to publish the requested immutable identity.
    workspace = Path(os.environ["AKASHIC_WORKSPACE"])
    document = json.loads(
        (workspace / ".runtime-ready.json").read_text(encoding="utf-8")
    )
    if document != {
        "bootId": document.get("bootId"),
        "pid": document.get("pid"),
        "state": "ready",
        "sourceCommit": os.environ["AKASHIC_RUNTIME_COMMIT"],
        "hostCheckout": os.environ["AKASHIC_RUNTIME_CHECKOUT"],
    }:
        raise RuntimeError("runtime readiness identity 不一致")
    if not isinstance(document["bootId"], str) or not document["bootId"]:
        raise RuntimeError("runtime readiness bootId 无效")
    pid = document["pid"]
    if not isinstance(pid, int) or pid <= 0:
        raise RuntimeError("runtime readiness pid 无效")
    os.kill(pid, 0)

    # 2. Exercise the public Web Shell proxy rather than only checking a PID.
    port = int(os.environ.get("AKASHIC_WEB_PORT", "2236"))
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/api/chat/health",
        timeout=3,
    ) as response:
        if response.status != 200:
            raise RuntimeError(f"Web Chat health 状态异常: {response.status}")
        payload = json.load(response)
    if payload.get("status") != "ready":
        raise RuntimeError("Web Chat health payload 异常")


if __name__ == "__main__":
    main()
