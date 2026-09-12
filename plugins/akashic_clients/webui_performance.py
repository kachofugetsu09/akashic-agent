from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from infra.channels.artifacts import ChannelAttachmentArtifactStore
from plugins.akashic_clients.chat_api import create_chat_app
from plugins.akashic_clients.web_chat import WebChatChannel
from session.artifact_store import ArtifactStore
from session.log import MessageLog


async def serve(port: int, workspace: Path) -> None:
    """启动只含 Web owner 的隔离持久化/API 栈。"""

    workspace.mkdir(parents=True, exist_ok=True)
    messages = MessageLog(workspace / "sessions.db")
    metadata = ArtifactStore(workspace / "artifacts.db")
    artifacts = ChannelAttachmentArtifactStore(
        workspace=workspace,
        metadata_store=metadata,
    )
    channel = WebChatChannel()
    await channel.start()
    app = create_chat_app(
        workspace=workspace,
        channel=channel,
        messages=cast(Any, messages.catalog()),
        artifact_store=cast(Any, artifacts),
    )

    @app.get("/api/shell/state")
    def shell_state() -> dict[str, object]:
        return {"status": "ready", "configured": True, "chatReady": True}

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
        )
    )
    try:
        await server.serve()
    finally:
        await channel.stop()
        metadata.close()
        messages.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=4174)
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    temporary = args.workspace is None
    workspace = args.workspace or Path(tempfile.mkdtemp(prefix="akashic-webui-runtime-"))
    try:
        print(
            f'{{"event":"webui.runtime_fixture_starting","workspace":"{workspace}","port":{args.port}}}',
            flush=True,
        )
        try:
            asyncio.run(serve(args.port, workspace))
        except KeyboardInterrupt:
            pass
    finally:
        if temporary:
            shutil.rmtree(workspace)


if __name__ == "__main__":
    main()
