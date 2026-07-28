from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agent.plugins import McpServerSpec, Plugin, ProactiveSourceSpec
from core.clock import clock_from_env
from infra.channels.contract import ChannelContext


def _replay_source_enabled() -> bool:
    return bool(
        os.environ.get("AKASHIC_REPLAY_CLOCK_FILE", "").strip()
        and os.environ.get("AKASHIC_REPLAY_EVENTS_FILE", "").strip()
    )


class CaptureChannel:
    name = "replay"

    def __init__(self, outbox_path: Path) -> None:
        self.name = os.environ.get("AKASHIC_REPLAY_CHANNEL", "replay").strip() or "replay"
        self._outbox_path = outbox_path
        self._registration: Any = None

    async def start(self, ctx: ChannelContext) -> None:
        self._outbox_path.parent.mkdir(parents=True, exist_ok=True)
        self._registration = ctx.push_tool.register_channel(
            self.name,
            text=self._send_text,
            file=self._send_file,
            image=self._send_image,
        )

    async def stop(self) -> None:
        if self._registration is not None:
            self._registration.close()
            self._registration = None

    async def _send_text(self, chat_id: str, message: str) -> None:
        self._append({"type": "text", "chat_id": chat_id, "message": message})

    async def _send_file(
        self,
        chat_id: str,
        file_path: str,
        name: str | None = None,
    ) -> None:
        self._append(
            {
                "type": "file",
                "chat_id": chat_id,
                "path": file_path,
                "name": name,
            }
        )

    async def _send_image(self, chat_id: str, image_path: str) -> None:
        self._append({"type": "image", "chat_id": chat_id, "path": image_path})

    def _append(self, payload: dict[str, Any]) -> None:
        record: dict[str, Any] = {
            "captured_at": clock_from_env().now().isoformat(),
            **payload,
        }
        with self._outbox_path.open("a", encoding="utf-8") as handle:
            _ = handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class ReplayDebugPlugin(Plugin):
    api_version = 2
    name = "replay_debug"

    def channels(self) -> list[CaptureChannel]:
        path = os.environ.get("AKASHIC_REPLAY_OUTBOX_FILE", "").strip()
        return [CaptureChannel(Path(path))] if path and _replay_source_enabled() else []

    @classmethod
    def mcp_servers(cls) -> list[McpServerSpec]:
        if not _replay_source_enabled():
            return []
        return [
            McpServerSpec(
                name="replay-debug",
                command=("python", "replay_mcp.py"),
            )
        ]

    def proactive_sources(self) -> list[ProactiveSourceSpec]:
        if not _replay_source_enabled():
            return []
        return [
            ProactiveSourceSpec(
                id="timeline",
                channels=("alert", "content", "context"),
                server="replay-debug",
                fetch_tool="fetch_replay_events",
                ack_tool="acknowledge_replay_events",
                fetch_page_size=50,
            )
        ]
