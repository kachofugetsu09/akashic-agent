#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.control.models import TurnRequest
from agent.control.runtime import ConversationRuntime
from agent.control.scoped_turn import ScopedTurnPort
from session.store import SessionStore


FIXED_NOW = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)


class RecordingScope:
    """Record exact-scope fork and release without owning product state."""

    def __init__(self, events: list[str], *, child: bool = False) -> None:
        self._events = events
        self._active = True
        self._child = child
        if child:
            events.append("scope.fork")

    @property
    def active(self) -> bool:
        return self._active

    def fork(self) -> RecordingScope:
        if not self._active:
            raise RuntimeError("fixture scope 已释放")
        return RecordingScope(self._events, child=True)

    async def release(self) -> None:
        if self._active:
            self._active = False
            kind = "child" if self._child else "owner"
            self._events.append(f"scope.{kind}.release")


class ScriptedTurnExecutor:
    """Return fixed responses while recording the real Runtime request boundary."""

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = iter(responses)
        self.requests: list[dict[str, object]] = []

    async def __call__(self, request: TurnRequest) -> str:
        self.requests.append(
            {
                "sessionId": request.thread_id,
                "input": request.input,
                "metadata": {
                    key: value
                    for key, value in request.metadata.items()
                    if not key.startswith("_control") and key != "turnId"
                },
            }
        )
        try:
            return next(self._responses)
        except StopIteration as error:
            raise RuntimeError("scripted executor 响应已耗尽") from error


async def run_fixture(workspace: Path) -> dict[str, object]:
    """Run one real scoped Turn in a disposable SessionStore and return its receipt."""

    # 1. Build isolated state and deterministic boundary adapters.
    workspace.mkdir(parents=True, exist_ok=True)
    store = SessionStore(workspace / "sessions.db")
    executor = ScriptedTurnExecutor(("fixture:done",))
    runtime = ConversationRuntime(store, executor)
    lifecycle: list[str] = []
    owner = RecordingScope(lifecycle)

    # 2. Admit through the public scoped port and wait for terminal plus cleanup.
    handle = await ScopedTurnPort(runtime, owner).start(
        TurnRequest(
            "fixture:child",
            "inspect fixture",
            {"scope": "fixture", "memoryWrite": False},
        )
    )
    result = await handle.result()
    await handle.cleanup()
    turns = store.list_turns("fixture:child")

    # 3. Project only observable facts; dynamic identity is normalized separately.
    await owner.release()
    receipt: dict[str, object] = {
        "scenario": "scoped-turn-completes",
        "clock": FIXED_NOW.isoformat().replace("+00:00", "Z"),
        "accepted": {
            "sessionId": handle.accepted.session_id,
            "turnId": handle.accepted.turn_id,
        },
        "providerRequests": executor.requests,
        "terminal": {
            "turnId": result.id,
            "status": result.status.value,
            "response": result.final_response,
        },
        "lifecycle": lifecycle,
        "state": {
            "turnRows": len(turns),
            "statuses": [turn.status.value for turn in turns],
        },
        "effects": [],
    }
    await runtime.shutdown()
    store.close()
    return receipt


def normalize_receipt(receipt: Mapping[str, object]) -> dict[str, object]:
    """Normalize only registered Turn identity fields for exact comparison."""

    normalized = json.loads(json.dumps(receipt, ensure_ascii=False))
    accepted = normalized.get("accepted")
    terminal = normalized.get("terminal")
    if isinstance(accepted, dict):
        accepted["turnId"] = "<turn-id>"
    if isinstance(terminal, dict):
        terminal["turnId"] = "<turn-id>"
    requests = normalized.get("providerRequests")
    if isinstance(requests, list):
        for request in requests:
            if not isinstance(request, dict):
                continue
            metadata = request.get("metadata")
            if isinstance(metadata, dict) and "interactionId" in metadata:
                metadata["interactionId"] = "<turn-id>"
    return normalized


def receipt_differences(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
) -> list[str]:
    """Return deterministic JSON paths whose normalized values differ."""

    differences: list[str] = []

    def compare(left: Any, right: Any, path: str) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                if key not in left or key not in right:
                    differences.append(f"{path}.{key}")
                else:
                    compare(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                differences.append(f"{path}.length")
                return
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                compare(left_item, right_item, f"{path}[{index}]")
            return
        if left != right:
            differences.append(path)

    compare(normalize_receipt(expected), normalize_receipt(actual), "$")
    return differences


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the disposable React Core fixture")
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    if args.workspace is not None:
        receipt = asyncio.run(run_fixture(args.workspace.resolve()))
    else:
        with tempfile.TemporaryDirectory(prefix="akashic-react-core-") as raw:
            receipt = asyncio.run(run_fixture(Path(raw)))
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
