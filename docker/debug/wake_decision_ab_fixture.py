from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sqlite3
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

import agent.plugins.manager as plugin_manager_module
import plugins.wake.plugin as wake_plugin_module
from agent.model_runtime.types import ToolCall
from agent.plugin_composition.durable_delivery_store import DurableDeliveryStore
from agent.provider import LLMResponse
from docker.debug.wake_v3_provider_e2e import (
    ControlledTimer,
    CountingProvider,
    _build_stack,
    _eventually,
    _write_plugin_configs,
)
from plugins.eventmail.store import EventMailStore
from tests.fixtures.content_clock_source.plugin import FixtureSourceStore

_FIXTURE_PATH = (
    _SOURCE_ROOT / "tests" / "fixtures" / "wake_decision_ab_v1.json"
)
_INVALID_MARKER = "INVALID_DECISION_MUST_NOT_LEAK_7F3A"


class InvalidThenValidProvider:
    """Return the production failure shape, then one valid repair decision."""

    context_window = 1_000_000

    def __init__(self) -> None:
        self.decision_requests = 0

    async def chat(self, **kwargs: object) -> LLMResponse:
        tools = kwargs.get("tools")
        if not isinstance(tools, list) or not tools:
            return LLMResponse(content="Wake decision recorded.")
        tool_names = {
            str(cast(Mapping[str, object], tool.get("function", {})).get("name"))
            for tool in tools
            if isinstance(tool, Mapping)
        }
        prompt = json.dumps(kwargs.get("messages"), ensure_ascii=False)
        candidate = re.search(r"candidate_[0-9a-f]{16}", prompt)
        if candidate is None:
            raise RuntimeError("Wake repair fixture prompt is missing candidate_id")
        if "screen_content" in tool_names:
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        "call:wake-screen",
                        "screen_content",
                        {
                            "items": [
                                {
                                    "candidate_id": candidate.group(0),
                                    "initial_interest": "likely_interesting",
                                    "question": "Is this genuinely useful?",
                                }
                            ]
                        },
                    )
                ],
            )
        self.decision_requests += 1
        if self.decision_requests == 1:
            return LLMResponse(content=f"Useful items. {_INVALID_MARKER}")
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    "call:wake-repair-share",
                    "share_content",
                    {
                        "message": "One memory item is worth sharing.",
                        "items": [candidate.group(0)],
                    },
                )
            ],
        )

    def estimate_context_tokens(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> int:
        return max(1, len(json.dumps([messages, tools], ensure_ascii=False)) // 4)


def load_fixture(path: Path = _FIXTURE_PATH) -> dict[str, object]:
    """Load and validate the frozen Wake decision sample."""

    raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Wake decision A/B fixture schema mismatch")
    candidates = raw.get("candidates")
    expected = raw.get("expected_share_ids")
    if not isinstance(candidates, list) or len(candidates) != 10:
        raise ValueError("Wake decision A/B fixture must contain ten candidates")
    candidate_count = raw.get("candidate_count")
    noise_template = raw.get("noise_template")
    if (
        not isinstance(candidate_count, int)
        or candidate_count < len(candidates)
        or not isinstance(noise_template, dict)
    ):
        raise ValueError("Wake decision A/B candidate expansion is invalid")
    expanded = [*candidates]
    for index in range(len(candidates) + 1, candidate_count + 1):
        expanded.append(
            {
                "candidate_id": f"candidate_noise_{index:03d}",
                "title": str(noise_template.get("title", "")).format(index=index),
                "summary": str(noise_template.get("summary", "")).format(index=index),
                "preprocess_score": noise_template.get("preprocess_score", 0.0),
            }
        )
    raw["candidates"] = expanded
    ids = [_candidate_id(candidate) for candidate in expanded]
    if len(ids) != len(set(ids)):
        raise ValueError("Wake decision A/B candidate ids must be unique")
    if (
        not isinstance(expected, list)
        or not expected
        or any(not isinstance(item, str) or item not in ids for item in expected)
    ):
        raise ValueError("Wake decision A/B expected ids are invalid")
    return cast(dict[str, object], raw)


def fixture_digest(fixture: Mapping[str, object]) -> str:
    """Return the stable identity used to compare later A/B reports."""

    payload = json.dumps(
        fixture, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


async def run_runtime_fixture(root: Path) -> dict[str, object]:
    """Measure the missing-decision fixture through the full Wake runtime."""

    fixture = load_fixture()
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    receipt_db = workspace / "recording-receipts.sqlite3"
    _write_plugin_configs(workspace, receipt_db)
    source_store = FixtureSourceStore(
        workspace / "plugin-data" / "content_clock_source-builtin" / "source.sqlite3"
    )
    candidates = cast(list[dict[str, object]], fixture["candidates"])
    seeded_at = datetime.now(UTC)
    source_store.seed(
        tuple(
            {
                "kind": "fixture",
                "wake_action": "select",
                "title": candidate["title"],
                "summary": candidate["summary"],
                "preprocess_score": candidate["preprocess_score"],
                "published_at": seeded_at.isoformat(),
            }
            for candidate in candidates
        ),
        seeded_at,
    )
    scripted = InvalidThenValidProvider()
    counted = CountingProvider(scripted)
    timer = ControlledTimer()
    original_timer = plugin_manager_module.AsyncioOneShotTimer
    plugin_manager_module.AsyncioOneShotTimer = lambda: timer
    stack = _build_stack(workspace, root, timer, counted)
    try:
        await stack.start()
        await _eventually(lambda: timer.pending_count() >= 1, "SOURCE_TIMER_NOT_ARMED")
        timer.fire_earliest()
        await _eventually(
            lambda: source_store.state(datetime.now(UTC))["cursor"] == 100,
            "SOURCE_CURSOR_NOT_COMMITTED",
        )
        await _eventually(lambda: timer.pending_count() >= 1, "WAKE_TIMER_NOT_ARMED")
        timer.fire_earliest()
        content = EventMailStore(
            workspace / "plugin-data" / "eventmail-builtin" / "eventmail.sqlite3"
        )
        ledger = DurableDeliveryStore(
            workspace / "runtime" / "deliveries" / "settlements.sqlite"
        )
        ledger.initialize()
        await _eventually(
            lambda: content.state_counts()
            in (
                {"deferred": 1, "pending": 99},
                {"delivered": 1, "pending": 99},
                {"pending": 99, "settled": 1},
            ),
            "INVALID_DECISION_NOT_TERMINAL",
        )
        turns = stack.sessions.control_store.list_turns("wake-provider-e2e")
        messages = stack.sessions.control_store.fetch_session_messages(
            "wake-provider-e2e"
        )
        delivery_count = _delivery_count(ledger.path)
        valid_decisions = int(delivery_count == 1)
        return {
            "schema_version": 1,
            "fixture_id": fixture["fixture_id"],
            "fixture_digest": fixture_digest(fixture),
            "variant": "candidate" if valid_decisions else "baseline",
            "provider": "scripted-invalid-then-valid",
            "trials": 1,
            "valid_decisions": valid_decisions,
            "valid_decision_rate": float(valid_decisions),
            "provider_decision_requests": scripted.decision_requests,
            "control_turn_count": len(turns),
            "content_counts": content.state_counts(),
            "delivery_count": delivery_count,
            "session_projection_count": len(messages),
            "invalid_marker_user_leak": any(
                _INVALID_MARKER in str(message) for message in messages
            ),
        }
    finally:
        plugin_manager_module.AsyncioOneShotTimer = original_timer
        await stack.close()


def _candidate_id(value: object) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("Wake decision candidate must be an object")
    candidate_id = value.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("Wake decision candidate_id must be non-empty")
    return candidate_id


def _delivery_count(path: Path) -> int:
    connection = sqlite3.connect(path)
    try:
        return int(connection.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0])
    finally:
        connection.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen Wake decision baseline")
    _ = parser.add_argument("--report", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report_path = Path(args.report)
    try:
        with TemporaryDirectory(prefix="akashic-wake-decision-runtime-") as root:
            report = asyncio.run(run_runtime_fixture(Path(root)))
        report["status"] = "passed"
        exit_code = 0
    except BaseException as error:
        report = {
            "status": "failed",
            "variant": "baseline",
            "failure_code": str(error),
        }
        exit_code = 1
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _ = report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "report": str(report_path)}))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
