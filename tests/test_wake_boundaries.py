from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re

from plugins.wake._boundary import ToolResultValue
from plugins.wake.api import DeliveryTarget
from plugins.wake.request import Request


WAKE_ROOT = Path(__file__).parents[1] / "plugins" / "wake"


def test_wake_does_not_import_other_plugin_implementations() -> None:
    """Wake 只通过自己的窄边界连接外部 owner。"""
    implementation_import = re.compile(r"^\s*(?:from|import) plugins\.")
    offenders = {
        path.relative_to(WAKE_ROOT).as_posix(): line
        for path in WAKE_ROOT.glob("*.py")
        for line in path.read_text(encoding="utf-8").splitlines()
        if implementation_import.match(line)
    }
    assert offenders == {}


def test_request_archives_sink_as_plain_boundary_mapping() -> None:
    request = Request(
        flow_id="a" * 32,
        owner="alert",
        now=datetime(2026, 9, 12, tzinfo=timezone.utc),
        timezone="UTC",
        target=DeliveryTarget(channel="test", recipient="room", session_id="test:room"),
        sink={"name": "test", "binding_id": "delivery-binding", "address": "room"},
        program_binding="wake-program",
        tools={"share_alert": "alert-tool"},
        snapshot_seq=0,
        alert_ref={"source_id": "source", "event_id": "event", "mail_id": "mail"},
        rules="",
        history="",
    )

    restored = Request.model_validate(request.model_dump())
    assert restored.sink == {
        "name": "test",
        "binding_id": "delivery-binding",
        "address": "room",
    }


def test_wake_tool_result_is_structural() -> None:
    result = ToolResultValue("success", ())
    assert result.outcome == "success"
    assert result.parts == ()
