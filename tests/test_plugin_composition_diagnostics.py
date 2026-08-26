from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, cast

import pytest

from agent.control.context import running_turn_id
from agent.plugin_composition import (
    CompositionRoot,
    PluginRuntime,
    SerialEventKey,
)
from agent.plugin_composition.diagnostics import CorePluginDiagnostics
from core.error_context import current_client_message_id, current_session_key

RUN = SerialEventKey[str, object]("probe.run")


def _fields(record: logging.LogRecord) -> dict[str, object]:
    return cast(dict[str, object], getattr(record, "akashic_fields"))


def _runtime(tmp_path: Path) -> PluginRuntime:
    return PluginRuntime(
        plugin_id="probe@builtin",
        generation_id="test-generation",
        plugin_dir=tmp_path / "plugin",
        data_dir=tmp_path / "data",
        workspace=tmp_path / "workspace",
        config=None,
    )


@pytest.mark.asyncio
async def test_core_boundary_and_plugin_details_share_one_parent_chain(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="akashic.plugin.diagnostics")
    root = CompositionRoot("generation-probe")

    async def apply(ctx: Any) -> None:
        async def listener(_: str) -> None:
            with ctx.diagnostics.operation("work.retrieve"):
                ctx.diagnostics.measure("candidates", 7)

        _ = await ctx.on(RUN, listener)

    _ = await root.mount(
        apply,
        name="probe",
        runtime=_runtime(tmp_path),
    )
    await root.context.serial(RUN, "payload")

    records = [_fields(record) for record in caplog.records]
    boundary = next(
        item
        for item in records
        if item["event"] == "plugin.operation.start"
        and item["operation"] == "event.serial"
    )
    internal = next(
        item
        for item in records
        if item["event"] == "plugin.operation.start"
        and item["operation"] == "work.retrieve"
    )
    measurement = next(
        item for item in records if item["event"] == "plugin.measurement"
    )
    terminal = next(
        item
        for item in records
        if item["event"] == "plugin.operation.done"
        and item["operation_id"] == boundary["operation_id"]
    )

    assert boundary["plugin_id"] == "probe@builtin"
    assert boundary["generation_id"] == "test-generation"
    assert boundary["plugin_entrypoint"] == "probe.run"
    assert internal["parent_operation_id"] == boundary["operation_id"]
    assert measurement["operation_id"] == internal["operation_id"]
    assert measurement["measurement"] == "candidates"
    assert measurement["measurement_value"] == 7
    assert terminal["outcome"] == "success"
    assert cast(float, terminal["duration_ms"]) >= 0


def test_plugin_measurements_reject_dynamic_or_non_finite_values() -> None:
    diagnostics = CorePluginDiagnostics(
        plugin_id="probe",
        generation_id="generation",
        fiber="probe",
    )

    with pytest.raises(ValueError, match="measurement 无效"):
        diagnostics.measure("bad name", 1)
    with pytest.raises(ValueError, match="有限数字"):
        diagnostics.measure("ratio", float("nan"))
    with pytest.raises(TypeError, match="必须是数字"):
        diagnostics.measure("enabled", cast(Any, True))
    with pytest.raises(ValueError, match="unit 无效"):
        diagnostics.measure("latency", 1, unit="milliseconds")

    with pytest.raises(ValueError, match="Core 签发"):
        with diagnostics.resume(cast(Any, object())):
            pass


def test_diagnostic_sink_failure_does_not_change_plugin_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostics = CorePluginDiagnostics(
        plugin_id="probe",
        generation_id="generation",
        fiber="probe",
    )

    def broken_sink(*_: object, **__: object) -> None:
        raise OSError("sink unavailable")

    monkeypatch.setattr(
        "agent.plugin_composition.diagnostics.log_event",
        broken_sink,
    )

    with diagnostics.operation("work"):
        diagnostics.measure("items", 1)


@pytest.mark.parametrize(
    ("error", "terminal", "outcome"),
    [
        (RuntimeError("broken"), "plugin.operation.error", "error"),
        (asyncio.CancelledError(), "plugin.operation.cancelled", "cancelled"),
    ],
)
def test_plugin_operation_records_error_and_cancelled_terminals(
    error: BaseException,
    terminal: str,
    outcome: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="akashic.plugin.diagnostics")
    diagnostics = CorePluginDiagnostics(
        plugin_id="probe",
        generation_id="generation",
        fiber="probe",
    )

    with pytest.raises(type(error)):
        with diagnostics.operation("work"):
            raise error

    record = next(
        _fields(item)
        for item in caplog.records
        if _fields(item).get("event") == terminal
    )
    assert record["outcome"] == outcome
    assert record["error_type"] == type(error).__name__


def test_capture_resume_preserves_handoff_parent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="akashic.plugin.diagnostics")
    diagnostics = CorePluginDiagnostics(
        plugin_id="probe",
        generation_id="generation",
        fiber="probe",
    )

    session_token = current_session_key.set("session-probe")
    turn_token = running_turn_id.set("turn-probe")
    client_token = current_client_message_id.set("client-probe")
    try:
        with diagnostics.operation("enqueue") as enqueue:
            captured = diagnostics.capture()
        clear_session_token = current_session_key.set(None)
        clear_turn_token = running_turn_id.set("")
        clear_client_token = current_client_message_id.set("")
        try:
            with diagnostics.resume(captured):
                with diagnostics.operation("dequeue"):
                    pass
        finally:
            current_client_message_id.reset(clear_client_token)
            running_turn_id.reset(clear_turn_token)
            current_session_key.reset(clear_session_token)
    finally:
        current_client_message_id.reset(client_token)
        running_turn_id.reset(turn_token)
        current_session_key.reset(session_token)

    dequeue = next(
        _fields(record)
        for record in caplog.records
        if _fields(record).get("event") == "plugin.operation.start"
        and _fields(record).get("operation") == "dequeue"
    )
    assert dequeue["parent_operation_id"] == enqueue.operation_id
    assert dequeue["session_id"] == "session-probe"
    assert dequeue["turn_id"] == "turn-probe"
    assert dequeue["client_message_id"] == "client-probe"


@pytest.mark.asyncio
async def test_plugin_effect_cleanup_uses_same_lifecycle_boundary(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="akashic.plugin.diagnostics")
    cleaned: list[bool] = []

    async def apply(ctx: Any) -> None:
        _ = await ctx.effect(lambda: lambda: cleaned.append(True))

    root = CompositionRoot("composition-generation")
    fiber = await root.mount(
        apply,
        name="probe",
        runtime=_runtime(tmp_path),
    )
    await fiber.dispose()

    terminal = next(
        _fields(record)
        for record in caplog.records
        if _fields(record).get("event") == "plugin.operation.done"
        and _fields(record).get("operation") == "lifecycle.cleanup"
    )
    assert cleaned == [True]
    assert terminal["plugin_id"] == "probe@builtin"
    assert terminal["generation_id"] == "test-generation"
