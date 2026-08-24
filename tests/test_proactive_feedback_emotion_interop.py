from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sqlite3
import sys
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from agent.control.timer import TimerReceipt, TimerStatus
from agent.plugin_composition import (
    RUNTIME_STARTED,
    RUNTIME_STOPPING,
    SESSION_READ,
    TIMERS,
    TOOL_CATALOG,
    UI_SLOTS,
    CompositionRoot,
    PluginRuntime,
    PluginTimers,
    RuntimeStarted,
    RuntimeStopping,
    SessionReadService,
)
from agent.plugin_composition.tool_catalog import PluginTools, _freeze_plugin_tools
from agent.plugin_composition.ui_slots import PluginUiSlots
from agent.turn_events.after_turn import AFTER_TURN_COMMITTED
from bus.events_lifecycle import TurnCommitted


NOW = datetime(2026, 8, 23, 8, tzinfo=UTC)


def _plugin_roots() -> dict[str, Path]:
    raw = os.environ.get("AKASHIC_INTEROP_PLUGIN_ROOTS")
    if raw is None:
        pytest.skip("proactive source interop Gate supplies exact external roots")
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise RuntimeError("AKASHIC_INTEROP_PLUGIN_ROOTS 必须是对象")
    roots = {str(key): Path(str(value)) for key, value in decoded.items()}
    if set(roots) != {"proactive_feedback", "emotion"}:
        raise RuntimeError(f"interop plugin roots 不匹配: {sorted(roots)}")
    return roots


def _load_plugin(root: Path, package: str) -> ModuleType:
    """Load one exact external checkout without copying its domain implementation."""

    entrypoint = root / "plugin.py"
    spec = importlib.util.spec_from_file_location(
        f"{package}.plugin",
        entrypoint,
        submodule_search_locations=[str(root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(str(entrypoint))
    namespace = ModuleType(package)
    namespace.__path__ = [str(root)]  # type: ignore[attr-defined]
    sys.modules[package] = namespace
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ManualHandle:
    def __init__(self, timer_id: str, deadline: datetime) -> None:
        self.id = timer_id
        self.deadline = deadline
        self._future: asyncio.Future[TimerReceipt] = (
            asyncio.get_running_loop().create_future()
        )

    async def result(self) -> TimerReceipt:
        return await asyncio.shield(self._future)

    async def cancel(self) -> TimerReceipt:
        if not self._future.done():
            self._future.set_result(self._receipt(TimerStatus.CANCELLED))
        return await asyncio.shield(self._future)

    async def cleanup(self) -> None:
        _ = await self.cancel()

    def fire(self) -> None:
        self._future.set_result(self._receipt(TimerStatus.FIRED))

    def _receipt(self, status: TimerStatus) -> TimerReceipt:
        return TimerReceipt(self.id, self.deadline, self.deadline, status)


class ManualTimer:
    def __init__(self) -> None:
        self.handles: list[ManualHandle] = []

    def schedule(self, deadline: datetime) -> ManualHandle:
        handle = ManualHandle(f"timer-{len(self.handles) + 1}", deadline)
        self.handles.append(handle)
        return handle

    @property
    def active(self) -> tuple[ManualHandle, ...]:
        return tuple(handle for handle in self.handles if not handle._future.done())


class EmptyDrift:
    def propose(self, *args: object, **kwargs: object) -> dict[str, object]:
        return {"inserted": True}

    def selection(self, accepted_turn: object) -> None:
        return None


class DeterministicEmbedder:
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _session_state(*, explicit_quote: bool) -> SimpleNamespace:
    user = (
        "被回复消息：主动提醒某个很长很长的主题\n\n"
        "【你当前新消息】我继续这个主题"
        if explicit_quote
        else "我继续这个主题"
    )
    return SimpleNamespace(
        messages=[
            {
                "id": "p1",
                "seq": 1,
                "role": "assistant",
                "content": "主动提醒某个很长很长的主题",
                "extra": '{"proactive": true}',
                "ts": "2026-08-23T07:59:00+00:00",
            },
            {
                "id": "u1",
                "seq": 2,
                "role": "user",
                "content": user,
                "extra": None,
                "ts": "2026-08-23T08:00:00+00:00",
            },
            {
                "id": "a1",
                "seq": 3,
                "role": "assistant",
                "content": "我接着回答这个主题",
                "extra": None,
                "ts": "2026-08-23T08:00:01+00:00",
            },
        ],
        last_consolidated=0,
    )


def _followup(*, explicit_quote: bool) -> TurnCommitted:
    user = (
        "被回复消息：主动提醒某个很长很长的主题\n\n"
        "【你当前新消息】我继续这个主题"
        if explicit_quote
        else "我继续这个主题"
    )
    return TurnCommitted(
        session_key="wake:interop",
        channel="mobile",
        chat_id="chat",
        input_message=user,
        persisted_user_message=user,
        assistant_response="我接着回答这个主题",
        tools_used=[],
        turn_id="turn-followup-1",
        persisted_user_message_id="u1",
        assistant_message_id="a1",
        timestamp=NOW,
    )


async def _eventually(predicate: Any) -> None:
    for _ in range(300):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("interop state did not settle")


def _count(path: Path, query: str) -> int:
    if not path.exists():
        return 0
    with closing(sqlite3.connect(path)) as connection:
        return int(connection.execute(query).fetchone()[0])


async def _mount(
    root: CompositionRoot,
    module: ModuleType,
    plugin_id: str,
    plugin_root: Path,
    sandbox: Path,
) -> None:
    workspace_roots = ("emotion",) if plugin_id == "emotion" else ()
    _ = await root.mount(
        lambda ctx: module.apply(ctx, object()),
        name=plugin_id,
        inject=module.inject,
        runtime=PluginRuntime(
            plugin_id=plugin_id,
            plugin_dir=plugin_root,
            data_dir=sandbox / "plugin-data" / plugin_id,
            workspace=sandbox / "workspace",
            config=None,
            workspace_roots=workspace_roots,
            data_access="read_write",
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "order",
    (("emotion", "proactive_feedback"), ("proactive_feedback", "emotion")),
)
@pytest.mark.parametrize("explicit_quote", (False, True))
async def test_wake_followup_reaches_emotion_once_on_next_timer(
    tmp_path: Path,
    order: tuple[str, str],
    explicit_quote: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove PF owns acceptance while Emotion pulls the immutable fact one tick later."""

    # 1. Mount both exact plugins over ordinary Core services in either order.
    roots = _plugin_roots()
    modules = {
        "proactive_feedback": _load_plugin(
            roots["proactive_feedback"], f"pf_interop_{order[0]}"
        ),
        "emotion": _load_plugin(roots["emotion"], f"emotion_interop_{order[0]}"),
    }
    monkeypatch.setattr(
        modules["proactive_feedback"],
        "_build_embedder",
        lambda _workspace: DeterministicEmbedder(),
    )
    root = CompositionRoot("pf-emotion-" + "-".join(order))
    timer = ManualTimer()
    tools = PluginTools(root.instance_token)
    ui = PluginUiSlots()
    drift = EmptyDrift()
    session_read = SessionReadService(
        cast(Any, lambda _key: (_session_state(explicit_quote=explicit_quote), None))
    )
    _ = await root.context.provide(SESSION_READ, session_read)
    _ = await root.context.provide(TIMERS, PluginTimers(timer))
    _ = await root.context.provide(TOOL_CATALOG, tools)
    _ = await root.context.provide(UI_SLOTS, ui)
    _ = await root.context.provide(modules["emotion"].DRIFT_PROPOSALS, drift)
    _ = await root.context.provide(modules["emotion"].DRIFT_WAKE, drift)
    for plugin_id in order:
        await _mount(root, modules[plugin_id], plugin_id, roots[plugin_id], tmp_path)
    _ = _freeze_plugin_tools(
        tools,
        root.instance_token,
        {plugin_id: root.generation_id for plugin_id in order},
    )

    feedback_db = tmp_path / "plugin-data/proactive_feedback/proactive_feedback.db"
    emotion_db = tmp_path / "workspace/emotion/emotion.db"
    try:
        await root.context.serial(RUNTIME_STARTED, RuntimeStarted())
        await _eventually(lambda: len(timer.active) >= 2)

        # 2. One ordinary committed follow-up is accepted by PF; Emotion has no PF import yet.
        root.context.emit(
            AFTER_TURN_COMMITTED,
            _followup(explicit_quote=explicit_quote),
        )
        await _eventually(
            lambda: _count(
                feedback_db, "SELECT count(*) FROM proactive_feedback_events"
            )
            == 1
        )
        assert _count(
            emotion_db,
            "SELECT count(*) FROM emotion_events "
            "WHERE source_plugin='proactive_feedback'",
        ) == 0
        assert _count(emotion_db, "SELECT count(*) FROM emotion_events") == (
            1 if explicit_quote else 0
        )
        assert _count(
            emotion_db, "SELECT count(*) FROM emotion_feedback_samples"
        ) == (1 if explicit_quote else 0)
        assert _count(
            emotion_db,
            "SELECT row_id FROM pf_history_cursor WHERE source='proactive_feedback'",
        ) == 0

        # 3. The ordinary immediate Timer pulls once; explicit quote keeps one effect/sample.
        history_timer = min(timer.active, key=lambda handle: handle.deadline)
        history_timer.fire()
        await _eventually(
            lambda: _count(
                emotion_db,
                "SELECT row_id FROM pf_history_cursor "
                "WHERE source='proactive_feedback'",
            )
            == 1
        )
        assert _count(emotion_db, "SELECT count(*) FROM emotion_events") == (
            2 if explicit_quote else 1
        )
        assert _count(
            emotion_db, "SELECT count(*) FROM emotion_feedback_samples"
        ) == 1
        assert _count(
            emotion_db,
            "SELECT count(*) FROM emotion_events "
            "WHERE valence_delta != 0.0 OR dominance_delta != 0.0",
        ) == 1
        assert _count(
            emotion_db,
            "SELECT count(*) FROM emotion_events "
            "WHERE source_type='explicit_quote_already_applied'",
        ) == (1 if explicit_quote else 0)
    finally:
        await root.context.serial(RUNTIME_STOPPING, RuntimeStopping())
        await root.dispose()
