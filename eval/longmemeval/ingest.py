"""Phase 1+2: insert haystack messages into SessionStore, then consolidate."""

from __future__ import annotations

import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .dataset import LMEInstance
from .runtime import BenchmarkRuntime

logger = logging.getLogger(__name__)
_FINALIZE_CHUNK_SIZE = 80


def _last_dialogue_pair(turns) -> tuple[str, str]:
    last_user = ""
    last_assistant = ""

    for turn in reversed(turns):
        role = str(getattr(turn, "role", "") or "")
        content = str(getattr(turn, "content", "") or "").strip()
        if not content:
            continue
        if not last_assistant and role == "assistant":
            last_assistant = content
            continue
        if role == "user":
            last_user = content
            break

    return last_user, last_assistant


def _parse_date(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return datetime.now(tz=timezone.utc).isoformat()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            continue
    return raw


def _ingest_state_path(rt: BenchmarkRuntime, question_id: str) -> Path:
    return rt.workspace / "ingest_state.json"


def _load_ingest_state(rt: BenchmarkRuntime, question_id: str) -> dict | None:
    path = _ingest_state_path(rt, question_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("failed to load ingest state: %s", path)
        return None


def _write_ingest_state(
    rt: BenchmarkRuntime,
    question_id: str,
    *,
    completed: bool,
    expected_turns: int,
    ingested_turns: int,
) -> None:
    _ingest_state_path(rt, question_id).write_text(
        json.dumps(
            {
                "question_id": question_id,
                "completed": completed,
                "expected_turns": expected_turns,
                "ingested_turns": ingested_turns,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _is_ingested(rt: BenchmarkRuntime, question_id: str) -> bool:
    state = _load_ingest_state(rt, question_id)
    return bool(state and state.get("completed") is True)


async def _finalize_tail_chunks(rt: BenchmarkRuntime, session) -> None:
    remaining = session.messages[session.last_consolidated :]
    if not remaining:
        return

    session_cls = session.__class__
    for start in range(0, len(remaining), _FINALIZE_CHUNK_SIZE):
        chunk = remaining[start : start + _FINALIZE_CHUNK_SIZE]
        temp_session = session_cls(key=session.key)
        temp_session.messages = list(chunk)
        temp_session.last_consolidated = 0
        for attr in ("_channel", "_chat_id"):
            if hasattr(session, attr):
                setattr(temp_session, attr, getattr(session, attr))
        await rt.consolidation.consolidate(temp_session, archive_all=True)


async def ingest_instance(
    rt: BenchmarkRuntime,
    instance: LMEInstance,
    *,
    force: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Insert all haystack sessions and run consolidation per session boundary.

    Returns total turn count. Calls on_progress(done, total) after each session.
    """
    session_key = instance.session_key
    sm = rt.core.session_manager

    expected_turns = sum(len(turns) for turns in instance.haystack_sessions)
    if not force and _is_ingested(rt, instance.question_id):
        logger.info("skip ingest (already done): %s", session_key)
        return 0

    dates = instance.haystack_dates
    sessions = instance.haystack_sessions

    if not sessions:
        logger.warning("instance %s has no haystack sessions", instance.question_id)
        return 0

    while len(dates) < len(sessions):
        dates.append("")

    total_turns = 0
    n = len(sessions)
    _write_ingest_state(
        rt,
        instance.question_id,
        completed=False,
        expected_turns=expected_turns,
        ingested_turns=0,
    )

    for idx, (date, turns) in enumerate(zip(dates, sessions)):
        ts = _parse_date(date)

        sm._cache.pop(session_key, None)
        session = sm.get_or_create(session_key)

        for turn in turns:
            session.add_message(turn.role, turn.content)
            session.messages[-1]["timestamp"] = ts
            total_turns += 1

        sm.save(session)
        sm._cache.pop(session_key, None)
        session = sm.get_or_create(session_key)

        await rt.consolidation.consolidate(session, archive_all=False)
        sm.save(session)

        worker = getattr(rt.core.memory_runtime, "post_response_worker", None)
        if worker is not None:
            user_msg, agent_response = _last_dialogue_pair(turns)
            if user_msg:
                await worker.run(
                    user_msg,
                    agent_response,
                    [],
                    source_ref=f"{session_key}#post:{idx}",
                    session_key=session_key,
                )

        if on_progress:
            on_progress(idx + 1, n)

    # Finalize the unarchived tail in bounded chunks so the benchmark
    # does not lose late-session facts while still avoiding giant prompts.
    sm._cache.pop(session_key, None)
    session = sm.get_or_create(session_key)
    await _finalize_tail_chunks(rt, session)
    session.last_consolidated = len(session.messages)
    sm.save(session)
    _write_ingest_state(
        rt,
        instance.question_id,
        completed=True,
        expected_turns=expected_turns,
        ingested_turns=total_turns,
    )

    logger.info(
        "ingest done: %s  sessions=%d  turns=%d",
        session_key,
        len(sessions),
        total_turns,
    )
    return total_turns
