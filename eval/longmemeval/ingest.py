"""Insert benchmark haystack messages into the canonical SessionStore."""

from __future__ import annotations

import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .dataset import LMEInstance
from .runtime import BenchmarkRuntime

logger = logging.getLogger(__name__)


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


async def ingest_instance(
    rt: BenchmarkRuntime,
    instance: LMEInstance,
    *,
    force: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Append every haystack session without bypassing production compaction.

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

        if on_progress:
            on_progress(idx + 1, n)

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
