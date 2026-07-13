from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Mapping, Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class ReplayClock:
    def __init__(self, state_path: Path, initial: datetime | None = None) -> None:
        self._state_path = state_path
        self._lock = threading.Lock()
        if initial is not None and not state_path.exists():
            _ = self.set(initial)

    @property
    def state_path(self) -> Path:
        return self._state_path

    def now(self) -> datetime:
        with self._lock:
            return self._read_current()

    def set(self, value: datetime) -> datetime:
        current = _as_utc(value)
        with self._lock:
            return self._write_current(current)

    def advance(self, delta: timedelta) -> datetime:
        with self._lock:
            return self._write_current(self._read_current() + delta)

    def _read_current(self) -> datetime:
        payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        return _as_utc(datetime.fromisoformat(str(payload["current_time"])))

    def _write_current(self, current: datetime) -> datetime:
        payload = {
            "current_time": current.isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        _ = temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _ = temporary.replace(self._state_path)
        return current


def clock_from_env(env: Mapping[str, str] | None = None) -> Clock:
    values = os.environ if env is None else env
    path = str(values.get("AKASHIC_REPLAY_CLOCK_FILE") or "").strip()
    return ReplayClock(Path(path)) if path else SystemClock()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("clock datetime 必须包含时区")
    return value.astimezone(UTC)
