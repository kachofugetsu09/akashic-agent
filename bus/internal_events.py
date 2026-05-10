from __future__ import annotations

from dataclasses import dataclass

SPAWN_COMPLETED = "spawn_completed"


@dataclass(frozen=True)
class SpawnCompletionEvent:
    job_id: str
    label: str
    task: str
    status: str
    exit_reason: str
    result: str
    retry_count: int = 0
    profile: str = ""
