"""LongMemEval dataset loader.

Expected JSON format (per instance):
  question_id, question_type, question, answer, question_date,
  haystack_session_ids, haystack_dates, haystack_sessions, answer_session_ids

haystack_sessions is a list of sessions; each session is a list of
{"role": "user"/"assistant", "content": "..."} turns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_QUESTION_TYPES = (
    "single-session-user",
    "single-session-preference",
    "knowledge-update",
)


@dataclass
class LMETurn:
    role: str
    content: str
    has_answer: bool = False


@dataclass
class LMEInstance:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    haystack_session_ids: list[str]
    haystack_dates: list[str]
    haystack_sessions: list[list[LMETurn]]
    answer_session_ids: list[str] = field(default_factory=list)

    @property
    def session_key(self) -> str:
        return f"lme:{self.question_id}"

    @property
    def qa_session_key(self) -> str:
        return f"lme:{self.question_id}:qa"


def _parse_turns(raw_session: list[dict]) -> list[LMETurn]:
    turns = []
    for t in raw_session:
        turns.append(
            LMETurn(
                role=str(t.get("role", "user")),
                content=str(t.get("content", "")),
                has_answer=bool(t.get("has_answer", False)),
            )
        )
    return turns


def load_dataset(path: Path | str) -> list[LMEInstance]:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array, got {type(raw)}")

    instances: list[LMEInstance] = []
    for item in raw:
        haystack_sessions = [
            _parse_turns(s) for s in (item.get("haystack_sessions") or [])
        ]
        instances.append(
            LMEInstance(
                question_id=str(item["question_id"]),
                question_type=str(item.get("question_type", "")),
                question=str(item["question"]),
                answer=str(item["answer"]),
                question_date=str(item.get("question_date", "")),
                haystack_session_ids=list(item.get("haystack_session_ids") or []),
                haystack_dates=list(item.get("haystack_dates") or []),
                haystack_sessions=haystack_sessions,
                answer_session_ids=list(item.get("answer_session_ids") or []),
            )
        )
    return instances
