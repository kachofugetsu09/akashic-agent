from __future__ import annotations

import ast
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from eval.longmemeval.dataset import LMETurn

SUPPORTED_QUESTION_TYPES = (
    "acknowledge_latest_user_preferences",
    "generalize_to_new_scenarios",
    "provide_preference_aligned_recommendations",
    "recall_user_shared_facts",
    "revisit_reasons_behind_preference_updates",
    "suggest_new_ideas",
    "track_full_preference_evolution",
)

_LABEL_RE = re.compile(r"\(([a-z])\)")


@dataclass
class PersonaMemInstance:
    question_id: str
    question_type: str
    question: str
    gold_label: str
    gold_option: str
    all_options: list[str]
    persona_id: str
    topic: str
    shared_context_id: str
    end_index_in_shared_context: int
    persona_profile: str = ""
    question_date: str = ""
    haystack_session_ids: list[str] = field(default_factory=list)
    haystack_dates: list[str] = field(default_factory=list)
    haystack_sessions: list[list[LMETurn]] = field(default_factory=list)
    answer_session_ids: list[str] = field(default_factory=list)

    @property
    def session_key(self) -> str:
        return f"pm:{self.question_id}"

    @property
    def qa_session_key(self) -> str:
        return f"pm:{self.question_id}:qa"


def _parse_options(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception as exc:
        raise ValueError(f"failed to parse all_options: {text[:120]!r}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"all_options must be a list, got {type(parsed)}")
    return [str(item).strip() for item in parsed]


def _parse_label(raw: str) -> str:
    text = (raw or "").strip().lower()
    match = _LABEL_RE.search(text)
    if match:
        return f"({match.group(1)})"
    if len(text) == 1 and "a" <= text <= "z":
        return f"({text})"
    raise ValueError(f"invalid correct_answer label: {raw!r}")


def _strip_role_prefix(role: str, content: str) -> str:
    text = str(content or "").strip()
    if role == "user" and text.lower().startswith("user:"):
        return text[5:].strip()
    if role == "assistant" and text.lower().startswith("assistant:"):
        return text[10:].strip()
    return text


def _load_shared_contexts(path: Path) -> dict[str, list[dict]]:
    contexts: dict[str, list[dict]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"shared context line must be a dict, got {type(payload)}")
            for key, value in payload.items():
                if not isinstance(value, list):
                    raise ValueError(f"shared context {key!r} must be a list")
                contexts[str(key)] = value
    return contexts


def _build_turns(raw_messages: list[dict]) -> tuple[str, list[LMETurn]]:
    persona_parts: list[str] = []
    turns: list[LMETurn] = []

    for item in raw_messages:
        role = str(item.get("role", "") or "").strip().lower()
        content = _strip_role_prefix(role, str(item.get("content", "")))
        if not content:
            continue
        if role == "system":
            persona_parts.append(content)
            continue
        if role not in {"user", "assistant"}:
            continue
        turns.append(LMETurn(role=role, content=content))

    return "\n\n".join(persona_parts).strip(), turns


def _split_into_sessions(turns: list[LMETurn]) -> list[list[LMETurn]]:
    sessions: list[list[LMETurn]] = []
    current: list[LMETurn] = []

    for turn in turns:
        current.append(turn)
        if turn.role == "assistant":
            sessions.append(current)
            current = []

    if current:
        if sessions:
            sessions[-1].extend(current)
        else:
            sessions.append(current)

    return [session for session in sessions if session]


def load_dataset(
    questions_path: Path | str,
    shared_contexts_path: Path | str,
) -> list[PersonaMemInstance]:
    questions_path = Path(questions_path)
    shared_contexts_path = Path(shared_contexts_path)
    shared_contexts = _load_shared_contexts(shared_contexts_path)

    instances: list[PersonaMemInstance] = []
    with questions_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            question_id = str(row["question_id"]).strip()
            shared_context_id = str(row["shared_context_id"]).strip()
            end_index = int(str(row["end_index_in_shared_context"]).strip())
            raw_context = list(shared_contexts.get(shared_context_id) or [])
            if not raw_context:
                raise ValueError(
                    f"shared_context_id not found for question {question_id}: {shared_context_id}"
                )

            sliced_context = raw_context[:end_index]
            persona_profile, turns = _build_turns(sliced_context)
            sessions = _split_into_sessions(turns)
            all_options = _parse_options(str(row.get("all_options", "")))
            gold_label = _parse_label(str(row.get("correct_answer", "")))
            gold_index = ord(gold_label[1]) - ord("a")
            if gold_index < 0 or gold_index >= len(all_options):
                raise ValueError(
                    f"gold option out of range for question {question_id}: {gold_label}"
                )

            instances.append(
                PersonaMemInstance(
                    question_id=question_id,
                    question_type=str(row.get("question_type", "")).strip(),
                    question=str(row.get("user_question_or_message", "")).strip(),
                    gold_label=gold_label,
                    gold_option=all_options[gold_index],
                    all_options=all_options,
                    persona_id=str(row.get("persona_id", "")).strip(),
                    topic=str(row.get("topic", "")).strip(),
                    shared_context_id=shared_context_id,
                    end_index_in_shared_context=end_index,
                    persona_profile=persona_profile,
                    haystack_session_ids=[
                        f"{shared_context_id}:{index}" for index in range(len(sessions))
                    ],
                    haystack_dates=[""] * len(sessions),
                    haystack_sessions=sessions,
                    answer_session_ids=[
                        f"{shared_context_id}:{index}" for index in range(len(sessions))
                    ],
                )
            )

    return instances
