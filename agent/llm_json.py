from __future__ import annotations

from typing import Any

import json_repair


def strip_json_fence(text: str) -> str:
    payload = text.strip()
    if payload.startswith("```"):
        payload = payload.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return payload


def load_json_object_loose(text: str) -> dict[str, Any] | None:
    payload = strip_json_fence(text)
    data = json_repair.loads(payload)
    if isinstance(data, dict):
        return data
    return None
