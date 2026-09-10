from __future__ import annotations

from typing import Any

import json_repair


def load_json_object_loose(text: str) -> dict[str, Any] | None:
    payload = text.strip()
    if payload.startswith("```"):
        payload = payload.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    data = json_repair.loads(payload)
    if isinstance(data, dict):
        return data
    return None
