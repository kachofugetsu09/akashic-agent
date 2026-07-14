from __future__ import annotations

from uuid import uuid4


def new_thread_id() -> str:
    return f"programmatic:{uuid4()}"


def new_turn_id() -> str:
    return f"turn:{uuid4()}"


def new_item_id() -> str:
    return f"item:{uuid4()}"


def new_operation_id() -> str:
    return f"operation:{uuid4()}"
