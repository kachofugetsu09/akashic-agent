"""Report wall time for real preparation and release work."""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from collections.abc import Iterator


@contextmanager
def measure(stage: str, **fields: object) -> Iterator[dict[str, object]]:
    """Emit start and end records, including failed or cancelled work."""
    started = time.monotonic_ns()
    record = {"event": "release.timing", "stage": stage, **fields}
    print(json.dumps({**record, "status": "started"}), file=sys.stderr, flush=True)
    result: dict[str, object] = {}
    status = "failed"
    try:
        yield result
        status = "complete"
    finally:
        print(json.dumps({**record, **result, "status": status,
                          "seconds": round((time.monotonic_ns() - started) / 1e9, 6)}),
              file=sys.stderr, flush=True)


# 不包在 with 里的耗时（例如进程收尾的各段）按同一格式补记一条完成记录。
def report(stage: str, seconds: float, **fields: object) -> None:
    print(json.dumps({"event": "release.timing", "stage": stage, **fields, "status": "complete",
                      "seconds": round(seconds, 6)}), file=sys.stderr, flush=True)
