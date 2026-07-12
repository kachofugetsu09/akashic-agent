from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier

import pytest

from infra.persistence.json_store import atomic_save_json


def test_atomic_save_json_uses_isolated_temp_files(monkeypatch, tmp_path) -> None:
    path = tmp_path / "state.json"
    replace_barrier = Barrier(2)
    original_replace = Path.replace

    def synchronized_replace(source: Path, target: Path) -> Path:
        replace_barrier.wait(timeout=5)
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", synchronized_replace)
    payloads = ({"writer": "a"}, {"writer": "b"})
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda payload: atomic_save_json(path, payload), payloads))

    assert json.loads(path.read_text(encoding="utf-8")) in payloads
    assert list(tmp_path.glob("state.json.*.tmp")) == []


def test_atomic_save_json_cleans_own_temp_after_replace_failure(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path / "state.json"
    path.write_text('{"version": "old"}', encoding="utf-8")

    def fail_replace(source: Path, target: Path) -> Path:
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        atomic_save_json(path, {"version": "new"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"version": "old"}
    assert list(tmp_path.glob("state.json.*.tmp")) == []
