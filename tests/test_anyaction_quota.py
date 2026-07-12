import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from plugins.default_proactive.anyaction import QuotaStore


def _valid_state() -> dict[str, object]:
    return {
        "version": 1,
        "window_key": "2025-06-01@08@UTC",
        "next_reset_at": "2025-06-02T08:00:00+00:00",
        "used": 2,
        "last_action_at": "",
    }


def test_quota_store_initializes_only_when_file_is_missing(tmp_path) -> None:
    store = QuotaStore(tmp_path / "quota.json")

    snapshot = store.snapshot(
        now_utc=datetime(2025, 6, 1, 12, tzinfo=timezone.utc),
        reset_hour=8,
        timezone_name="UTC",
    )

    assert snapshot.used == 0


def test_quota_store_preserves_valid_version_one_state(tmp_path) -> None:
    path = tmp_path / "quota.json"
    path.write_text(json.dumps(_valid_state()), encoding="utf-8")

    store = QuotaStore(path)

    assert store._state == _valid_state()


@pytest.mark.parametrize(
    "payload, field",
    [
        ([], "JSON 对象"),
        ({"version": 1}, "fields="),
        ({**_valid_state(), "version": "1"}, "field=version"),
        ({**_valid_state(), "used": -1}, "field=used"),
        ({**_valid_state(), "window_key": "bad"}, "field=window_key"),
        ({**_valid_state(), "next_reset_at": "bad"}, "field=next_reset_at"),
        ({**_valid_state(), "last_action_at": "2025-06-01"}, "field=last_action_at"),
    ],
)
def test_quota_store_rejects_invalid_state_schema(tmp_path, payload, field) -> None:
    path = tmp_path / "quota.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        QuotaStore(path)


def test_quota_store_preserves_json_decode_error(tmp_path) -> None:
    path = tmp_path / "quota.json"
    path.write_text("not-json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        QuotaStore(path)


def test_quota_store_preserves_read_error(monkeypatch, tmp_path) -> None:
    path = tmp_path / "quota.json"
    path.write_text(json.dumps(_valid_state()), encoding="utf-8")

    def fail_read_text(_path: Path, *, encoding: str) -> str:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_text", fail_read_text)
    with pytest.raises(PermissionError, match="denied"):
        QuotaStore(path)
