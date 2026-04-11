from datetime import datetime, timezone

from proactive_v2.presence import PresenceStore
from session.store import SessionStore


def _utc(year, month, day, hour=0, minute=0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def _store(tmp_path) -> SessionStore:
    return SessionStore(tmp_path / "sessions.db")


def test_fresh_store_has_no_sessions(tmp_path):
    store = PresenceStore(_store(tmp_path))
    assert store.get_all_sessions() == {}


def test_record_user_message_stores_timestamp(tmp_path):
    store = PresenceStore(_store(tmp_path))
    t = _utc(2026, 2, 23, 10, 0)
    store.record_user_message("telegram:123", now=t)
    assert store.get_last_user_at("telegram:123") == t


def test_record_user_message_overwrites_previous(tmp_path):
    store = PresenceStore(_store(tmp_path))
    t1 = _utc(2026, 2, 20, 10, 0)
    t2 = _utc(2026, 2, 23, 10, 0)
    store.record_user_message("telegram:123", now=t1)
    store.record_user_message("telegram:123", now=t2)
    assert store.get_last_user_at("telegram:123") == t2


def test_record_proactive_sent_stores_timestamp(tmp_path):
    store = PresenceStore(_store(tmp_path))
    t = _utc(2026, 2, 22, 15, 0)
    store.record_proactive_sent("telegram:123", now=t)
    assert store.get_last_proactive_at("telegram:123") == t


def test_nonexistent_session_returns_none(tmp_path):
    store = PresenceStore(_store(tmp_path))
    assert store.get_last_user_at("no:session") is None
    assert store.get_last_proactive_at("no:session") is None


def test_most_recent_user_at_returns_latest_across_sessions(tmp_path):
    store = PresenceStore(_store(tmp_path))
    t_old = _utc(2026, 2, 20, 10, 0)
    t_new = _utc(2026, 2, 23, 8, 0)
    store.record_user_message("telegram:111", now=t_old)
    store.record_user_message("qq:222", now=t_new)
    assert store.most_recent_user_at() == t_new


def test_most_recent_user_at_is_none_when_no_sessions(tmp_path):
    store = PresenceStore(_store(tmp_path))
    assert store.most_recent_user_at() is None


def test_get_all_sessions_returns_both_sessions(tmp_path):
    store = PresenceStore(_store(tmp_path))
    t1 = _utc(2026, 2, 20, 10, 0)
    t2 = _utc(2026, 2, 23, 8, 0)
    store.record_user_message("telegram:111", now=t1)
    store.record_user_message("qq:222", now=t2)
    all_s = store.get_all_sessions()
    assert set(all_s.keys()) == {"telegram:111", "qq:222"}
    assert all_s["telegram:111"]["last_user_at"] == t1
    assert all_s["qq:222"]["last_user_at"] == t2


def test_persistence_survives_reload(tmp_path):
    db_path = tmp_path / "sessions.db"
    t_user = _utc(2026, 2, 23, 10, 0)
    t_pro = _utc(2026, 2, 22, 15, 0)

    store = PresenceStore(SessionStore(db_path))
    store.record_user_message("telegram:123", now=t_user)
    store.record_proactive_sent("telegram:123", now=t_pro)

    store2 = PresenceStore(SessionStore(db_path))
    assert store2.get_last_user_at("telegram:123") == t_user
    assert store2.get_last_proactive_at("telegram:123") == t_pro
