from datetime import datetime, timedelta, timezone

from plugins.eventmail.store import EventMailStore


def test_exact_alert_claim_preserves_version_identity_and_completed_claims(tmp_path):
    store = EventMailStore(tmp_path / "eventmail.db")
    store.initialize()
    now = datetime.now(timezone.utc)
    store.report_alert(source_id="source", event_id="original", payload={"body": "original"}, observed_at=now)
    ref = store.peek_alert(now)
    accepted = {"session_id": "wake:flow", "turn_id": "input"}
    # 快照后出现更早的另一条目；原请求仍只能领取准确候选。
    store.report_alert(source_id="source", event_id="earlier", payload={"body": "earlier"}, observed_at=now - timedelta(seconds=1))
    selected = store.select_alert(accepted, now, item_ref=ref)
    assert selected["event_id"] == "original"
    assert store.select_alert(accepted, now, item_ref=ref) == selected
    assert store.change_alert(ref, accepted, "defer", now, not_before=now + timedelta(seconds=1))
    assert store.select_alert(accepted, now + timedelta(seconds=2), item_ref=ref) is None
    assert store.alert_status("source", "earlier") == "pending"

    # 同一 logical identity 的更新产生新 envelope；旧 Input 不得领取或结算新版本。
    store.report_alert(source_id="source", event_id="original", payload={"body": "new revision"}, observed_at=now + timedelta(seconds=3))
    assert store.select_alert(accepted, now + timedelta(seconds=4), item_ref=ref) is None
    assert store.alert_status("source", "original", mail_id=ref["mail_id"]) == "superseded"
    for action in ("deliver", "skip", "expire"):
        assert not store.change_alert(ref, accepted, action, now + timedelta(days=1))
    assert not store.change_alert(ref, accepted, "defer", now, not_before=now + timedelta(days=1))
    assert store.alert_status("source", "original") == "pending"


def test_pending_expiration_never_removes_selected_alert_recovery_identity(tmp_path):
    store = EventMailStore(tmp_path / "eventmail.db")
    store.initialize()
    now = datetime.now(timezone.utc)
    expiry = now + timedelta(seconds=1)
    for identity in ("selected", "pending"):
        store.report_alert(source_id="source", event_id=identity, payload={"body": identity}, observed_at=now, expires_at=expiry)
    ref = store.peek_alert(now)
    accepted = {"session_id": "wake:flow", "turn_id": "input"}
    selected = store.select_alert(accepted, now, item_ref=ref)
    later = expiry + timedelta(seconds=1)
    assert store.peek_alert(later) is None  # 只读 peek 不执行过期。
    assert store.alert_status("source", "pending") == "pending"
    assert store.alert_deadline(later) is None
    assert store.alert_status("source", "pending") == "expired"
    assert store.selected_alert(accepted) == selected
    assert store.select_alert({"session_id": "other", "turn_id": "other"}, later) is None
    assert store.selected_alert(accepted) == selected
    assert store.change_alert(ref, accepted, "expire", later)
    assert store.alert_status("source", "selected") == "expired"
    assert store.select_alert(accepted, later, item_ref=ref) is None
