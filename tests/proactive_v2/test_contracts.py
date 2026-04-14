from __future__ import annotations

from proactive_v2.contracts import (
    MAX_METRICS_KEYS,
    AlertContract,
    normalize_alert,
    normalize_content,
    normalize_context,
)


def test_normalize_alert_supports_body_alias_and_default_severity():
    event = {
        "ack_server": "fitbit",
        "event_id": "evt-1",
        "title": "心率提醒",
        "body": "静息心率偏高",
    }

    contract = normalize_alert(event)

    assert isinstance(contract, AlertContract)
    assert contract.item_id == "fitbit:evt-1"
    assert contract.content == "静息心率偏高"
    assert contract.severity == ""


def test_alert_prompt_line_hides_empty_severity():
    event = {"ack_server": "fitbit", "event_id": "evt-3", "title": "心率提醒"}
    contract = normalize_alert(event)
    line = contract.to_prompt_line(index=1)
    assert "severity=" not in line


def test_normalize_alert_metrics_trimmed_by_keys_and_value_len():
    metrics = {f"k{i}": f"value-{i}-" + ("x" * 100) for i in range(MAX_METRICS_KEYS + 2)}
    event = {"ack_server": "fitbit", "id": "evt-2", "metrics": metrics}

    contract = normalize_alert(event)
    assert contract.metrics is not None
    assert len([k for k in contract.metrics if k.startswith("k")]) == MAX_METRICS_KEYS
    assert contract.metrics["_truncated_keys"] == 2
    assert contract.metrics["k0"].endswith("...")


def test_normalize_content_source_alias_and_url_validation():
    item = {
        "id": "feed:item-1",
        "title": "新闻标题",
        "source_name": "HLTV",
        "url": "",
    }

    contract = normalize_content(item)
    assert contract.source == "HLTV"
    assert contract.has_valid_url is False


def test_normalize_context_keeps_raw_and_adds_awake_prob():
    item = {"_source": "fitbit", "available": True, "sleep_prob": 0.2, "foo": "bar"}
    contract = normalize_context(item)
    payload = contract.to_prompt_item()

    assert contract.source == "fitbit"
    assert contract.available is True
    assert payload["foo"] == "bar"
    assert payload["awake_prob"] == 0.8


def test_normalize_context_adds_local_time_fields_for_aware_timestamps():
    item = {
        "_source": "zigbee",
        "updated_at": "2026-04-14T17:58:54+00:00",
        "device": {
            "last_seen": "2026-04-14T17:58:54+00:00",
        },
    }

    payload = normalize_context(item).to_prompt_item()

    assert payload["updated_at_local"] == "2026-04-15 01:58:54 +0800"
    assert payload["device"]["last_seen_local"] == "2026-04-15 01:58:54 +0800"


def test_normalize_context_skips_local_time_for_naive_timestamp():
    item = {"_source": "zigbee", "updated_at": "2026-04-14T17:58:54"}

    payload = normalize_context(item).to_prompt_item()

    assert "updated_at_local" not in payload
