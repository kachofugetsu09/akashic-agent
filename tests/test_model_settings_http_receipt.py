from __future__ import annotations

import json

from agent.plugin_composition import SettingsReceipt
from agent.plugin_composition.model_settings_http import _receipt_payload


def test_auth_challenge_is_plain_json_at_http_boundary() -> None:
    receipt = SettingsReceipt(
        revision=0,
        status="pending",
        attempt_id="attempt-1",
        challenge={"steps": [{"name": "wait"}]},
    )

    payload = _receipt_payload(receipt)

    assert payload["challenge"] == {"steps": [{"name": "wait"}]}
    assert json.loads(json.dumps(payload))["challenge"] == payload["challenge"]
