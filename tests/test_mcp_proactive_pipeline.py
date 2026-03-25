"""
tests/test_mcp_proactive_pipeline.py

端到端测试：mock fitbit-monitor HTTP → fitbit-mcp → mcp_sources → GenericAlertEvent

测试事件标注为 TEST_EVENT，不会进入真实 fitbit-monitor。
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest

# ── 测试用假事件 ──────────────────────────────────────────────────────────────

TEST_EVENT = {
    "id": "TEST-0001",
    "type": "test_alert",
    "severity": "high",
    "message": "[TEST] 这是一条测试告警事件，请忽略",
    "created_at": "2099-01-01 00:00",
    "suggested_tone": "测试专用，无需关心语气",
    "metrics": {"test_value": 42, "test_label": "unit_test"},
}

MOCK_AGENT_RESPONSE = {
    "sleep": {"state": "awake", "prob": 0.03, "prob_source": "ml", "data_lag_min": 5},
    "health_events": [TEST_EVENT],
    "last_updated": "00:00:00",
}


# ── Mock HTTP Server ──────────────────────────────────────────────────────────

class _MockFitbitHandler(BaseHTTPRequestHandler):
    acked_ids: list[str] = []

    def do_GET(self):
        if self.path == "/api/agent":
            body = json.dumps(MOCK_AGENT_RESPONSE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path.startswith("/api/agent/acknowledge/"):
            event_id = self.path.split("/")[-1]
            _MockFitbitHandler.acked_ids.append(event_id)
            body = json.dumps({"acknowledged": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass  # 静默 HTTP 日志


def _start_mock_server() -> tuple[HTTPServer, int]:
    server = HTTPServer(("127.0.0.1", 0), _MockFitbitHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestMcpProactivePipeline:

    def setup_method(self):
        _MockFitbitHandler.acked_ids = []

    @pytest.mark.asyncio
    async def test_fetch_alert_events_returns_test_event(self):
        """fitbit-mcp → mcp_sources.fetch_alert_events_async() 能正确拿到测试告警。"""
        server, port = _start_mock_server()
        try:
            # 把 fitbit-mcp 指向 mock server
            env_patch = {
                "FITBIT_MONITOR_HOST": "127.0.0.1",
                "FITBIT_MONITOR_PORT": str(port),
            }
            # 临时替换 mcp_servers.json 里的 command env
            from proactive_v2 import mcp_sources

            original_get_server_cfg = mcp_sources._get_server_cfg

            def patched_cfg(server_name):
                cfg = original_get_server_cfg(server_name)
                if cfg and server_name == "fitbit":
                    cfg = dict(cfg)
                    cfg["env"] = env_patch
                return cfg

            with patch.object(mcp_sources, "_get_server_cfg", side_effect=patched_cfg):
                pool = mcp_sources.McpClientPool()
                await pool.connect_all()
                events = await mcp_sources.fetch_alert_events_async(pool)
                await pool.disconnect_all()

            assert len(events) == 1, f"期望 1 条事件，实际: {events}"
            evt = events[0]
            assert evt["event_id"] == "TEST-0001"
            assert evt["kind"] == "alert"
            assert evt["ack_server"] == "fitbit"
            assert evt["source_name"] == "fitbit"
            assert evt["source_type"] == "health_event"
            assert evt["severity"] == "high"
            assert "[TEST]" in evt["content"]
            assert evt.get("metrics") == {"test_value": 42, "test_label": "unit_test"}
        finally:
            server.shutdown()

    def test_generic_alert_event_from_mcp_payload(self):
        """GenericAlertEvent.from_mcp_payload() 正确解析标准 schema。"""
        from proactive_v2.event import GenericAlertEvent

        # 模拟 fetch_alert_events() 返回的 dict
        payload = {
            "event_id": "TEST-0001",
            "kind": "alert",
            "source_type": "health_event",
            "source_name": "fitbit",
            "title": "test_alert",
            "content": "[TEST] 这是一条测试告警事件，请忽略",
            "severity": "high",
            "published_at": "2099-01-01T00:00:00",
        }

        evt = GenericAlertEvent.from_mcp_payload(payload)

        assert evt.kind == "alert"
        assert evt.event_id == "TEST-0001"
        assert evt.ack_id == "TEST-0001"
        assert evt._ack_server is None
        assert evt.source_name == "fitbit"
        assert evt.is_urgent() is True   # severity == "high"
        assert "[TEST]" in evt.content

        sig = evt.to_signal_dict()
        assert sig["kind"] == "alert"
        assert sig["severity"] == "high"
        assert sig["message"] == evt.content   # _extra_signal_fields backward compat

    @pytest.mark.asyncio
    async def test_acknowledge_events_calls_ack_endpoint(self):
        """acknowledge_events_async() 正确把 ack_id 送回 fitbit-monitor。"""
        server, port = _start_mock_server()
        try:
            from proactive_v2.event import GenericAlertEvent
            from proactive_v2 import mcp_sources

            payload = {
                "event_id": "TEST-0001",
                "kind": "alert",
                "source_type": "health_event",
                "source_name": "fitbit-display-name",
                "ack_server": "fitbit",
                "title": "test_alert",
                "content": "[TEST] 测试 ack",
                "severity": "normal",
            }
            evt = GenericAlertEvent.from_mcp_payload(payload)

            env_patch = {
                "FITBIT_MONITOR_HOST": "127.0.0.1",
                "FITBIT_MONITOR_PORT": str(port),
            }
            original_get_server_cfg = mcp_sources._get_server_cfg

            def patched_cfg(server_name):
                cfg = original_get_server_cfg(server_name)
                if cfg and server_name == "fitbit":
                    cfg = dict(cfg)
                    cfg["env"] = env_patch
                return cfg

            with patch.object(mcp_sources, "_get_server_cfg", side_effect=patched_cfg):
                pool = mcp_sources.McpClientPool()
                await pool.connect_all()
                await mcp_sources.acknowledge_events_async(pool, [evt])
                await pool.disconnect_all()

            assert "TEST-0001" in _MockFitbitHandler.acked_ids, (
                f"期望 TEST-0001 被 ack，实际 acked: {_MockFitbitHandler.acked_ids}"
            )
        finally:
            server.shutdown()
