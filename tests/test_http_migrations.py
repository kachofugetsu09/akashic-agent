import json
from pathlib import Path

import httpx
import pytest

from agent.tools.fitbit import FitbitHealthSnapshotTool, FitbitSleepReportTool
from agent.tools.web_fetch import WebFetchTool
from infra.channels.qq_channel import _download_to_temp
from core.net.http import (
    HttpRequester,
    RequestBudget,
    RetryPolicy,
    SharedHttpResources,
    clear_default_shared_http_resources,
    configure_default_shared_http_resources,
    get_default_shared_http_resources,
)
from memory2.embedder import Embedder


def _build_requester(handler) -> HttpRequester:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return HttpRequester(
        client=client,
        retry_policy=RetryPolicy(max_attempts=1, base_delay_s=0.0, max_delay_s=0.0),
        default_timeout_s=1.0,
        default_budget=RequestBudget(total_timeout_s=2.0),
    )


@pytest.mark.asyncio
async def test_default_shared_http_resources_requires_explicit_configuration():
    clear_default_shared_http_resources()

    with pytest.raises(RuntimeError, match="not configured"):
        get_default_shared_http_resources()

    resources = SharedHttpResources()
    try:
        configure_default_shared_http_resources(resources)
        assert get_default_shared_http_resources() is resources
    finally:
        clear_default_shared_http_resources(resources)
        await resources.aclose()


@pytest.mark.asyncio
async def test_web_fetch_tool_uses_injected_requester():
    async def _handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept"].startswith("text/plain")
        return httpx.Response(
            200,
            request=request,
            text="hello from shared requester",
            headers={"content-type": "text/plain; charset=utf-8"},
        )

    requester = _build_requester(_handler)
    try:
        tool = WebFetchTool(requester)
        payload = json.loads(
            await tool.execute(url="https://example.com/data.txt", format="text")
        )
        assert payload["status"] == 200
        assert payload["text"] == "hello from shared requester"
    finally:
        await requester.client.aclose()


@pytest.mark.asyncio
async def test_fitbit_tools_use_injected_requester():
    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tool/fitbit_health_snapshot":
            return httpx.Response(
                200,
                request=request,
                json={
                    "heart_rate": 72,
                    "spo2": 98.1,
                    "steps": 1234,
                    "sleep_state": "sleeping",
                    "sleep_state_raw": "uncertain",
                    "sleep_prob": 0.1,
                    "sleep_24h": {"00:00-01:00": "sleeping"},
                    "data_lag_min": 15,
                    "latest_sleep_spo2_time": "07:10:00",
                    "spo2_lag_min": 65,
                    "last_updated": "2026-03-08T08:15:00+08:00",
                },
            )
        if request.url.path == "/api/data":
            return httpx.Response(
                200,
                request=request,
                json={
                    "sleep": {"state": "uncertain", "reason": "测试原因", "since": "2026-03-08 08:00:00"},
                },
            )
        assert request.url.path == "/api/sleep_report"
        assert request.url.params["days"] == "3"
        return httpx.Response(
            200,
            request=request,
            json={
                "summary": {
                    "avg_duration_min": 420,
                    "avg_efficiency": 92,
                    "avg_deep_min": 75,
                    "avg_rem_min": 95,
                    "avg_hrv_ms": 48,
                    "days_with_data": 1,
                },
                "days": [
                    {
                        "date": "2026-03-07",
                        "start_time": "23:30",
                        "end_time": "06:30",
                        "duration_min": 420,
                        "efficiency": 92,
                        "deep_min": 75,
                        "rem_min": 95,
                        "light_min": 210,
                        "wake_min": 40,
                        "hrv_ms": 48,
                    }
                ],
            },
        )

    requester = _build_requester(_handler)
    try:
        snapshot_tool = FitbitHealthSnapshotTool(
            "http://monitor.local",
            requester=requester,
        )
        snapshot = await snapshot_tool.execute()
        assert '"heart_rate": 72' in snapshot
        assert '"latest_sleep_spo2": 98.1' in snapshot
        assert '"sleep_state": "sleeping"' in snapshot
        assert "最近一次睡眠血氧：98.1%" in snapshot
        assert "睡眠状态：sleeping" in snapshot
        assert "原因：测试原因" in snapshot

        sleep_tool = FitbitSleepReportTool(
            "http://monitor.local",
            requester=requester,
        )
        report = await sleep_tool.execute(days=3)
        assert "最近 3 天" in report
        assert "7h00m" in report
    finally:
        await requester.client.aclose()


@pytest.mark.asyncio
async def test_download_to_temp_uses_injected_requester(tmp_path: Path):
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            content=b"fake-image-bytes",
            headers={"content-type": "image/png"},
        )

    requester = _build_requester(_handler)
    try:
        paths = await _download_to_temp(["https://example.com/image.png"], requester)
        assert len(paths) == 1
        path = Path(paths[0])
        assert path.suffix == ".png"
        assert path.read_bytes() == b"fake-image-bytes"
    finally:
        for raw_path in paths if "paths" in locals() else []:
            Path(raw_path).unlink(missing_ok=True)
        await requester.client.aclose()


@pytest.mark.asyncio
async def test_embedder_uses_injected_requester():
    def _handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["input"] == ["first", "second"]
        return httpx.Response(
            200,
            request=request,
            json={
                "data": [
                    {"index": 1, "embedding": [0.2, 0.3]},
                    {"index": 0, "embedding": [0.0, 0.1]},
                ]
            },
        )

    requester = _build_requester(_handler)
    try:
        embedder = Embedder(
            base_url="https://embeddings.example.com/v1",
            api_key="test-key",
            requester=requester,
        )
        vectors = await embedder.embed_batch(["first", "second"])
        assert vectors == [[0.0, 0.1], [0.2, 0.3]]
    finally:
        await requester.client.aclose()
