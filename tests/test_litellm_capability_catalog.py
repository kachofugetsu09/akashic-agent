from __future__ import annotations

import asyncio
import fcntl
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from agent.plugin_composition import (
    CapabilitySources,
    DiscoveredModel,
    ModelCapabilities,
    ModelKind,
)
from plugins.models.litellm_catalog import LiteLlmCapabilityCatalog


def _model(
    name: str = "deepseek-v4-flash-vision-exp",
    *,
    source: str = "unknown",
) -> DiscoveredModel:
    return DiscoveredModel(
        kind=ModelKind.CHAT,
        model=name,
        capabilities=ModelCapabilities(input_modalities=("text",)),
        capability_sources=CapabilitySources(input_modalities=source),
    )


def _catalog(
    cache_path: Path,
    handler: httpx.MockTransport,
    *,
    bundled: dict[str, dict[str, object]] | None = None,
) -> LiteLlmCapabilityCatalog:
    return LiteLlmCapabilityCatalog(
        cache_path,
        writable=True,
        transport=handler,
        bundled_models=bundled or {"fallback": {"supports_vision": False}},
        minimum_entries=1,
    )


@pytest.mark.asyncio
async def test_remote_catalog_recognizes_exact_new_vision_model(tmp_path: Path) -> None:
    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"deepseek-v4-flash-vision-exp": {"supports_vision": True}},
            headers={"etag": '"catalog-1"'},
        )

    cache_path = tmp_path / "litellm-capabilities.json"
    catalog = _catalog(cache_path, httpx.MockTransport(respond))
    result = await catalog.enrich((_model(),), provider_id="opencode-go")

    assert len(result) == 1
    assert result[0].capabilities.input_modalities == ("text", "image")
    assert result[0].capability_sources.input_modalities.startswith(
        "litellm-remote@sha256:"
    )
    envelope = json.loads(cache_path.read_text(encoding="utf-8"))
    assert envelope["schema_version"] == 1
    assert envelope["etag"] == '"catalog-1"'
    assert envelope["sha256"]


@pytest.mark.asyncio
async def test_remote_catalog_bounds_gzip_before_using_fallback(tmp_path: Path) -> None:
    oversized = json.dumps(
        {"target": {"supports_vision": False, "padding": "x" * (8 * 1024 * 1024)}}
    ).encode()

    async def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept-encoding"] == "gzip, identity"
        return httpx.Response(
            200,
            stream=httpx.ByteStream(gzip.compress(oversized)),
            headers={"content-encoding": "gzip"},
            request=request,
        )

    catalog = _catalog(
        tmp_path / "catalog.json",
        httpx.MockTransport(respond),
        bundled={"target": {"supports_vision": True}},
    )
    result = await catalog.enrich((_model("target"),), provider_id="opencode-go")

    assert result[0].capabilities.input_modalities == ("text", "image")
    assert result[0].capability_sources.input_modalities.startswith("litellm-wheel@")


@pytest.mark.asyncio
async def test_provider_capability_fact_is_not_overwritten(tmp_path: Path) -> None:
    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"deepseek-v4-flash-vision-exp": {"supports_vision": True}},
        )

    catalog = _catalog(
        tmp_path / "catalog.json",
        httpx.MockTransport(respond),
    )
    original = _model(source="provider")
    result = await catalog.enrich((original,), provider_id="opencode-go")

    assert result == (original,)


@pytest.mark.asyncio
async def test_refresh_failure_reuses_last_valid_remote_snapshot(
    tmp_path: Path,
) -> None:
    async def first(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"deepseek-v4-flash-vision-exp": {"supports_vision": True}},
            headers={"etag": '"catalog-1"'},
        )

    cache_path = tmp_path / "catalog.json"
    await _catalog(cache_path, httpx.MockTransport(first)).enrich(
        (_model(),), provider_id="opencode-go"
    )

    async def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    recovered = await _catalog(
        cache_path,
        httpx.MockTransport(unavailable),
    ).enrich((_model(),), provider_id="opencode-go")

    assert recovered[0].capabilities.input_modalities == ("text", "image")
    assert recovered[0].capability_sources.input_modalities.startswith(
        "litellm-remote@sha256:"
    )


@pytest.mark.asyncio
async def test_first_offline_sync_uses_bundled_facts_and_keeps_unknowns(
    tmp_path: Path,
) -> None:
    async def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    catalog = _catalog(
        tmp_path / "catalog.json",
        httpx.MockTransport(unavailable),
        bundled={"known-text-model": {"supports_vision": False}},
    )
    known, unknown = await catalog.enrich(
        (_model("known-text-model"), _model("brand-new-model")),
        provider_id="opencode-go",
    )

    assert known.capabilities.input_modalities == ("text",)
    assert known.capability_sources.input_modalities.startswith("litellm-wheel@")
    assert unknown.capability_sources.input_modalities == "unknown"


@pytest.mark.asyncio
async def test_invalid_remote_and_corrupt_cache_fall_back_to_bundled(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "catalog.json"
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "etag": '"bad"',
                "sha256": "not-the-model-digest",
                "models": {"target": {"supports_vision": True}},
            }
        ),
        encoding="utf-8",
    )

    async def invalid_json(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    catalog = _catalog(
        cache_path,
        httpx.MockTransport(invalid_json),
        bundled={"target": {"supports_vision": False}},
    )
    result = await catalog.enrich((_model("target"),), provider_id="opencode-go")

    assert result[0].capabilities.input_modalities == ("text",)
    assert result[0].capability_sources.input_modalities.startswith("litellm-wheel@")


@pytest.mark.asyncio
async def test_non_utf8_cache_does_not_block_offline_sync(tmp_path: Path) -> None:
    cache_path = tmp_path / "catalog.json"
    cache_path.write_bytes(b"\xff")

    async def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    catalog = _catalog(
        cache_path,
        httpx.MockTransport(unavailable),
        bundled={"target": {"supports_vision": True}},
    )
    result = await catalog.enrich((_model("target"),), provider_id="opencode-go")

    assert result[0].capabilities.input_modalities == ("text", "image")
    assert result[0].capability_sources.input_modalities.startswith("litellm-wheel@")


@pytest.mark.asyncio
async def test_suspiciously_small_remote_does_not_replace_last_snapshot(
    tmp_path: Path,
) -> None:
    async def first(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "target": {"supports_vision": True},
                "stable-entry": {"supports_vision": False},
            },
        )

    cache_path = tmp_path / "catalog.json"
    first_catalog = LiteLlmCapabilityCatalog(
        cache_path,
        writable=True,
        transport=httpx.MockTransport(first),
        bundled_models={
            "target": {"supports_vision": False},
            "stable-entry": {"supports_vision": False},
        },
        minimum_entries=2,
    )
    await first_catalog.enrich((_model("target"),), provider_id="opencode-go")

    async def shrunk(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"target": {"supports_vision": False}})

    second_catalog = LiteLlmCapabilityCatalog(
        cache_path,
        writable=True,
        transport=httpx.MockTransport(shrunk),
        bundled_models={
            "target": {"supports_vision": False},
            "stable-entry": {"supports_vision": False},
        },
        minimum_entries=2,
    )
    result = await second_catalog.enrich((_model("target"),), provider_id="opencode-go")

    assert result[0].capabilities.input_modalities == ("text", "image")
    assert result[0].capability_sources.input_modalities.startswith(
        "litellm-remote@sha256:"
    )


@pytest.mark.asyncio
async def test_catalog_matching_is_exact_and_never_adds_provider_models(
    tmp_path: Path,
) -> None:
    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "deepseek-v4-flash-vision-exp": {"supports_vision": True},
                "another-model": {"supports_vision": True},
            },
        )

    catalog = _catalog(tmp_path / "catalog.json", httpx.MockTransport(respond))
    result = await catalog.enrich(
        (_model("DeepSeek-V4-Flash-Vision-exp"),),
        provider_id="opencode-go",
    )

    assert len(result) == 1
    assert result[0].capability_sources.input_modalities == "unknown"


@pytest.mark.asyncio
async def test_each_sync_rechecks_and_can_upgrade_existing_capabilities(
    tmp_path: Path,
) -> None:
    requests = 0

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            200,
            json={
                "deepseek-v4-flash-vision-exp": {
                    "supports_vision": requests > 1,
                }
            },
            headers={"etag": f'"catalog-{requests}"'},
            request=request,
        )

    catalog = _catalog(
        tmp_path / "catalog.json",
        httpx.MockTransport(respond),
    )
    first = await catalog.enrich((_model(),), provider_id="opencode-go")
    second = await catalog.enrich((_model(),), provider_id="opencode-go")

    assert requests == 2
    assert first[0].capabilities.input_modalities == ("text",)
    assert second[0].capabilities.input_modalities == ("text", "image")


@pytest.mark.asyncio
async def test_concurrent_catalog_instances_publish_in_refresh_order(
    tmp_path: Path,
) -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    requests = 0

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            first_started.set()
            await release_first.wait()
            return httpx.Response(
                200,
                json={"target": {"supports_vision": False}},
                headers={"etag": '"catalog-1"'},
                request=request,
            )
        assert request.headers["if-none-match"] == '"catalog-1"'
        return httpx.Response(
            200,
            json={"target": {"supports_vision": True}},
            headers={"etag": '"catalog-2"'},
            request=request,
        )

    cache_path = tmp_path / "catalog.json"
    transport = httpx.MockTransport(respond)
    first_catalog = _catalog(cache_path, transport)
    second_catalog = _catalog(cache_path, transport)
    first_task = asyncio.create_task(
        first_catalog.enrich((_model("target"),), provider_id="opencode-go")
    )
    await first_started.wait()
    second_task = asyncio.create_task(
        second_catalog.enrich((_model("target"),), provider_id="opencode-go")
    )
    await asyncio.sleep(0)
    release_first.set()
    first, second = await asyncio.gather(first_task, second_task)

    assert first[0].capabilities.input_modalities == ("text",)
    assert second[0].capabilities.input_modalities == ("text", "image")
    envelope = json.loads(cache_path.read_text(encoding="utf-8"))
    assert envelope["etag"] == '"catalog-2"'
    assert envelope["models"]["target"]["supports_vision"] is True


@pytest.mark.asyncio
async def test_cancel_while_waiting_for_catalog_lock_returns_immediately(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "catalog.json"
    lock_fd = os.open(f"{cache_path}.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)

    async def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    task = asyncio.create_task(
        _catalog(cache_path, httpx.MockTransport(unavailable)).enrich(
            (_model(),), provider_id="opencode-go"
        )
    )
    await asyncio.sleep(0)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.2)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


@pytest.mark.asyncio
async def test_catalog_lock_blocks_refresh_in_another_process(tmp_path: Path) -> None:
    cache_path = tmp_path / "catalog.json"
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl, os, sys\n"
                "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)\n"
                "fcntl.flock(fd, fcntl.LOCK_EX)\n"
                "print('locked', flush=True)\n"
                "sys.stdin.read(1)\n"
                "fcntl.flock(fd, fcntl.LOCK_UN)\n"
                "os.close(fd)\n"
            ),
            f"{cache_path}.lock",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert child.stdout is not None
    assert await asyncio.to_thread(child.stdout.readline) == "locked\n"
    request_started = asyncio.Event()

    async def respond(_request: httpx.Request) -> httpx.Response:
        request_started.set()
        return httpx.Response(200, json={"target": {"supports_vision": True}})

    task = asyncio.create_task(
        _catalog(cache_path, httpx.MockTransport(respond)).enrich(
            (_model("target"),), provider_id="opencode-go"
        )
    )
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                asyncio.shield(request_started.wait()),
                timeout=0.1,
            )
        assert child.stdin is not None
        child.stdin.write("x")
        child.stdin.flush()
        result = await asyncio.wait_for(task, timeout=2.0)
        assert result[0].capabilities.input_modalities == ("text", "image")
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if child.poll() is None:
            child.terminate()
        await asyncio.to_thread(child.wait)
        if child.stdin is not None:
            child.stdin.close()
        if child.stdout is not None:
            child.stdout.close()


@pytest.mark.asyncio
async def test_deep_remote_json_falls_back_instead_of_failing_sync(
    tmp_path: Path,
) -> None:
    deep_json = b'{"target":{"nested":' + b"[" * 10_000 + b"0" + b"]" * 10_000 + b"}}"

    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=deep_json)

    catalog = _catalog(
        tmp_path / "catalog.json",
        httpx.MockTransport(respond),
        bundled={"target": {"supports_vision": True}},
    )
    result = await catalog.enrich((_model("target"),), provider_id="opencode-go")

    assert result[0].capabilities.input_modalities == ("text", "image")
    assert result[0].capability_sources.input_modalities.startswith("litellm-wheel@")


@pytest.mark.asyncio
async def test_deep_cache_json_is_ignored_during_offline_sync(tmp_path: Path) -> None:
    cache_path = tmp_path / "catalog.json"
    cache_path.write_bytes(b"[" * 10_000 + b"0" + b"]" * 10_000)

    async def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    catalog = _catalog(
        cache_path,
        httpx.MockTransport(unavailable),
        bundled={"target": {"supports_vision": True}},
    )
    result = await catalog.enrich((_model("target"),), provider_id="opencode-go")

    assert result[0].capabilities.input_modalities == ("text", "image")
    assert result[0].capability_sources.input_modalities.startswith("litellm-wheel@")


@pytest.mark.asyncio
async def test_unimplemented_modalities_are_not_published(tmp_path: Path) -> None:
    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"target": {"supported_modalities": ["text", "audio", "video"]}},
        )

    catalog = _catalog(tmp_path / "catalog.json", httpx.MockTransport(respond))
    result = await catalog.enrich((_model("target"),), provider_id="opencode-go")

    assert result[0].capabilities.input_modalities == ("text",)
    assert result[0].capability_sources.input_modalities.startswith(
        "litellm-remote@sha256:"
    )
