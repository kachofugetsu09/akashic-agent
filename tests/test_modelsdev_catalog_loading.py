"""模型目录首次读取随包 models.dev 快照时不占用事件循环线程。"""
import asyncio
import threading
from pathlib import Path

import httpx
import pytest

from plugins.models import modelsdev_catalog
from plugins.models.modelsdev_catalog import ModelsDevCapabilityCatalog

_SNAPSHOT = {"openai": {"models": {"gpt-test": {"limit": {"context": 1000}, "modalities": {"input": ["text", "image"]}}}}}


def test_first_enrich_reads_snapshot_off_the_event_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    threads: list[str] = []

    def snapshot() -> object:
        threads.append(threading.current_thread().name)
        return dict(_SNAPSHOT)

    monkeypatch.setattr(modelsdev_catalog, "_read_snapshot", snapshot)
    catalog = ModelsDevCapabilityCatalog(
        tmp_path / "catalog.json", writable=False,
        transport=httpx.MockTransport(lambda _request: httpx.Response(503)),
    )

    async def run() -> str:
        _ = await catalog.enrich((), provider_id="openai")
        return threading.current_thread().name

    loop_thread = asyncio.run(run())
    assert threads and loop_thread not in threads
