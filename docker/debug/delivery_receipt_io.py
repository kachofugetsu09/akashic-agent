"""Delivery 慢事务中的实际效果、取消、撤回和 guard 顺序。"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agent.plugin_composition.tasks import Tasks
from plugins.eventmail.plugin import _StoreIO, _WakeServices, _AlertSourceServices
from plugins.delivery.api import Receipt
from plugins.delivery.execution import Deliveries
from plugins.delivery.records import DeliveryRecords
from plugins.eventmail.store import EventMailStore
from plugins.ledger.log import MessageLog, OwnerTransaction
from plugins.ledger.contract import ContentPart, ContentReferences, Output


async def check(directory: Path, phase: str, cancel: bool) -> dict:
    """真实落盘与本地 sender 文件效果；屏障不替换任何提交或回执。"""
    log, tasks = MessageLog(directory / "sessions.db"), Tasks()
    loop = asyncio.get_running_loop()
    reached, release = asyncio.Event(), threading.Event()
    stamps, effects = [], []
    records = DeliveryRecords(log.owner("delivery"), "scenario")
    original_save = OwnerTransaction.save
    domain = EventMailStore(directory / "eventmail.db")
    domain.initialize()
    now = datetime.now(UTC)
    domain.report_alert(source_id="sensor", event_id="event", payload={"text": "first"}, observed_at=now)
    selected = domain.select_alert({"session_id": "s", "turn_id": "turn"}, now)
    assert selected is not None
    io = _StoreIO()
    wake = _WakeServices(domain, io)
    alerts = _AlertSourceServices(domain, io, lambda: None).bind("sensor")
    sending, release_sender = asyncio.Event(), asyncio.Event()

    def guard():
        return wake.alert_start(selected, None, lambda: now)

    class Sender:
        idempotent = False
        async def send(self, key, address, message):
            (directory / "effect.txt").write_text(json.dumps([key, address, message.message_id]))
            effects.append(message.message_id)
            if phase == "guard":
                sending.set()
                await release_sender.wait()
            if phase == "failed":
                raise RuntimeError("local send failed after effect")
            return Receipt(status="delivered", provider_ids=("local",))
        async def query(self, key, address):
            return None

    @asynccontextmanager
    async def open_sender(_binding):
        yield Sender()

    delivery = Deliveries(records, log.catalog(), tasks, open_sender, task_key="delivery")
    log.save_binding("local", {"scenario": True})
    writer = log.writer("s", author="scenario", source="scenario", body_types=(Output,),
                        content={"text": lambda _: ContentReferences()})
    message = writer.append("message", Output((ContentPart("text", "original body"),), "complete"))
    await delivery.prepare_async(log.reader("s"), message, ({"name": "local", "binding_id": "local", "address": "one"},))
    if phase == "retry":
        await delivery.cancel_prepared("message", "local", "fixture rejected")
    target_phase = {"cancel_prepared": "rejected", "retry": "prepared", "guard": "started", "guard_reject": "rejected"}.get(phase, phase)

    def save(tx, key, value, **kwargs):
        row = original_save(tx, key, value, **kwargs)
        if key.startswith("delivery:") and value.get("phase") == target_phase and not stamps:
            stamps.append(time.perf_counter())
            loop.call_soon_threadsafe(reached.set)
            release.wait(1.0)
        return row

    async def update():
        return await alerts.report(event_id="event", payload={"text": "new version"},
                                   observed_at=now + timedelta(seconds=1))

    if phase == "guard_reject":
        await update()
    job = updating = None
    try:
        with patch.object(OwnerTransaction, "save", save):
            if phase == "cancel_prepared":
                operation = delivery.cancel_prepared("message", "local", "explicit withdrawal")
            elif phase == "retry":
                operation = delivery.retry("message", "local")
            else:
                operation = delivery.send("message", "local", **({"start_guard": guard} if phase in {"guard", "guard_reject"} else {}))
            job = asyncio.create_task(operation)
            await asyncio.wait_for(reached.wait(), 3)
            lag = time.perf_counter() - stamps[0]
            baseline = os.environ.get("BASELINE") == "1"
            if not baseline:
                assert lag < 0.2, lag
                assert not job.done()
                if phase == "guard":
                    updating = asyncio.create_task(update())
                    assert io.lock.locked()
                    assert not updating.done()
                if cancel:
                    job.cancel()
                    job.cancel()
                    await loop.run_in_executor(None, lambda: None)
                    assert not job.done()
            release.set()
            if phase == "guard":
                await asyncio.wait_for(sending.wait(), 3)
                await asyncio.wait_for(updating, 3)
                # started 已保存，版本更新无需等待网络发送结束。
                assert not job.done()
                release_sender.set()
            try:
                await job
            except asyncio.CancelledError:
                assert cancel
            except RuntimeError as error:
                assert phase == "failed" and str(error) == "local send failed after effect"
        saved = records.read("message", "local")[1]
        if cancel and not baseline:
            expected = {"started": "failed", "delivered": "delivered", "failed": "failed",
                        "cancel_prepared": "rejected", "retry": "prepared"}[phase]
            expected_effects = int(phase in {"delivered", "failed"})
        else:
            expected = {"failed": "failed", "cancel_prepared": "rejected", "guard_reject": "rejected"}.get(phase, "delivered")
            expected_effects = int(phase not in {"cancel_prepared", "guard_reject"})
        assert saved.phase == expected, (phase, saved.phase)
        assert len(effects) == expected_effects
        assert log.reader("s").get("message") == message
        with sqlite3.connect(directory / "sessions.db") as db:
            assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        confirmations = [key for key, _ in log.owner("delivery").list() if key.startswith("confirmed-message:")]
        assert len(confirmations) == int(expected == "delivered")
        return {"phase": phase, "cancel": cancel, "loop_delay_seconds": lag,
                "saved_phase": saved.phase, "effects": len(effects), "confirmation_count": len(confirmations)}
    finally:
        release.set()
        release_sender.set()
        for pending in (job, updating):
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
        alerts.close()
        await tasks.close()
        writer.expire()
        log.close()


async def main(directory):
    results = []
    cases = [(phase, cancel) for phase in ("started", "delivered", "failed", "cancel_prepared", "retry") for cancel in (False, True)]
    if os.environ.get("BASELINE") != "1":
        cases.extend((("guard", False), ("guard_reject", False)))
    for phase, cancel in cases:
        target = directory / f"{phase}-{cancel}"
        target.mkdir()
        results.append(await check(target, phase, cancel))
    return results


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-delivery-receipt-") as directory:
        results = asyncio.run(main(Path(directory)))
    print(json.dumps({"cases": results, "cleanup": "passed"}, indent=2))
