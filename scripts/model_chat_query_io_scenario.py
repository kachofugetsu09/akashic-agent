"""Check real model UI queries under controlled SQLite reads in an installed runtime."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import closing, contextmanager
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ReadBarrier:
    """Pause actual SQLite statements; queries still read their real saved rows."""

    def __init__(self, *, cancel: bool = False):
        self.loop = asyncio.get_running_loop()
        self.cancel = cancel
        self.entered = asyncio.Event()
        self.release = threading.Event()
        self.reads = []

    @contextmanager
    def install(self):
        """Attach a trace only to model-store read connections, then restore it."""
        original = sqlite3.connect

        def connect(database, *args, **kwargs):
            connection = original(database, *args, **kwargs)
            if "model-registry.sqlite3?mode=ro" in str(database):
                used = False

                def trace(statement):
                    nonlocal used
                    if used or not statement.lstrip().upper().startswith("SELECT"):
                        return
                    used = True
                    heartbeat = threading.Event()
                    started = time.perf_counter()
                    observation = {"on_loop": threading.get_ident() == self.loop_thread}
                    self.reads.append(observation)

                    def tick():
                        observation["loop_delay_ms"] = (time.perf_counter() - started) * 1000
                        heartbeat.set()
                        self.entered.set()

                    self.loop.call_soon_threadsafe(tick)
                    # The bound ends a blocked baseline; success requires the loop receipt.
                    observation["heartbeat_before_read"] = heartbeat.wait(1)
                    if self.cancel:
                        observation["released"] = self.release.wait(5)

                connection.set_trace_callback(trace)
            return connection

        self.loop_thread = threading.get_ident()
        sqlite3.connect = connect
        try:
            yield
        finally:
            self.release.set()
            sqlite3.connect = original


def saved_rows(workspace: Path) -> dict[str, list]:
    """Keep complete configuration and call rows, plus the seeded session metadata."""
    with closing(sqlite3.connect(workspace / "model-registry.sqlite3")) as db:
        result = {table: db.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
                  for table in ("model_connections", "model_definitions", "model_role_bindings", "model_calls")}
    with closing(sqlite3.connect(workspace / "sessions.db")) as db:
        result["session"] = db.execute("SELECT metadata FROM sessions WHERE key=?", ("programmatic:bundle-0",)).fetchall()
    return result


async def run(distribution: Path, directory: Path, baseline: bool) -> dict:
    """Install plugins, make a CLI reply, and query through the actual UI provider."""
    from agent.config import Config
    from bootstrap.app import AppRuntime
    from bootstrap.init_workspace import init_workspace
    from plugins.ui.contract import PLUGIN_UI, PluginUiRpcInvalidRequest
    from scripts.bundle_modes_scenario import headless_reply
    from scripts.install_plugin_distribution import ensure_bundle

    # 1. Build fresh runtime state and one real local-HTTP model call.
    workspace, home, config = directory / "workspace", directory / "home", directory / "config.toml"
    os.environ.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(home), AKASHIC_PLUGIN_DISTRIBUTION=str(distribution),
                      AKASHIC_PLUGIN_BUNDLE="base", AKASHIC_EXTRA_PLUGIN_DIRS="", AKASHIC_EXECUTION_MODE="local",
                      PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "sdk/python/src"))))
    os.environ.pop("AKASHIC_WEB_PORT", None)
    config.write_text("[runtime]\n")
    init_workspace(config_path=config, workspace=workspace)
    ensure_bundle(distribution, distribution / "bundles/base.toml", workspace=workspace,
                  plugins_home=home, config_path=config, receipt_path=workspace / "runtime/distribution-install.json")
    app = AppRuntime(Config.load(config, workspace=workspace), workspace)
    try:
        await app.start()
        await headless_reply(app, workspace, config)
        # Seed a saved preference in this disposable fixture, without mocking its reader.
        with closing(sqlite3.connect(workspace / "sessions.db")) as db:
            row = db.execute("SELECT metadata FROM sessions WHERE key=?", ("programmatic:bundle-0",)).fetchone()
            metadata = {} if row[0] is None else json.loads(row[0])
            metadata["model_selection"] = {"schema_version": 1, "model_ref": "local", "reasoning_effort": ""}
            db.execute("UPDATE sessions SET metadata=? WHERE key=?", (json.dumps(metadata), "programmatic:bundle-0"))
            db.commit()
        root = app.core.plugin_manager.live_root
        ui = root.service_value(PLUGIN_UI)
        catalog = await ui.catalog()
        item = next(item for item in catalog["items"] if item["id"] == "models@release")

        async def query(method, payload):
            return await ui.query(item["id"], item["revision"], method, payload,
                                  session_id="programmatic:bundle-0", turn_id=None)

        expected_models = await query("chat.models", {})
        expected_call = await query("call.latest", {})
        assert expected_models["sessionOverride"] == "local" and expected_models["runtimes"]
        assert expected_call["call"] is not None
        call_id = expected_call["call"]["call_record_id"]
        expected_stats = await query("call.stats", {"call_id": call_id})
        before = saved_rows(workspace)

        # 2. Every catalog, preference, icon and stats read must leave the loop available.
        observations = {}
        for method, payload, expected in (("chat.models", {}, expected_models),
                                          ("call.stats", {"call_id": call_id}, expected_stats)):
            barrier = ReadBarrier()
            with barrier.install():
                assert await query(method, payload) == expected
            assert barrier.reads
            if baseline:
                assert any(read["on_loop"] and not read["heartbeat_before_read"] for read in barrier.reads)
            else:
                assert all(not read["on_loop"] and read["heartbeat_before_read"] for read in barrier.reads)
            observations[method] = barrier.reads
        try:
            await query("call.stats", {"call_id": "absent-call"})
        except PluginUiRpcInvalidRequest:
            pass
        else:
            raise AssertionError("Missing calls must remain invalid requests")
        assert saved_rows(workspace) == before

        # 3. Caller cancellation may return, but provider shutdown must drain the read.
        if not baseline:
            barrier = ReadBarrier(cancel=True)
            with barrier.install():
                reading = asyncio.create_task(query("chat.models", {}))
                closing_ui = None
                try:
                    await asyncio.wait_for(barrier.entered.wait(), 3)
                    reading.cancel()
                    try:
                        await reading
                    except asyncio.CancelledError:
                        pass
                    closing_ui = asyncio.create_task(ui.aclose())
                    tick = asyncio.Event()
                    asyncio.get_running_loop().call_soon(tick.set)
                    await tick.wait()
                    assert not closing_ui.done(), "UI owner closed before its storage read drained"
                finally:
                    barrier.release.set()
                    await asyncio.gather(reading, return_exceptions=True)
                    if closing_ui is not None:
                        await closing_ui
                assert all(read.get("released") for read in barrier.reads)
            assert saved_rows(workspace) == before
        return {"baseline": baseline, "reads": observations, "saved_rows_unchanged": True,
                "cancel_drained": not baseline, "missing_call_rejected": True}
    finally:
        await app.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("distribution", type=Path)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    directory = Path(tempfile.mkdtemp(prefix="akashic-model-query-io-"))
    result = asyncio.run(run(args.distribution.resolve(), directory, args.baseline))
    print(json.dumps({**result, "evidence": str(directory)}))
