"""Manual acceptance: real SQLite owners and Chat ASGI; no model/network/production data."""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import cast
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile

os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CUA_TELEMETRY_ENABLED"] = "false"
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.source = args.source.resolve()
if args.output.exists():
    parser.error("output must be a new file")
sys.path.insert(0, str(args.source))

import httpx
from agent.plugin_contracts.message import ContentPart, ContentReferences, Input
from plugins.akashic_clients.chat_api import create_chat_app
from plugins.akashic_clients.navigation import NavigationPreferences, PinReference
from plugins.akashic_clients.web_chat import WebChatChannel
from plugins.ui.contract import PluginUiStaleRevision, PluginUiRpcExecutionError
from plugins.projects.plugin import Projects
from session.log import MessageLog, SessionAttributes

checks: list[str] = []
def check(condition: bool, description: str) -> None:
    assert condition, description
    checks.append(description)


class ProjectsQuery:
    """HTTP provider fixture delegates the actual project.list to its real owner."""
    def __init__(self, projects: Projects):
        self.projects = projects
        self.available = True
        self.failure = None
        self.wait = None
        self.entered = None

    async def catalog(self):
        return {"items": [{"id": "projects", "revision": "fixture"}] if self.available else []}

    async def query(self, plugin_id, revision, method, payload, **kwargs):
        if self.wait is not None:
            self.entered.set()
            await self.wait.wait()
        if self.failure is not None:
            raise self.failure
        assert (plugin_id, revision, method, payload) == ("projects", "fixture", "project.list", {})
        return {"items": self.projects.list()}


def protected(path: Path):
    with sqlite3.connect(path) as db:
        return {table: db.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                for table in ("sessions", "messages", "bindings")}


async def run(root: Path):
    path = root / "sessions.db"
    log = MessageLog(path)
    projects = Projects(log.owner("plugin:projects"))
    project_id = "p_" + "1" * 32
    archived_id = "p_" + "2" * 32
    projects.create(project_id, "Real project")
    projects.create(archived_id, "Archived project")
    projects.update(archived_id, archived=True)
    keys = ["akashic:old", "akashic:ordinary", "akashic:other-dimension", "akashic:project", "akashic:orphan", "akashic:archived", "akashic:internal", "telegram:foreign"]
    attrs = [SessionAttributes(), SessionAttributes(), SessionAttributes(scope=(("computer", "desk"),)),
             SessionAttributes(scope=(("project", project_id),)), SessionAttributes(scope=(("project", "missing"),)),
             SessionAttributes(scope=(("project", archived_id),)), SessionAttributes(visibility="internal"), SessionAttributes()]
    for key, attributes in zip(keys, attrs):
        log.ensure_session(key, attributes)
        log.writer(key, author="user", source="conversation", body_types=(Input,),
                   content={"text": lambda _: ContentReferences()}).append(key, Input((ContentPart("text", "Title " + key),)))
    for index in range(65):
        log.ensure_session(f"akashic:recent-{index}", SessionAttributes())
    before = protected(path)
    projects_before = log.owner("plugin:projects").list()
    store = log.owner("plugin:akashic_clients")
    navigation = NavigationPreferences(lambda: store)
    provider = ProjectsQuery(projects)
    app = create_chat_app(workspace=root, channel=WebChatChannel("akashic"), messages=log.catalog(),
                          navigation=navigation, plugin_ui_provider=provider)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://scenario") as client:
        async def update(kind, id, pinned=True):
            return await client.post("/api/chat/navigation/pins", json={"kind": kind, "id": id, "pinned": pinned})
        response = await client.get("/api/chat/navigation/pins")
        check(response.json() == {"pins": [], "sessions": []}, "empty preferences do not create an owner record")
        check(store.list() == (), "read-only empty GET writes no record")
        check(response.headers["cache-control"] == "no-store", "pins are not cached")
        for key in keys[3:]:
            check((await update("session", key)).status_code in (400, 422), "reject ineligible session " + key)
        for invalid in ["default", "fake", "p_" + "3" * 32, archived_id]:
            check((await update("project", invalid)).status_code in (400, 422), "reject invalid/unavailable project " + invalid)
        check((await update("session", keys[0])).status_code == 200, "pin old session")
        previous = navigation.read()
        for failure, status in ((PluginUiStaleRevision("changed"), 409), (PluginUiRpcExecutionError("failed"), 502)):
            provider.failure = failure
            check((await update("project", project_id)).status_code == status and navigation.read() == previous,
                  "project provider " + type(failure).__name__ + " preserves existing preferences")
        provider.failure = None
        provider.wait, provider.entered = asyncio.Event(), asyncio.Event()
        pending_pin = asyncio.create_task(update("project", project_id))
        await provider.entered.wait()
        pending_pin.cancel()
        try:
            await pending_pin
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancelled pin completed")
        provider.wait = None
        check(navigation.read() == previous, "cancel during project validation writes no preference")
        check((await update("project", project_id)).status_code == 200, "pin real project")
        check((await update("session", keys[2])).status_code == 200, "non-project scope dimension remains eligible")
        expected = [{"kind": "session", "id": keys[0]}, {"kind": "project", "id": project_id}, {"kind": "session", "id": keys[2]}]
        version = store.read("navigation:pins").version
        check((await update("session", keys[0])).json()["pins"] == expected, "retry preserves mixed stable order")
        check(store.read("navigation:pins").version == version, "retry has no redundant write")
        recent = (await client.get("/api/chat/sessions")).json()["items"]
        check(keys[0] not in [row["key"] for row in recent], "old target lies beyond recent 50")
        result = (await client.get("/api/chat/navigation/pins")).json()
        check(result["sessions"][0]["first_message_content"] == "Title " + keys[0], "old pin resolves by actual first Message outside recent page")
        provider.available = False
        check((await update("project", project_id)).status_code == 200, "committed project pin replay survives provider unavailable")
        check((await client.get("/api/chat/navigation/pins")).json()["pins"] == expected, "provider absence does not discard refs")
        check((await update("project", project_id, False)).status_code == 200, "unpin works while project owner unavailable")
        provider.available = True
        check((await update("project", project_id)).json()["pins"][-1] == expected[1], "repin appends at end")
        check((await update("session", keys[0], False)).status_code == 200, "explicit unpin")
        check(protected(path) == before and log.owner("plugin:projects").list() == projects_before, "API changes no Session, Message, binding or Project fact")
        unknown = PinReference(kind="session", id="akashic:temporarily-missing")
        navigation.update(unknown, pinned=True)
        result = (await client.get("/api/chat/navigation/pins")).json()
        check(unknown.model_dump() in result["pins"] and unknown.id not in [row["key"] for row in result["sessions"]], "unresolved refs survive read without fabricated session")
        check((await update("session", unknown.id, False)).status_code == 200, "missing reference can be explicitly unpinned")
        check((await client.post("/api/chat/navigation/pins", json={"kind":"session","id":keys[0],"pinned":"true"})).status_code == 422, "strict boolean boundary")
        check((await client.post("/api/chat/navigation/pins", json={"kind":"session","id":keys[0],"pinned":True,"title":"copy"})).status_code == 422, "reject redundant copied data")
    with ThreadPoolExecutor(max_workers=8) as executor:
        refs = [PinReference(kind="session", id=f"akashic:concurrent-{n}") for n in range(24)]
        list(executor.map(lambda ref: navigation.update(ref, pinned=True), refs))
    check(all(ref in navigation.read() for ref in refs), "concurrent target updates preserve each other")
    previous = navigation.read()
    log._connection.set_authorizer(lambda action, table, *rest: sqlite3.SQLITE_DENY
                                   if action in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE) and table == "owner_records" else sqlite3.SQLITE_OK)
    try:
        navigation.update(PinReference(kind="session", id="akashic:failed-write"), pinned=True)
    except sqlite3.DatabaseError:
        pass
    else:
        raise AssertionError("denied SQLite write succeeded")
    finally:
        log._connection.set_authorizer(None)
    check(navigation.read() == previous, "actual SQLite denied write leaves prior preference list intact")
    persisted = navigation.read()
    check(len(persisted) == len(set(persisted)), "concurrent list has no duplicate refs")
    log.close()
    log = MessageLog(path)
    restored = NavigationPreferences(lambda: log.owner("plugin:akashic_clients"))
    check(restored.read() == persisted, "database reopen preserves exact mixed order")
    check(protected(path) == before, "reopen preserves protected facts")
    with sqlite3.connect(path) as db:
        check(db.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "SQLite integrity")
        value = json.loads(db.execute("SELECT value FROM owner_records WHERE owner='plugin:akashic_clients'").fetchone()[0])
        check(set(value) == {"pins"} and all(set(ref) == {"kind", "id"} for ref in value["pins"]), "one ordered list stores only typed references")
    log.close()


async def composition(root_path: Path):
    from agent.plugin_composition import CompositionRoot, PluginRuntime, CHANNELS
    from plugins.ui.contract import UI_SLOTS
    from agent.plugin_composition.channels import ChannelFactoryContext
    from agent.plugin_composition.messages import MESSAGE_CATALOG, OWNER_STATE, SESSION_ADMISSION, OwnerState, SessionAdmission
    from plugins.ui.contract import PLUGIN_UI
    from plugins.ui.queries import LivePluginUiProvider
    from plugins.akashic_clients import plugin
    from plugins.projects import plugin as project_plugin
    from plugins.ui.plugin_ui import PluginUiSlots

    root = CompositionRoot("navigation-scenario")
    log = MessageLog(root_path / "composition.db")
    declarations = []
    class Registrar:
        async def register(self, ctx, definition):
            declarations.append(definition)
    async def services(ctx):
        values = {CHANNELS: Registrar(), OWNER_STATE: OwnerState(log), MESSAGE_CATALOG: log.catalog(),
                  SESSION_ADMISSION: SessionAdmission(log)}
        for key in plugin.inject:
            if key != PLUGIN_UI:
                await ctx.provide(key, values.get(key, object()))
        await ctx.provide(SESSION_ADMISSION, values[SESSION_ADMISSION])
    async def ui(ctx):
        slots = PluginUiSlots(ctx)
        await ctx.provide(UI_SLOTS, slots)
        await ctx.provide(PLUGIN_UI, LivePluginUiProvider(ctx, slots))
    await root.mount(ui, name="ui")
    provider = cast(LivePluginUiProvider, root.context.require(PLUGIN_UI))
    await root.mount(services, name="storage")
    await root.mount(project_plugin.apply, name="projects", inject=project_plugin.inject,
                     runtime=PluginRuntime("projects", "scenario", args.source / "plugins/projects", root_path, root_path, {}))
    contexts = []
    async def client(ctx):
        contexts.append(ctx)
        await plugin.apply(ctx)
    owner = await root.mount(client, name="akashic_clients", inject=plugin.inject,
                             runtime=PluginRuntime("akashic_clients", "scenario", args.source / "plugins/akashic_clients", root_path, root_path, {}))
    ctx = contexts[-1]
    @asynccontextmanager
    async def scope():
        async with ctx.runtime_scope():
            yield ctx
    adapter = declarations[-1].factory(ChannelFactoryContext("scenario", "scenario", "boot", "binding", {}, None, None,
                                                             data_root=root_path, open_scope=scope))
    app = create_chat_app(workspace=root_path, channel=adapter._web, navigation=adapter._state.navigation,
                          message_scope=adapter._message_scope, plugin_ui_scope=adapter._plugin_ui_scope)
    log.ensure_session("akashic:scoped", SessionAttributes())
    project_id = "p_" + "a" * 32
    catalog = await provider.catalog()
    entry = catalog["items"][0]
    await provider.query(entry["id"], entry["revision"], "project.create", {"project_id": project_id, "name":"Scoped"}, session_id=None, turn_id=None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://scenario") as http:
        for kind, id in (("session", "akashic:scoped"), ("project", project_id)):
            response = await http.post("/api/chat/navigation/pins", json={"kind":kind, "id":id, "pinned":True})
            check(response.status_code == 200, "actual plugin factory + owner call + HTTP pin " + kind)
        saved = (await http.get("/api/chat/navigation/pins")).json()["pins"]
        check(len(saved) == 2, "actual owner scope GET reads both refs")
    await owner.dispose()
    check(log.owner("plugin:akashic_clients").read("navigation:pins") is not None, "plugin disposal retains owner state")
    try:
        async with adapter._message_scope():
            pass
    except Exception:
        check(True, "old channel request scope rejects disposed owner")
    else:
        raise AssertionError("disposed owner accepted a request")
    owner = await root.mount(client, name="akashic_clients", inject=plugin.inject,
                             runtime=PluginRuntime("akashic_clients", "scenario", args.source / "plugins/akashic_clients", root_path, root_path, {}))
    ctx = contexts[-1]
    adapter = declarations[-1].factory(ChannelFactoryContext("scenario", "scenario", "boot", "binding-next", {}, None, None,
                                                             data_root=root_path, open_scope=scope))
    async with adapter._message_scope():
        check([ref.model_dump() for ref in adapter._state.navigation.read()] == saved, "fresh plugin activation reads same durable ordered preferences")
    await provider.aclose()
    await root.dispose()
    log.close()

with tempfile.TemporaryDirectory(prefix="akashic-pins-") as directory:
    asyncio.run(run(Path(directory)))
    asyncio.run(composition(Path(directory)))
files = ["plugins/akashic_clients/" + name for name in ("navigation.py", "plugin.py", "channel.py", "chat_api.py", "services.py")]
report = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.source, text=True).strip(),
          "source_sha256": {file: hashlib.sha256((args.source/file).read_bytes()).hexdigest() for file in files},
          "scenario_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "checks": checks,
          "limitations": ["ASGI in-process, no real browser/device or network listener", "Projects query transport is fixture delegating actual Projects owner", "CompositionRoot uses a registrar fixture; no formal installed distribution or network Channel start/stop"]}
args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(f"{len(checks)} checks passed: {args.output}")
