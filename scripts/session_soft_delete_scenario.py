"""Manual acceptance: session soft delete over real HTTP (curl) and real Akasha paths."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time

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

from yoyo import get_backend, read_migrations

from agent.migrations.context import bind_migration_context
from agent.plugin_composition.messages import SessionAdmin
from plugins.akashic_clients.chat_api import create_chat_app
from plugins.akashic_clients.navigation import NavigationPreferences, PinReference
from plugins.akashic_clients.web_chat import WebChatChannel
from session.log import MessageLog, SessionAttributes
from session.message import ContentPart, ContentReferences, Input, Output

checks: list[str] = []
http_log: list[dict[str, object]] = []
def check(condition: bool, description: str) -> None:
    assert condition, description
    checks.append(description)


def curl(method: str, url: str) -> tuple[int, dict[str, object]]:
    raw = subprocess.run(["curl", "-sS", "-X", method, "-w", "\n%{http_code}", url],
                         capture_output=True, text=True, check=True).stdout
    body, status = raw.rsplit("\n", 1)
    http_log.append({"method": method, "url": url, "status": int(status), "body": body[:400]})
    return int(status), json.loads(body) if body else {}


def table_snapshot(path: Path, table: str) -> list[tuple[object, ...]]:
    with sqlite3.connect(path) as db:
        return db.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()


def run_real_migration(workspace: Path) -> None:
    """用真实 yoyo 账本只应用软删迁移，证明加列路径可重放。"""
    backend = get_backend(f"sqlite:///{workspace / 'migrations.sqlite3'}")
    with backend, bind_migration_context(config_path=workspace / "config.toml", workspace=workspace):
        loaded = read_migrations(str(args.source / "migrations" / "core"))
        selected = [m for m in loaded if m.id == "20261004_03_session_soft_delete"]
        assert len(selected) == 1, "migration not found"
        backend.apply_migrations(backend.to_apply(type(loaded)(selected, loaded.post_apply)))


async def akasha_participation(root: Path, deleted_session: str) -> None:
    """软删会话仍被在线学习与全量重建消费：同一 MessageConsumer、同一输入。"""
    from agent.plugin_composition import CompositionRoot, Context
    from agent.plugin_composition.archive import PluginArchive
    from agent.plugin_composition.bindings import Bindings
    from plugins.akasha.application.rebuild import rebuild_from_catalog
    from plugins.akasha.application.consumer import MessageConsumer
    from plugins.akasha.domain.model import MemoryConfig
    from plugins.akasha.learning import AKASHA_LEARNING, Learning, LearningConfig
    from plugins.content.api import legacy_post_commit_effect
    from plugins.content.plugin import check_text
    from plugins.turn_projection.plugin import TurnProjection
    from session.embedding_store import MessageEmbeddings

    log = MessageLog(root / "sessions.db")
    embeddings = MessageEmbeddings(log)
    learning = Learning(TurnProjection(), owner="akasha", post_commit_effect=legacy_post_commit_effect)
    rule = LearningConfig(embedding_model="fixed", dimension=3, sources=("conversation",))
    bodies = (("user", Input, lambda s: Input((ContentPart("text", f"episode of {s}"),))),
              ("assistant", Output, lambda s: Output((ContentPart("text", f"answer of {s}"),), "complete")))
    for session in ("akashic:learn-a", deleted_session):
        log.ensure_session(session, SessionAttributes())
        for author, kind, make in bodies:
            message = log.writer(session, author=author, source="conversation",
                                 body_types=(kind,), content={"text": check_text}).append(
                                     f"{author}-{session}", make(session))
            embeddings.bind(learning.text).save(message, model="fixed", embedding=[1.0, 0.0, 0.0])
    log.save_binding("fixed-learning", {"version": 1, "service": AKASHA_LEARNING.name,
                                        "root_ref": "fixture-archive", "metadata": rule.model_dump()})
    await SessionAdmin(log).set_deleted(deleted_session, deleted=True)

    root_cm = CompositionRoot("soft-delete-akasha")
    bindings = Bindings(log, PluginArchive(root / "archives"), root_cm)

    async def provide(ctx: Context) -> None:
        await ctx.provide(AKASHA_LEARNING, learning)

    await root_cm.mount(provide, name="learning")

    async def never_embed(texts: list[str]) -> list[list[float]]:
        raise AssertionError(f"不应重新嵌入: {len(texts)} 条")

    catalog = log.catalog()
    consumer = await MessageConsumer.load(root / "memory-online.db", catalog=catalog,
                                          embeddings=embeddings, bindings=bindings,
                                          config=MemoryConfig(), cutover=False)
    try:
        learned = await consumer.consume(catalog=catalog, learning_binding="fixed-learning",
                                         embeddings=embeddings, bindings=bindings,
                                         embed_batch=never_embed)
        check(learned == 2 and {turn.session_key for turn in consumer.cycle.turns}
              == {"akashic:learn-a", deleted_session}, "在线学习仍消费软删会话的样本")
    finally:
        consumer.close()
    report = await rebuild_from_catalog(
        catalog=catalog, embeddings=embeddings, bindings=bindings, config=MemoryConfig(),
        learning_binding="fixed-learning", embed_batch=never_embed,
        memory_path=root / "memory-rebuild.db", backup_root=root / "backups" / "rebuild")
    check(report.turns == 2 and report.sessions == 2 and report.embedded_messages == 0,
          "全量重建重放仍包含软删会话且复用已存向量")
    await root_cm.dispose()
    log.close()


def main() -> None:
    workspace = Path(tempfile.mkdtemp(prefix="akashic-soft-delete-"))
    os.environ["HOME"] = str(workspace)
    path = workspace / "sessions.db"
    log = MessageLog(path)
    for key in ("akashic:alpha", "akashic:beta", "telegram:foreign"):
        log.ensure_session(key, SessionAttributes())
        log.writer(key, author="user", source="conversation", body_types=(Input,),
                   content={"text": lambda _: ContentReferences()}).append(
                       "m-" + key, Input((ContentPart("text", "正文 " + key),)))
    messages_before = table_snapshot(path, "messages")
    log.close()

    # 1. 旧库形态（无 deleted_at 列）可直接读取；真实 yoyo 迁移只加列、不改数据。
    with sqlite3.connect(path) as db:
        db.execute("ALTER TABLE sessions DROP COLUMN deleted_at")
    log = MessageLog(path)
    catalog = log.catalog()
    check({e.session_id for e in catalog.sessions(prefix="akashic:").items}
          == {"akashic:alpha", "akashic:beta"}, "无 deleted_at 的旧库仍可读且全量列出")
    check(not catalog.reader("akashic:alpha").deleted, "旧库按未软删读取")
    log.close()
    run_real_migration(workspace)
    with sqlite3.connect(path) as db:
        check("deleted_at" in [row[1] for row in db.execute("PRAGMA table_info(sessions)")],
              "yoyo 迁移只增加 deleted_at 列")
        check(table_snapshot(path, "messages") == messages_before, "迁移不改写 messages 数据")

    # 2. 真实 uvicorn + curl 验证软删/恢复语义。
    log = MessageLog(path)
    navigation = NavigationPreferences(lambda: log.owner("plugin:akashic_clients"))
    navigation.update(PinReference(kind="session", id="akashic:alpha"), pinned=True)

    @asynccontextmanager
    async def admin_scope():
        yield SessionAdmin(log)

    app = create_chat_app(workspace=workspace, channel=WebChatChannel("akashic"),
                          messages=log.catalog(), navigation=navigation,
                          session_admin_scope=admin_scope)
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    base = f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    try:
        status, body = curl("GET", f"{base}/api/chat/sessions")
        check(status == 200 and {row["key"] for row in body["items"]}
              == {"akashic:alpha", "akashic:beta"}, "初始列表包含两个 akashic 会话")
        status, body = curl("GET", f"{base}/api/chat/sessions/akashic:alpha/messages")
        check(status == 200 and body["deleted"] is False
              and body["items"][0]["body"]["parts"][0]["value"] == "正文 akashic:alpha",
              "直接访问未删会话返回消息且 deleted=false")
        status, body = curl("POST", f"{base}/api/chat/sessions/akashic:alpha/delete")
        stamp = body.get("deleted_at")
        check(status == 200 and body["deleted"] is True and isinstance(stamp, str), "delete 软删成功")
        status, body = curl("POST", f"{base}/api/chat/sessions/akashic:alpha/delete")
        check(status == 200 and body["deleted_at"] == stamp, "重复 delete 幂等且时间戳不变")
        status, body = curl("GET", f"{base}/api/chat/sessions")
        check(status == 200 and [row["key"] for row in body["items"]] == ["akashic:beta"], "软删后列表排除")
        status, body = curl("GET", f"{base}/api/chat/sessions/akashic:alpha/messages")
        check(status == 200 and body["deleted"] is True and len(body["items"]) == 1,
              "已删会话直接访问只读且标记 deleted")
        status, body = curl("GET", f"{base}/api/chat/navigation/pins")
        check([ref["id"] for ref in body["pins"]] == ["akashic:alpha"] and body["sessions"] == [],
              "pin 引用保留但会话行缺席")
        status, body = curl("POST", f"{base}/api/chat/sessions/akashic:alpha/undelete")
        check(status == 200 and body["deleted"] is False and body["deleted_at"] is None, "undelete 恢复成功")
        status, body = curl("POST", f"{base}/api/chat/sessions/akashic:alpha/undelete")
        check(status == 200 and body["deleted"] is False, "重复 undelete 幂等")
        status, body = curl("GET", f"{base}/api/chat/sessions")
        check(status == 200 and len(body["items"]) == 2, "恢复后列表回来")
        status, body = curl("GET", f"{base}/api/chat/navigation/pins")
        check(len(body["sessions"]) == 1 and body["sessions"][0]["key"] == "akashic:alpha",
              "恢复后 pin 会话行原样可用")
        status, _ = curl("POST", f"{base}/api/chat/sessions/akashic:missing/delete")
        check(status == 404, "删除不存在的会话返回 404")
        status, _ = curl("POST", f"{base}/api/chat/sessions/telegram:foreign/delete")
        check(status == 400, "删除非本聊天目录会话返回 400")
    finally:
        server.should_exit = True
        thread.join(timeout=20)
        log.close()

    with sqlite3.connect(path) as db:
        check(db.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "SQLite integrity")
        check(table_snapshot(path, "messages") == messages_before, "消息物理保留")
        check(db.execute("SELECT COUNT(*) FROM sessions WHERE deleted_at IS NOT NULL").fetchone()[0] == 0,
              "恢复后 deleted_at 清空")
        check("正文 akashic:alpha" in
              db.execute("SELECT body FROM messages WHERE session_key='akashic:alpha'").fetchone()[0],
              "消息正文逐字节保留")

    asyncio.run(akasha_participation(workspace, "akashic:learn-b"))
    report = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.source,
                                              text=True).strip(),
              "checks": checks, "http_log": http_log,
              "limitations": ["uvicorn 直绑 create_chat_app，未经完整 CompositionRoot 装配",
                              "Akasha embedding 使用固定向量 fixture，未调用真实模型"]}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(checks)} checks passed: {args.output}")


main()
