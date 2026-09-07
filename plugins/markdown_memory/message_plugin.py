"""从成功 Output 的摘要引用恢复 Markdown 更新；不监听瞬时提交事件。"""
from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition import CHAT_MODELS, RUNTIME_STARTED, RUNTIME_STOPPING, Context
from agent.plugin_composition.bindings import BINDINGS, Bindings
from agent.plugin_composition.messages import MESSAGE_CATALOG
from agent.plugin_composition.models import ChatModels
from agent.turn_effects import PostCommitEffect
from plugins.compaction.records import COMPACTION_SUMMARIES, SummaryLookup, SummaryRecord
from plugins.compaction.message_summary import source_text, summary_groups
from plugins.content.api import legacy_post_commit_effect
from plugins.context.api import Materials, check_summary, summary_range
from plugins.context.materials import MATERIALS
from plugins.turn_projection.plugin import TURN_PROJECTION, TurnProjection
from session.log import MessageCatalog, MessageReader
from session.message import ContentPart, Message, Output

from .plugin import prepare_profile_draft, profile_lock, start_store, check_draft
from .store import DEFAULT_SELF_MD, MarkdownProfileStore

api_version = 3
name = "markdown_memory"
version = "4.0.0"
desc = "把已使用摘要的确切原文投影到 MEMORY.md 和 SELF.md"
inject = (CHAT_MODELS, MATERIALS, BINDINGS, MESSAGE_CATALOG, TURN_PROJECTION)
workspace_files = (
    "memory/MEMORY.md", "memory/SELF.md", "memory/markdown-profile-writes.db",
    "memory/markdown-profile.lock", "memory/PENDING.md", "memory/PENDING.snapshot.md",
    "memory/PENDING.retired.md",
)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sources: tuple[str, ...] = Field(default=("conversation", "programmatic"), min_length=1)


def _unapplied_messages(record: SummaryRecord, lookup: SummaryLookup, reader: MessageReader,
                        store: MarkdownProfileStore, sources: tuple[str, ...],
                        projection: TurnProjection) -> tuple[Message, ...] | None:
    """从最近已写入的祖先之后取原文，跳过未使用的摘要不会漏掉它覆盖的事实。"""
    start = 0
    latest = store.latest_applied(record.session_id)
    if latest is not None:
        latest_ref, generation = latest
        newer = record if record.generation > generation else lookup.resolve(
            {"record_ref": latest_ref, "session_id": record.session_id}, session_id=reader.session_id)
        target_ref = latest_ref if record.generation > generation else record.reference
        target_generation = min(record.generation, generation)
        while newer.generation > target_generation:
            newer = lookup.resolve({"record_ref": newer.parent, "session_id": record.session_id},
                                   session_id=reader.session_id)
        if newer.reference != target_ref:
            raise ValueError("Markdown 当前档案与摘要不属于同一父链")
        if record.generation <= generation:
            return None
        start = len(newer.source_message_ids)
    snapshot = reader.snapshot()
    covered = summary_range(snapshot, record.source_message_ids)
    # 历史 suppress 是整个工作单元的资格，不能只删用户行后继续学习其回答。
    by_id = {message.message_id: message for message in snapshot[:covered.stop]}
    excluded: set[str] = set()
    for source in sources:
        for turn in projection.project(snapshot[:covered.stop], source):
            ids = (*turn.message_ids, *(identity for _, identity in turn.observations))
            effects = tuple(legacy_post_commit_effect(by_id[identity]) for identity in ids)
            if PostCommitEffect.SUPPRESS in effects:
                excluded.update(ids)
    selected = tuple(message for message in snapshot[covered.start + start:covered.stop]
                     if message.source in sources and message.message_id not in excluded)
    # 摘要使用了哪些原始 ID 不等于每条原文都可沉淀；迟到结果沿同一放弃边界排除。
    return tuple(message for group in summary_groups((selected,), snapshot[:covered.stop]) for message in group)


async def project(message: Message, *, reader: MessageReader, bindings: Bindings,
                  store: MarkdownProfileStore, models: ChatModels, lock_path: Path,
                  sources: tuple[str, ...], projection: TurnProjection) -> None:
    """只处理已提交的模型 Output；两份文件沿原 before-image receipt 恢复。"""
    if reader.attributes.learning != "eligible" or message.source not in sources or not isinstance(message.body, Output):
        return
    refs = [part for part in message.body.parts if isinstance(part, ContentPart) and part.kind == "context.summary"]
    if not refs:
        return
    if len(refs) != 1:
        raise ValueError("一个 Output 只能声明实际使用的一份摘要")
    reference = check_summary(refs[0]).binding_ids[0]
    async with profile_lock(lock_path):
        # 1. 先完成已固定的文件写入；重复 Output 不重新调用模型。
        for pending in store.pending_source_refs():
            store.apply_pending(pending)
        async with bindings.open(reference, COMPACTION_SUMMARIES) as (lookup, metadata):
            record = lookup.resolve(metadata, session_id=message.session_id)
            if store.is_applied(record.reference):
                return
            draft = store.read_draft(record.reference)
            selected = _unapplied_messages(record, lookup, reader, store, sources, projection)
            if not selected:
                return
        # 模型属于当前 Markdown 作用域；先关闭旧摘要的只读归档 scope。
        if draft is None:
            draft = await prepare_profile_draft(source_text(selected), store, models)
        # model draft 后退出也可能缺 order；用实际摘要身份补齐整份准备再写文件。
        _ = store.write_draft(record.reference, draft, session_key=record.session_id, generation=record.generation)
        # 2. 取消前若已留下 draft，下一次沿同一恢复点继续，不重算 before-image。
        check_draft(draft)
        store.apply_draft(record.reference, draft)


async def apply(ctx: Context, config: Config) -> None:
    """启动后才创建文件与跟随日志；归档 apply 不写入正式记忆。"""
    store: MarkdownProfileStore | None = None
    watcher: asyncio.Task[None] | None = None
    lock_path = ctx.workspace_file("memory/markdown-profile.lock")

    async def prepare(snapshot: tuple[Message, ...], source: str) -> Materials:
        # 完整初始态只投影 Store 的同一默认值；不创建文件或消费旧队列。
        state_files = tuple(ctx.workspace_file(name) for name in workspace_files
                            if name != "memory/markdown-profile.lock")
        if not any(path.exists() for path in state_files):
            self_profile, memory = DEFAULT_SELF_MD.strip(), ""
        else:
            async with profile_lock(lock_path, create=False):
                self_profile = ctx.workspace_file("memory/SELF.md").read_text(encoding="utf-8").strip()
                memory = ctx.workspace_file("memory/MEMORY.md").read_text(encoding="utf-8").strip()
        parts: list[str] = []
        if self_profile:
            parts.append("## Akashic 自我认知\n\n" + self_profile)
        if memory:
            parts.append("## Long-term Memory\n" + memory)
        return Materials("\n\n".join(parts))

    async def follow(catalog: MessageCatalog) -> None:
        cursor: dict[str, int] = {}
        async for heads in catalog.follow():
            for session, head in heads.items():
                if head <= cursor.get(session, -1):
                    continue
                async with ctx.runtime_scope():
                    assert store is not None
                    reader = catalog.reader(session)
                    for message in reader.snapshot():
                        if cursor.get(session, -1) < message.seq <= head:
                            await project(message, reader=reader, bindings=ctx.require(BINDINGS),
                                          store=store, models=ctx.require(CHAT_MODELS), lock_path=lock_path,
                                          sources=config.sources, projection=ctx.require(TURN_PROJECTION))
                            cursor[session] = message.seq

    async def start(_event: object) -> None:
        nonlocal store, watcher
        store = MarkdownProfileStore(ctx.workspace_file("memory/MEMORY.md"), ctx.workspace_file("memory/SELF.md"),
                                     ctx.workspace_file("memory/markdown-profile-writes.db"))
        await start_store(store, lock_path, ctx.workspace_file("memory/PENDING.md"),
                           ctx.workspace_file("memory/PENDING.snapshot.md"), ctx.workspace_file("memory/PENDING.retired.md"))
        watcher = await ctx.spawn(follow(ctx.require(MESSAGE_CATALOG)), name="markdown-memory")

    async def stop(_event: object) -> None:
        if watcher is not None:
            _ = watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass

    _ = await ctx.require(MATERIALS).register(ctx, name="markdown_memory", prepare=prepare, prompt=True)
    _ = await ctx.on(RUNTIME_STARTED, start)
    _ = await ctx.on(RUNTIME_STOPPING, stop)
