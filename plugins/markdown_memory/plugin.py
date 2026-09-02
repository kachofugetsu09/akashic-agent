from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, BinaryIO, cast

from agent.lifecycle.composition import PROMPT_RENDER_EVENT
from agent.lifecycle.types import PromptRenderCtx
from agent.llm_json import load_json_object_loose
from agent.plugin_composition import (
    CHAT_MODELS,
    CONTEXT_PROJECTION_COMMITTED,
    CONTEXT_PROJECTION_FACTS,
    ChatModels,
    Context,
    ContextProjectionCommitted,
    ModelRequest,
    ModelRole,
    RUNTIME_STARTED,
)
from agent.prompting import PromptSectionRender
from agent.prompting.section_names import (
    LONG_TERM_PROFILE_SECTION,
    SELF_PROFILE_SECTION,
)
from infra.persistence.json_store import atomic_write_text
from .store import MarkdownProfileStore, content_digest

api_version = 3
name = "markdown_memory"
version = "1.0.0"
desc = "Project committed conversation context into MEMORY.md and SELF.md"
author = "Akashic Core"
inject = (CHAT_MODELS,)
skill_roots = ()
drift_skill_roots = ()
workspace_roots = ()
workspace_files = (
    "memory/MEMORY.md",
    "memory/SELF.md",
    "memory/markdown-profile-writes.db",
    "memory/markdown-profile.lock",
    "memory/PENDING.md",
    "memory/PENDING.snapshot.md",
    "memory/PENDING.retired.md",
)

_MEMORY_HEADINGS = (
    "# 用户长期记忆",
    "## 用户事实",
    "## 用户偏好",
    "## 用户明确要求长期记住的关键内容",
)
_MEMORY_OPTIONAL_HEADING = "## 助手操作上下文"
_SELF_HEADINGS = (
    "# Akashic 的自我认知",
    "## 人格与形象",
    "## 我对当前用户的理解",
    "## 我们关系的定义",
)


async def apply(ctx: Context, config: object) -> None:
    """Mount prompt projection and post-response profile maintenance."""

    _ = config
    store = MarkdownProfileStore(
        ctx.workspace_file("memory/MEMORY.md"),
        ctx.workspace_file("memory/SELF.md"),
        ctx.workspace_file("memory/markdown-profile-writes.db"),
    )
    lock_path = ctx.workspace_file("memory/markdown-profile.lock")
    pending_path = ctx.workspace_file("memory/PENDING.md")
    pending_snapshot_path = ctx.workspace_file("memory/PENDING.snapshot.md")
    retired_pending_path = ctx.workspace_file("memory/PENDING.retired.md")
    chat_models = ctx.require(CHAT_MODELS)
    _ = await ctx.on(PROMPT_RENDER_EVENT, lambda event: _inject_profiles(event, store))
    _ = await ctx.on(
        CONTEXT_PROJECTION_COMMITTED,
        lambda event: _project_committed(ctx, event, store, chat_models, lock_path),
    )
    _ = await ctx.on(
        RUNTIME_STARTED,
        lambda _event: _start_store(
            store,
            lock_path,
            pending_path,
            pending_snapshot_path,
            retired_pending_path,
        ),
    )


async def _inject_profiles(event: PromptRenderCtx, store: MarkdownProfileStore) -> None:
    """Append ordinary prompt sections from this plugin's declared files."""

    if SELF_PROFILE_SECTION not in event.disabled_sections:
        self_profile = store.read_self().strip()
        if self_profile:
            event.system_sections_bottom.append(
                PromptSectionRender(
                    name=SELF_PROFILE_SECTION,
                    content=f"## Akashic 自我认知\n\n{self_profile}",
                    is_static=False,
                    order=30,
                )
            )
    if LONG_TERM_PROFILE_SECTION not in event.disabled_sections:
        memory = store.read_memory().strip()
        if memory:
            event.system_sections_bottom.append(
                PromptSectionRender(
                    name=LONG_TERM_PROFILE_SECTION,
                    content=f"## Long-term Memory\n{memory}",
                    is_static=False,
                    order=35,
                )
            )


async def _project_committed(
    ctx: Context,
    event: ContextProjectionCommitted,
    store: MarkdownProfileStore,
    chat_models: ChatModels,
    lock_path: Path,
) -> None:
    """Consume one validated durable fact exactly once after the business response."""

    async with ctx.runtime_scope():
        facts = ctx.get(CONTEXT_PROJECTION_FACTS)
        if facts is None:
            raise RuntimeError("Markdown memory 收到 projection event 但缺少 fact reader")
        fact = facts.get_committed(
            event.access_grant,
            session_key=event.session_key,
            source_ref=event.source_ref,
        )
        if fact is None:
            raise RuntimeError("Markdown memory 无法读取 committed projection fact")
        async with _profile_lock(lock_path):
            for pending_source_ref in store.pending_source_refs():
                store.apply_pending(pending_source_ref)
            if store.is_applied(event.source_ref):
                return
            draft = store.read_draft(event.source_ref)
            if draft is None:
                draft = await _prepare_draft(
                    fact.checkpoint_json,
                    event.source_ref,
                    store,
                    chat_models,
                )
                _ = store.write_draft(
                    event.source_ref,
                    draft,
                    session_key=fact.session_key,
                    generation=fact.generation,
                )
            _validate_draft(draft)
            store.apply_draft(event.source_ref, draft)


async def _prepare_draft(
    checkpoint_json: str,
    source_ref: str,
    store: MarkdownProfileStore,
    chat_models: ChatModels,
) -> dict[str, object]:
    receipt = cast(Any, json.loads(checkpoint_json))
    if not isinstance(receipt, dict):
        raise ValueError("Markdown memory checkpoint fact schema 无效")
    if receipt.get("version") == 2:
        legacy = receipt.get("markdown_draft")
        if not isinstance(legacy, dict):
            raise ValueError("v2 compaction receipt 缺少 markdown_draft")
        if legacy.get("source_ref") != source_ref:
            raise ValueError("v2 compaction markdown_draft source_ref 冲突")
        pending_items = legacy.get("pending_items", "")
        if not isinstance(pending_items, str):
            raise ValueError("v2 compaction markdown_draft pending_items 无效")
        return _prepare_legacy_draft(pending_items, store)
    if receipt.get("version") != 4:
        raise ValueError("Markdown memory 不支持此 compaction receipt version")
    source = _source_text(cast(dict[str, object], receipt))
    return await _prepare_profile_draft(source, store, chat_models)


async def _prepare_profile_draft(
    source: str,
    store: MarkdownProfileStore,
    chat_models: ChatModels,
) -> dict[str, object]:
    current_memory = store.read_memory()
    current_self = store.read_self()
    prompt = _profile_prompt(current_memory, current_self, source)
    async with chat_models.independent_execution() as execution:
        provider = execution.chat(ModelRole.DEFAULT)
        output_cap = provider.descriptor.capabilities.max_output_tokens or 4_096
        response = await provider.complete(
            ModelRequest(
                messages=[{"role": "user", "content": prompt}],
                max_output_tokens=min(4_096, output_cap),
                disable_reasoning=True,
            )
        )
    raw = load_json_object_loose(response.content or "")
    if not isinstance(raw, dict):
        raise ValueError("Markdown memory 模型必须返回 JSON object")
    memory = raw.get("memory")
    self_profile = raw.get("self")
    if not isinstance(memory, str) or not isinstance(self_profile, str):
        raise ValueError("Markdown memory 模型缺少 memory/self 字符串")
    memory = memory.strip()
    self_profile = self_profile.strip() + "\n"
    if memory:
        memory += "\n"
    _validate_memory(memory)
    _validate_self(self_profile)
    _validate_preserved_bullets(current_memory, memory, document="MEMORY.md")
    _validate_preserved_bullets(current_self, self_profile, document="SELF.md")
    return {
        "version": 1,
        "memory": memory,
        "self": self_profile,
        "memory_before": current_memory,
        "self_before": current_self,
        "memory_before_digest": content_digest(current_memory),
        "self_before_digest": content_digest(current_self),
        "memory_after_digest": content_digest(memory),
        "self_after_digest": content_digest(self_profile),
    }


async def _start_store(
    store: MarkdownProfileStore,
    lock_path: Path,
    pending_path: Path,
    snapshot_path: Path,
    retired_path: Path,
) -> None:
    """Recover document commits, then retire the old pending queue."""

    async with _profile_lock(lock_path):
        for source_ref in store.pending_source_refs():
            store.apply_pending(source_ref)
    await _migrate_pending(
        store,
        lock_path,
        pending_path,
        snapshot_path,
        retired_path,
    )


async def _migrate_pending(
    store: MarkdownProfileStore,
    lock_path: Path,
    pending_path: Path,
    snapshot_path: Path,
    retired_path: Path,
) -> None:
    """Merge exact retired queue bytes once, then preserve their file boundary."""

    async with _profile_lock(lock_path):
        migration = store.read_legacy_pending_migration()
        if migration is None:
            pending = (
                pending_path.read_text(encoding="utf-8")
                if pending_path.exists()
                else ""
            )
            snapshot = (
                snapshot_path.read_text(encoding="utf-8")
                if snapshot_path.exists()
                else ""
            )
            if not pending and not snapshot:
                return
            migration = {
                "version": 1,
                "pending": pending,
                "pending_digest": content_digest(pending),
                "snapshot": snapshot,
                "snapshot_digest": content_digest(snapshot),
            }
            store.write_legacy_pending_migration(migration)
        pending = migration.get("pending")
        snapshot = migration.get("snapshot")
        pending_digest = migration.get("pending_digest")
        snapshot_digest = migration.get("snapshot_digest")
        if not all(
            isinstance(value, str)
            for value in (pending, snapshot, pending_digest, snapshot_digest)
        ):
            raise ValueError("legacy PENDING migration receipt schema 无效")
        assert isinstance(pending, str)
        assert isinstance(snapshot, str)
        if (
            content_digest(pending) != pending_digest
            or content_digest(snapshot) != snapshot_digest
        ):
            raise ValueError("legacy PENDING migration receipt digest 无效")
        encoded_migration = json.dumps(
            migration,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(encoded_migration.encode("utf-8")).hexdigest()
        source_ref = f"legacy-pending:{digest}"
        combined = "\n".join(item for item in (snapshot, pending) if item)
        if not store.is_applied(source_ref):
            draft = store.read_draft(source_ref)
            if draft is None:
                draft = _prepare_legacy_draft(combined, store)
                _ = store.write_draft(
                    source_ref,
                    draft,
                    session_key="legacy-pending",
                    generation=0,
                )
            _validate_draft(draft)
            store.apply_draft(source_ref, draft)
        archive = json.dumps(
            {"source_ref": source_ref, **migration},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ) + "\n"
        if retired_path.exists() and retired_path.read_text(encoding="utf-8") != archive:
            raise RuntimeError("PENDING retired archive 内容冲突")
        current_pending = pending_path.read_text(encoding="utf-8") if pending_path.exists() else ""
        current_snapshot = (
            snapshot_path.read_text(encoding="utf-8") if snapshot_path.exists() else ""
        )
        if current_pending not in {"", pending}:
            raise RuntimeError("PENDING.md 在退休 receipt 后出现新内容，拒绝清空")
        if current_snapshot not in {"", snapshot}:
            raise RuntimeError("PENDING.snapshot.md 在退休 receipt 后出现新内容，拒绝清空")
        atomic_write_text(retired_path, archive, domain="pending_retirement")
        atomic_write_text(pending_path, "", domain="pending_retirement")
        atomic_write_text(snapshot_path, "", domain="pending_retirement")
        store.mark_legacy_pending_retired(source_ref)


def _source_text(receipt: dict[str, object]) -> str:
    """Use the v4 exact source plan, never SessionDB."""

    checkpoint = receipt.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("v4 compaction receipt 缺少 checkpoint")
    source = cast(dict[str, object], checkpoint).get("selected_source_messages")
    if not isinstance(source, list):
        raise ValueError("v4 compaction receipt 缺少 exact source plan")
    return json.dumps(source, ensure_ascii=False, sort_keys=True)


def _prepare_legacy_draft(
    pending_items: str,
    store: MarkdownProfileStore,
) -> dict[str, object]:
    """Preserve every retired pending line without another model interpretation."""

    current_memory = store.read_memory()
    current_self = store.read_self()
    memory = _merge_legacy_pending(current_memory, pending_items)
    _validate_memory(memory)
    _validate_self(current_self)
    return {
        "version": 1,
        "memory": memory,
        "self": current_self,
        "memory_before": current_memory,
        "self_before": current_self,
        "memory_before_digest": content_digest(current_memory),
        "self_before_digest": content_digest(current_self),
        "memory_after_digest": content_digest(memory),
        "self_after_digest": content_digest(current_self),
    }


def _merge_legacy_pending(memory: str, pending_items: str) -> str:
    """Map old tagged lines into the fixed MEMORY schema without dropping text."""

    if not pending_items.strip():
        return memory
    content = memory
    if not content.strip():
        content = "\n\n".join(_MEMORY_HEADINGS) + "\n"
    grouped: dict[str, list[str]] = {heading: [] for heading in _MEMORY_HEADINGS[1:]}
    grouped[_MEMORY_OPTIONAL_HEADING] = []
    heading_by_tag = {
        "identity": _MEMORY_HEADINGS[1],
        "health_long_term": _MEMORY_HEADINGS[1],
        "preference": _MEMORY_HEADINGS[2],
        "key_info": _MEMORY_HEADINGS[3],
        "requested_memory": _MEMORY_HEADINGS[3],
        "correction": _MEMORY_HEADINGS[3],
        "agent_context": _MEMORY_OPTIONAL_HEADING,
    }
    for raw_line in pending_items.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        tag = ""
        if line.startswith("- [") and "]" in line:
            tag = line[3 : line.index("]")].strip().lower()
        target = heading_by_tag.get(tag, _MEMORY_HEADINGS[3])
        preserved = line if line.startswith("- ") else f"- [legacy_pending] {line}"
        grouped[target].append(preserved)
    for heading, lines in grouped.items():
        content = _append_section_lines(content, heading, lines)
    return content.rstrip() + "\n"


def _append_section_lines(content: str, heading: str, lines: list[str]) -> str:
    unique = [line for line in lines if line not in content.splitlines()]
    if not unique:
        return content
    values = content.rstrip().splitlines()
    if heading not in values:
        values.extend(["", heading])
    start = values.index(heading) + 1
    end = next(
        (index for index in range(start, len(values)) if values[index].startswith("#")),
        len(values),
    )
    values[end:end] = unique
    return "\n".join(values) + "\n"


def _profile_prompt(memory: str, self_profile: str, source: str) -> str:
    return f"""你维护两个长期 Markdown 档案。根据本次已提交的精确对话事实，返回完整的新档案。

只返回 JSON：{{"memory":"完整 MEMORY.md", "self":"完整 SELF.md"}}。

MEMORY.md 只保留跨对话稳定的用户事实、偏好、用户明确要求记住的内容，以及已部署且已授权使用的助手操作上下文。不要写短期状态、动态指标、网络诊断、方案讨论、SOP 或助手建议。没有新事实时保持原文。

SELF.md 只能包含这四个标题：# Akashic 的自我认知、## 人格与形象、## 我对当前用户的理解、## 我们关系的定义。它不是用户资料清单；大多数事实不应改变 SELF.md，没有关系层面的长期证据时保持原文。

当前 MEMORY.md：
{memory or "（空）"}

当前 SELF.md：
{self_profile}

本次精确来源（v2 legacy draft 或 v4 source plan）：
{source}
"""


def _validate_draft(payload: dict[str, object]) -> None:
    memory = payload.get("memory")
    self_profile = payload.get("self")
    memory_before = payload.get("memory_before")
    self_before = payload.get("self_before")
    if not all(
        isinstance(value, str)
        for value in (memory, self_profile, memory_before, self_before)
    ):
        raise ValueError("Markdown profile draft schema 无效")
    assert isinstance(memory, str)
    assert isinstance(self_profile, str)
    assert isinstance(memory_before, str)
    assert isinstance(self_before, str)
    _validate_memory(memory)
    _validate_self(self_profile)
    _validate_preserved_bullets(memory_before, memory, document="MEMORY.md")
    _validate_preserved_bullets(self_before, self_profile, document="SELF.md")


def _validate_preserved_bullets(before: str, after: str, *, document: str) -> None:
    """Reject implicit deletion of any previously committed profile fact."""

    old_facts = {
        line.strip() for line in before.splitlines() if line.lstrip().startswith("- ")
    }
    new_facts = {
        line.strip() for line in after.splitlines() if line.lstrip().startswith("- ")
    }
    removed = sorted(old_facts - new_facts)
    if removed:
        raise ValueError(f"{document} 不得隐式删除既有事实: {removed}")


def _headings(content: str) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in content.splitlines()
        if line.lstrip().startswith("#")
    )


def _validate_memory(content: str) -> None:
    if not content:
        return
    headings = _headings(content)
    if headings not in {
        _MEMORY_HEADINGS,
        _MEMORY_HEADINGS + (_MEMORY_OPTIONAL_HEADING,),
    } or "```" in content:
        raise ValueError("MEMORY.md 模型输出格式无效")
    if not any(line.lstrip().startswith("- ") for line in content.splitlines()):
        raise ValueError("MEMORY.md 模型输出不包含记忆条目")


def _validate_self(content: str) -> None:
    lines = content.splitlines()
    if _headings(content) != _SELF_HEADINGS or "```" in content:
        raise ValueError("SELF.md 模型输出格式无效")
    positions = [lines.index(heading) for heading in _SELF_HEADINGS] + [len(lines)]
    for index in range(1, len(_SELF_HEADINGS)):
        if not any(
            line.lstrip().startswith("- ")
            for line in lines[positions[index] + 1 : positions[index + 1]]
        ):
            raise ValueError(f"SELF.md section 为空: {_SELF_HEADINGS[index]}")


@asynccontextmanager
async def _profile_lock(path: Path) -> AsyncGenerator[None]:
    """Serialize profile projections across sessions and live plugin generations."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = await asyncio.to_thread(_open_and_lock, path)
    try:
        yield
    finally:
        await asyncio.to_thread(_unlock_and_close, handle)


def _open_and_lock(path: Path) -> BinaryIO:
    handle = path.open("a+b")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def _unlock_and_close(handle: BinaryIO) -> None:
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()
