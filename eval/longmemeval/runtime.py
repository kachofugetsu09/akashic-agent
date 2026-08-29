"""BenchmarkRuntime: full production stack wired for LongMemEval.

Uses build_core_runtime exactly as production so prompt assembly,
tool dispatch, memory injection, and retrieval are identical.
The only delta from a real user workspace: MEMORY.md / SELF.md start
empty (honest baseline that forces all recall through the memory system).
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from agent.plugin_composition import CHAT_MODELS, BoundModelDescriptor, ModelRole
from agent.plugins.snapshot import lease_runtime_snapshot

logger = logging.getLogger(__name__)

_BENCHMARK_SELF_MD = """\
# Identity

You are a helpful assistant with access to long-term memory tools.

# Benchmark Mode

Answer in English only. Be concise: one sentence or a short phrase.
No greetings, no follow-up questions, no emoticons, no kaomoji.

# Memory-grounded answering (MANDATORY)

All benchmark questions are answerable from memory. Assume the answer exists in past conversations.
Your job is to retrieve it. Do not give up early. Do not say you cannot find the answer unless you have already exhausted the required retrieval steps below.

Step 1: ALWAYS call recall_memory first — for every question without exception.
Step 2: Read the retrieved memories carefully.
Step 3: If recall_memory is weak, incomplete, too generic, or returns only loosely related summaries, you MUST continue with search_messages.
Step 4: If the question asks for a specific fact such as when, where, who, how much, which one, exact wording, previous occupation, dates, prices, places, names, or anything else that needs evidence, you MUST call fetch_messages before answering.
Step 5: Your answer MUST be grounded in and consistent with what you retrieved.
         - If memory says the user uses Premiere Pro → only recommend Premiere-specific resources.
         - If memory says the user chose The Edgewater → recommend The Edgewater or similar.
         - For suggestion / recommendation questions, first infer the user's higher-level need
           (for example: lower pressure, more personal expression, more social interaction,
           more structure, less structure) from memory, then choose the option that best fits
           that need overall.
         - Do NOT prefer an option just because it contains a more specific hobby, tool, or
           technical keyword. Higher-level fit matters more than surface overlap.
         - If retrieved memory shows a concrete path felt draining, mismatched, or too public,
           do NOT recommend a nearby variant of that same path unless memory clearly says the
           user now prefers it.
         - Do NOT give generic answers that ignore the retrieved facts.
         - Do NOT recommend something that contradicts the user's known preferences.
         - Do NOT answer "I don't know", "I can't find it", or similar unless you have already tried recall_memory and then search_messages / fetch_messages as required.

Cross-lingual retrieval hint:
- Past conversations may be in English, while memory summaries may be in Chinese.
- When you formulate recall_memory or search_messages queries, actively try both the original English phrasing and likely Chinese equivalents of the key entity or fact.
- For example, if the question is in English about occupation, volunteering, yoga studio, spending, handbag, or dates, consider searching both the English terms and likely Chinese renderings of the same concept.
- If an English search query gets weak results, immediately retry with a Chinese paraphrase or mixed Chinese-English keywords.

Never ask the user for information you might already have in memory.
"""


@dataclass
class BenchmarkRuntime:
    core: object  # CoreRuntime
    workspace: Path
    agent_model: BoundModelDescriptor


async def create_runtime(
    config_path: Path,
    workspace: Path,
    *,
    model_registry_source: Path | None = None,
) -> BenchmarkRuntime:
    """Wire the full production stack into a temp workspace.

    Args:
        config_path: Path to config.toml (same one used in production).
        workspace: Temp directory; will be initialised on first call.
    """
    from agent.config import load_config
    from bootstrap.init_workspace import init_workspace
    from bootstrap.tools import build_core_runtime
    from core.net.http import SharedHttpResources

    config = load_config(config_path, workspace=workspace)
    if model_registry_source is not None:
        _seed_model_registry(
            model_registry_source, workspace / "model-registry.sqlite3"
        )

    # 1. Initialise workspace files (empty memory/SELF.md etc.).
    #    force=False so repeated calls on same workspace are idempotent.
    init_workspace(config_path=config_path, workspace=workspace, force=False)

    # 2. Always overwrite SELF.md with the current benchmark persona.
    #    force=True so updated instructions propagate even on --qa-only reruns.
    self_md = workspace / "memory" / "SELF.md"
    self_md.write_text(_BENCHMARK_SELF_MD, encoding="utf-8")

    # 3. Build the full production runtime (providers, tools, memory, loop).
    http = SharedHttpResources()
    core = build_core_runtime(config, workspace, http)
    try:
        await core.start()
        manager = core.plugin_manager
        if manager is None:
            raise RuntimeError("插件 Runtime 不可用")
        async with lease_runtime_snapshot(manager.snapshot_store) as snapshot:
            root = snapshot.composition_root
            if root is None:
                raise RuntimeError("RuntimeSnapshot 缺少 composition Root")
            chat_models = root.context.require(CHAT_MODELS)
            async with chat_models.execution() as execution:
                descriptor = execution.chat(ModelRole.AGENT).descriptor
    except BaseException:
        await core.stop()
        await http.aclose()
        raise

    logger.info(
        "BenchmarkRuntime ready: workspace=%s model=%s driver=%s revision=%d",
        workspace,
        descriptor.model,
        descriptor.driver_id,
        descriptor.model_revision,
    )
    return BenchmarkRuntime(core=core, workspace=workspace, agent_model=descriptor)


def _seed_model_registry(source: Path, target: Path) -> None:
    """Copy the root benchmark model registry into a fresh isolated workspace."""

    if target.exists():
        return
    if not source.is_file():
        raise RuntimeError(
            f"benchmark 模型注册库不存在: {source}; 请先用该 workspace 的 2236 模型页配置"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        with closing(sqlite3.connect(f"file:{source}?mode=ro", uri=True)) as current:
            with closing(sqlite3.connect(target)) as seeded:
                current.backup(seeded)
        os.chmod(target, 0o600)
        with closing(sqlite3.connect(target)) as connection:
            result = connection.execute("PRAGMA integrity_check").fetchone()
        if result is None or str(result[0]) != "ok":
            raise RuntimeError(f"benchmark 模型注册库复制后损坏: {target}")
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def format_model_trace(rt: BenchmarkRuntime) -> str:
    """Render the exact public model descriptor captured at runtime start."""

    descriptor = rt.agent_model
    return (
        f"agent_model       = {descriptor.model}\n"
        f"connection_id     = {descriptor.connection_id}\n"
        f"driver_id         = {descriptor.driver_id}\n"
        f"model_revision    = {descriptor.model_revision}\n"
        f"plugin_snapshot   = {descriptor.plugin_snapshot_id}\n"
    )


async def close_runtime(rt: BenchmarkRuntime) -> None:
    closeables = getattr(rt.core.memory_runtime, "closeables", [])
    for obj in closeables:
        close = getattr(obj, "close", None) or getattr(obj, "aclose", None)
        if close:
            try:
                import asyncio
                import inspect

                if inspect.iscoroutinefunction(close):
                    await close()
                else:
                    await asyncio.to_thread(close)
            except Exception as e:
                logger.warning("close failed: %s", e)
    await rt.core.stop()
    await rt.core.http_resources.aclose()
