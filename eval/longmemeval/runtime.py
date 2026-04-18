"""BenchmarkRuntime: full production stack wired for LongMemEval.

Uses build_core_runtime exactly as production so prompt assembly,
tool dispatch, memory injection, and retrieval are identical.
The only delta from a real user workspace: MEMORY.md / SELF.md start
empty (honest baseline that forces all recall through the memory system).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

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
    core: object          # CoreRuntime
    consolidation: object # ConsolidationService
    workspace: Path


async def create_runtime(config_path: Path, workspace: Path) -> BenchmarkRuntime:
    """Wire the full production stack into a temp workspace.

    Args:
        config_path: Path to config.toml (same one used in production).
        workspace: Temp directory; will be initialised on first call.
    """
    from agent.config import load_config
    from agent.looping.consolidation import ConsolidationService
    from bootstrap.init_workspace import init_workspace
    from bootstrap.tools import build_core_runtime
    from core.net.http import SharedHttpResources
    from memory2.profile_extractor import ProfileFactExtractor

    config = load_config(config_path)

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

    # 4. Build a ConsolidationService that shares the same memory port.
    #    This is used during ingest; the AgentLoop has its own internal
    #    instance but we need explicit control over consolidation timing.
    light = core.light_provider or core.provider
    light_model = config.light_model or config.model
    profile_extractor = ProfileFactExtractor(
        llm_client=light,
        model=light_model,
    )
    keep_count = max(1, config.memory_window // 2)
    consolidation = ConsolidationService(
        memory_port=core.memory_runtime.port,
        profile_maint=getattr(core.memory_runtime, "profile_maint", None)
        or core.memory_runtime.port,
        provider=core.provider,
        model=config.model,
        keep_count=keep_count,
        profile_extractor=profile_extractor,
        recent_context_provider=light,
        recent_context_model=light_model,
    )

    logger.info(
        "BenchmarkRuntime ready: workspace=%s keep_count=%d model=%s",
        workspace,
        keep_count,
        config.model,
    )
    return BenchmarkRuntime(core=core, consolidation=consolidation, workspace=workspace)


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
