"""
LongMemEval 生产链路 Ingestor（Phase 1）

目标：复用主链路 consolidation 实现，替换 benchmark 自定义离线提取。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import openai

from agent.looping.consolidation import AgentLoopConsolidationMixin
from agent.memory import MemoryStore
from memory2.memorizer import Memorizer
from memory2.profile_extractor import ProfileFactExtractor
from memory2.store import MemoryStore2

logger = logging.getLogger(__name__)

LME_SCOPE_CHANNEL = "lme_benchmark"


@dataclass
class _ChatResponse:
    content: str


class _ProviderAdapter:
    """将 openai.OpenAI 同步客户端包装为生产 LLMProvider 异步接口。"""

    def __init__(self, client: openai.OpenAI, default_model: str) -> None:
        self._client = client
        self._default_model = default_model

    async def chat(
        self,
        messages: list[dict],
        tools: list | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        **_kwargs: Any,
    ) -> _ChatResponse:
        response = await asyncio.to_thread(
            self._client.chat.completions.create,
            model=model or self._default_model,
            messages=messages,
            max_tokens=max_tokens or 1024,
            temperature=0.1,
        )
        return _ChatResponse(content=response.choices[0].message.content or "")


class _MemoryPortAdapter:
    """将 MemoryStore + Memorizer 组合成 consolidation 所需的 memory_port 接口。"""

    def __init__(self, memory_store: MemoryStore, memorizer: Memorizer) -> None:
        self._memory_store = memory_store
        self._memorizer = memorizer

    def read_long_term(self) -> str:
        return self._memory_store.read_long_term()

    def append_history_once(self, *args, **kwargs) -> bool:
        return self._memory_store.append_history_once(*args, **kwargs)

    def append_pending_once(self, *args, **kwargs) -> bool:
        return self._memory_store.append_pending_once(*args, **kwargs)

    async def save_from_consolidation(self, *args, **kwargs) -> None:
        await self._memorizer.save_from_consolidation(*args, **kwargs)

    async def save_item(self, *args, **kwargs) -> str:
        return await self._memorizer.save_item(*args, **kwargs)


class _ConsolidationRunner(AgentLoopConsolidationMixin):
    """最小运行器：仅提供 _consolidate_memory 需要的属性。"""

    def __init__(
        self,
        *,
        memory_port: _MemoryPortAdapter,
        provider: _ProviderAdapter,
        model: str,
        memory_window: int,
        profile_extractor: ProfileFactExtractor | None,
    ) -> None:
        self._memory_port = memory_port
        self.provider = provider
        self.model = model
        self.memory_window = memory_window
        self._profile_extractor = profile_extractor


class LMEProductionIngestor:
    """按主链路 consolidation 对 LME haystack 进行写入。"""

    def __init__(
        self,
        *,
        workspace: Path,
        store: MemoryStore2,
        embedder,
        light_client: openai.OpenAI,
        light_model: str,
        memory_window: int,
    ) -> None:
        self._store = store
        self._memory_store = MemoryStore(workspace)
        self._profile_extractor = ProfileFactExtractor(
            llm_client=_ProviderAdapter(light_client, light_model),
            model=light_model,
            max_tokens=600,
            timeout_ms=5000,
        )
        self._runner = _ConsolidationRunner(
            memory_port=_MemoryPortAdapter(self._memory_store, Memorizer(store, embedder)),
            provider=_ProviderAdapter(light_client, light_model),
            model=light_model,
            memory_window=memory_window,
            profile_extractor=self._profile_extractor,
        )

    def ingest_question_sync(
        self,
        *,
        question_id: str,
        haystack_sessions: list[list[dict]],
        haystack_dates: list[str],
    ) -> int:
        return asyncio.run(
            self._ingest_question(
                question_id=question_id,
                haystack_sessions=haystack_sessions,
                haystack_dates=haystack_dates,
            )
        )

    async def _ingest_question(
        self,
        *,
        question_id: str,
        haystack_sessions: list[list[dict]],
        haystack_dates: list[str],
    ) -> int:
        # 1. 构造可增量 consolidation 的会话状态。
        session = SimpleNamespace(
            key=f"lme:{question_id}",
            messages=[],
            last_consolidated=0,
            _channel=LME_SCOPE_CHANNEL,
            _chat_id=question_id,
        )

        # 2. 按 session 顺序追加消息并执行真实 consolidation。
        for sess_idx, turns in enumerate(haystack_sessions):
            date_str = haystack_dates[sess_idx] if sess_idx < len(haystack_dates) else ""
            session.messages.extend(
                self._build_messages(question_id, sess_idx, turns, date_str)
            )
            await self._runner._consolidate_memory(session)

        # 3. 对未归档尾段做一次 archive_all 收尾，避免最后窗口消息漏写。
        tail_session = self._build_tail_session(session)
        if tail_session is not None:
            await self._runner._consolidate_memory(tail_session, archive_all=True)

        # 4. 返回该 question 在 memory2 中的 event 数，供日志统计。
        return self._count_scope_events(question_id)

    def _build_tail_session(self, session: SimpleNamespace) -> SimpleNamespace | None:
        # 1. 计算尾段起点：即最后一次增量 consolidation 处理到的位置。
        start = int(getattr(session, "last_consolidated", 0) or 0)
        if start >= len(session.messages):
            return None
        # 2. 仅截取未归档尾段，避免 archive_all 重复归档整个历史。
        tail_messages = list(session.messages[start:])
        if not tail_messages:
            return None
        # 3. 构造尾段临时会话，复用原 scope，交给 archive_all 一次性收尾。
        return SimpleNamespace(
            key=f"{session.key}:tail",
            messages=tail_messages,
            last_consolidated=0,
            _channel=getattr(session, "_channel", ""),
            _chat_id=getattr(session, "_chat_id", ""),
        )

    def _build_messages(
        self,
        question_id: str,
        sess_idx: int,
        turns: list[dict],
        session_date: str,
    ) -> list[dict]:
        messages: list[dict] = []
        day = (session_date or "1970-01-01").strip()
        for turn_idx, turn in enumerate(turns):
            role = str(turn.get("role", "")).strip().lower()
            content = str(turn.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            hh = (turn_idx // 60) % 24
            mm = turn_idx % 60
            messages.append(
                {
                    "id": f"lme:{question_id}:{sess_idx}:{turn_idx}",
                    "role": role,
                    "content": content,
                    "timestamp": f"{day} {hh:02d}:{mm:02d}",
                }
            )
        return messages

    def _count_scope_events(self, question_id: str) -> int:
        row = self._store._db.execute(
            """SELECT COUNT(*)
               FROM memory_items
               WHERE memory_type='event'
                 AND json_extract(extra_json, '$.scope_channel')=?
                 AND json_extract(extra_json, '$.scope_chat_id')=?""",
            (LME_SCOPE_CHANNEL, question_id),
        ).fetchone()
        return int(row[0] if row else 0)


__all__ = ["LMEProductionIngestor", "LME_SCOPE_CHANNEL"]
