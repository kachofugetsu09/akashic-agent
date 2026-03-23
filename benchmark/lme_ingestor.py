"""
LongMemEval 生产链路 Ingestor（Phase 1）

目标：复用主链路 consolidation 实现，替换 benchmark 自定义离线提取。
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import openai

from agent.looping.consolidation import ConsolidationService
from agent.memory import MemoryStore
from memory2.memorizer import Memorizer
from memory2.post_response_worker import PostResponseMemoryWorker
from memory2.profile_extractor import ProfileFactExtractor
from memory2.store import MemoryStore2

logger = logging.getLogger(__name__)

LME_SCOPE_CHANNEL = "lme_benchmark"


class _MultiProgressBoard:
    """并发 worker 多行进度板。"""

    _lock = threading.Lock()
    _order: list[str] = []
    _rows: dict[str, str] = {}
    _started = False

    @classmethod
    def _enabled(cls) -> bool:
        return sys.stderr.isatty()

    @classmethod
    def _register_locked(cls, slot: str) -> None:
        if slot in cls._order:
            return
        cls._order.append(slot)
        cls._rows[slot] = ""
        sys.stderr.write("\n")
        cls._started = True

    @classmethod
    def _render_locked(cls) -> None:
        if not cls._started or not cls._enabled() or not cls._order:
            return
        line_count = len(cls._order)
        sys.stderr.write(f"\x1b[{line_count}A")
        for slot in cls._order:
            text = cls._rows.get(slot, "")
            sys.stderr.write(f"\r\x1b[2K{text}\n")
        sys.stderr.flush()

    @classmethod
    def update(cls, slot: str, text: str) -> None:
        if not cls._enabled():
            return
        with cls._lock:
            cls._register_locked(slot)
            cls._rows[slot] = text
            cls._render_locked()


class _ProgressLine:
    """终端单行动态进度条（类似 uv 风格）。"""

    _SPIN = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(
        self,
        total: int,
        *,
        enabled: bool = True,
        slot: str = "",
        label: str = "ingest",
        min_interval_s: float = 0.12,
    ) -> None:
        self._total = max(1, int(total))
        self._enabled = bool(enabled and sys.stderr.isatty())
        self._slot = slot.strip()
        self._label = label.strip() or "ingest"
        self._min_interval_s = max(0.05, float(min_interval_s))
        self._start = time.monotonic()
        self._tick = 0
        self._last = 0.0

    def update(self, done: int, *, pending_post: int = 0) -> None:
        if not self._enabled:
            return
        now = time.monotonic()
        if now - self._last < self._min_interval_s and done < self._total:
            return
        self._last = now
        self._tick = (self._tick + 1) % len(self._SPIN)
        pct = min(100.0, (done / self._total) * 100.0)
        elapsed = max(0.001, now - self._start)
        speed = done / elapsed
        bar_len = 24
        fill = int((done / self._total) * bar_len)
        bar = "█" * fill + "░" * (bar_len - fill)
        body = (
            f"{self._SPIN[self._tick]} {self._label} [{bar}] {pct:6.2f}% "
            f"{done}/{self._total} turns  {speed:5.1f} t/s  post_pending={pending_post}"
        )
        if self._slot:
            _MultiProgressBoard.update(self._slot, body)
            return
        sys.stderr.write(f"\r{body}")
        sys.stderr.flush()

    def close(self) -> None:
        if not self._enabled:
            return
        self._last = 0.0
        self.update(self._total, pending_post=0)
        if self._slot:
            return
        sys.stderr.write("\n")
        sys.stderr.flush()


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


class _ScopedMemorizerAdapter:
    """为 post-response 写入补齐 benchmark scope，避免跨 question 污染。"""

    def __init__(self, memorizer: Memorizer, scope_channel: str) -> None:
        self._memorizer = memorizer
        self._scope_channel = scope_channel

    async def save_item(
        self,
        summary: str,
        memory_type: str,
        extra: dict,
        source_ref: str,
        happened_at: str | None = None,
    ) -> str:
        payload = dict(extra or {})
        # 1. 若上游未写 scope，再按 source_ref 补齐 question 作用域。
        if not payload.get("scope_channel"):
            payload["scope_channel"] = self._scope_channel
        if not payload.get("scope_chat_id"):
            qid = self._extract_question_id(source_ref)
            if qid:
                payload["scope_chat_id"] = qid
        # 2. 复用原 memorizer 保存，保持去重/强化逻辑不变。
        return await self._memorizer.save_item(
            summary=summary,
            memory_type=memory_type,
            extra=payload,
            source_ref=source_ref,
            happened_at=happened_at,
        )

    def supersede_batch(self, ids: list[str]) -> None:
        self._memorizer.supersede_batch(ids)

    @staticmethod
    def _extract_question_id(source_ref: str) -> str:
        text = str(source_ref or "")
        if not text.startswith("lme:"):
            return ""
        parts = text.split(":")
        return parts[1] if len(parts) >= 2 else ""


class _ConsolidationRunner:
    """最小运行器：封装 ConsolidationService。"""

    def __init__(
        self,
        *,
        memory_port: _MemoryPortAdapter,
        provider: _ProviderAdapter,
        model: str,
        memory_window: int,
        profile_extractor: ProfileFactExtractor | None,
    ) -> None:
        self.memory_window = memory_window
        self._service = ConsolidationService(
            memory_port=memory_port,
            provider=provider,
            model=model,
            memory_window=memory_window,
            profile_extractor=profile_extractor,
        )

    async def _consolidate_memory(
        self,
        session,
        archive_all: bool = False,
        await_vector_store: bool = False,
    ) -> None:
        await self._service.consolidate(
            session,
            archive_all=archive_all,
            await_vector_store=await_vector_store,
        )


class LMEProductionIngestor:
    """按主链路 consolidation 对 LME haystack 进行写入。"""

    def __init__(
        self,
        *,
        workspace: Path,
        store: MemoryStore2,
        embedder,
        retriever,
        light_client: openai.OpenAI,
        light_model: str,
        memory_window: int,
        post_worker_concurrency: int = 4,
        progress_enabled: bool = True,
        progress_slot: str = "",
        progress_min_interval_s: float = 0.12,
    ) -> None:
        self._store = store
        self._memory_store = MemoryStore(workspace)
        self._provider = _ProviderAdapter(light_client, light_model)
        self._memorizer = Memorizer(store, embedder)
        self._profile_extractor = ProfileFactExtractor(
            llm_client=self._provider,
            model=light_model,
            max_tokens=600,
            timeout_ms=5000,
        )
        self._runner = _ConsolidationRunner(
            memory_port=_MemoryPortAdapter(self._memory_store, self._memorizer),
            provider=self._provider,
            model=light_model,
            memory_window=memory_window,
            profile_extractor=self._profile_extractor,
        )
        self._post_worker = PostResponseMemoryWorker(
            memorizer=_ScopedMemorizerAdapter(self._memorizer, LME_SCOPE_CHANNEL),
            retriever=retriever,
            light_provider=self._provider,
            light_model=light_model,
            tagger=None,
            profile_extractor=self._profile_extractor,
            profile_supersede_enabled=True,
            observe_writer=None,
        )
        self._post_worker_concurrency = max(1, int(post_worker_concurrency))
        self._progress_enabled = bool(progress_enabled)
        self._progress_slot = progress_slot.strip()
        self._progress_min_interval_s = max(0.05, float(progress_min_interval_s))

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

        # 2. 按 turn 顺序回放：每条消息后检查 consolidation；每轮 user/assistant 后立刻跑 post-response worker。
        total_turns = sum(
            1
            for sess in haystack_sessions
            for turn in sess
            if str(turn.get("role", "")).strip().lower() in {"user", "assistant"}
            and str(turn.get("content", "")).strip()
        )
        progress = _ProgressLine(
            total_turns,
            enabled=self._progress_enabled,
            slot=self._progress_slot,
            label=f"{self._progress_slot or 'ingest'}:{question_id[:8]}",
            min_interval_s=self._progress_min_interval_s,
        )
        done_turns = 0
        post_sem = asyncio.Semaphore(self._post_worker_concurrency)
        post_tasks: list[asyncio.Task] = []
        post_backlog_limit = self._post_worker_concurrency * 3
        consolidation_task: asyncio.Task | None = None
        try:
            for sess_idx, turns in enumerate(haystack_sessions):
                date_str = haystack_dates[sess_idx] if sess_idx < len(haystack_dates) else ""
                pending_users: list[str] = []
                for turn_idx, turn in enumerate(turns):
                    msg = self._build_message(question_id, sess_idx, turn_idx, turn, date_str)
                    if msg is None:
                        continue
                    session.messages.append(msg)
                    done_turns += 1
                    consolidation_task = await self._schedule_consolidation_if_needed(
                        session=session,
                        task=consolidation_task,
                    )
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        pending_users.append(content)
                        progress.update(done_turns, pending_post=len(post_tasks))
                        continue
                    if role == "assistant" and pending_users:
                        merged_user = "\n".join(pending_users).strip()
                        post_tasks = [t for t in post_tasks if not t.done()]
                        while len(post_tasks) >= post_backlog_limit:
                            done, pending = await asyncio.wait(
                                post_tasks,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            post_tasks = list(pending)
                            progress.update(done_turns, pending_post=len(post_tasks))
                        post_tasks.append(
                            asyncio.create_task(
                                self._run_post_worker_once(
                                    sem=post_sem,
                                    question_id=question_id,
                                    sess_idx=sess_idx,
                                    turn_idx=turn_idx,
                                    user_msg=merged_user,
                                    assistant_msg=content,
                                )
                    )
                        )
                        pending_users.clear()
                    progress.update(done_turns, pending_post=len(post_tasks))

            # 3. 对未归档尾段做一次 archive_all 收尾，避免最后窗口消息漏写。
            if consolidation_task is not None:
                await consolidation_task
            # 3.1 若仍有超窗 backlog，按主窗口补齐几轮 consolidation，避免尾段一次性压缩过宽。
            while self._backlog_size(session) > int(self._runner.memory_window):
                await self._runner._consolidate_memory(
                    session,
                    await_vector_store=True,
                )
            tail_session = self._build_tail_session(session)
            if tail_session is not None:
                await self._runner._consolidate_memory(
                    tail_session,
                    archive_all=True,
                    await_vector_store=True,
                )
            if post_tasks:
                while post_tasks:
                    done, pending = await asyncio.wait(
                        post_tasks,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    post_tasks = list(pending)
                    progress.update(done_turns, pending_post=len(post_tasks))
        finally:
            progress.close()

        # 4. 返回该 question 在 memory2 中的 event 数，供日志统计。
        return self._count_scope_events(question_id)

    async def _schedule_consolidation_if_needed(
        self,
        *,
        session: SimpleNamespace,
        task: asyncio.Task | None,
    ) -> asyncio.Task | None:
        # 1. 先回收已完成任务，避免悬挂异常。
        if task is not None and task.done():
            await task
            task = None
        # 2. 回放速度远高于线上时，若 backlog 过大则先等待在跑任务，防止一次压缩覆盖过长窗口。
        if task is not None and self._backlog_size(session) > int(self._runner.memory_window) * 2:
            await task
            task = None
        # 3. 再按窗口 backlog 调度下一次 consolidation。
        if task is None and self._backlog_size(session) > int(self._runner.memory_window):
            task = asyncio.create_task(
                self._runner._consolidate_memory(
                    session,
                    await_vector_store=True,
                )
            )
        # 4. 返回最新任务句柄供下一轮复用。
        return task

    @staticmethod
    def _backlog_size(session: SimpleNamespace) -> int:
        consolidated = int(getattr(session, "last_consolidated", 0) or 0)
        return max(0, len(session.messages) - consolidated)

    async def _run_post_worker_once(
        self,
        *,
        sem: asyncio.Semaphore,
        question_id: str,
        sess_idx: int,
        turn_idx: int,
        user_msg: str,
        assistant_msg: str,
    ) -> None:
        async with sem:
            await self._post_worker.run(
                user_msg=user_msg,
                agent_response=assistant_msg,
                tool_chain=[],
                source_ref=f"lme:{question_id}:{sess_idx}:{turn_idx}@post_response",
                session_key=f"lme:{question_id}",
            )

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

    def _build_message(
        self,
        question_id: str,
        sess_idx: int,
        turn_idx: int,
        turn: dict,
        session_date: str,
    ) -> dict | None:
        day = (session_date or "1970-01-01").strip()
        role = str(turn.get("role", "")).strip().lower()
        content = str(turn.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            return None
        hh = (turn_idx // 60) % 24
        mm = turn_idx % 60
        return {
            "id": f"lme:{question_id}:{sess_idx}:{turn_idx}",
            "role": role,
            "content": content,
            "timestamp": f"{day} {hh:02d}:{mm:02d}",
        }

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
