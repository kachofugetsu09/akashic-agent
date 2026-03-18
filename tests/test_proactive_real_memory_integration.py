from __future__ import annotations

import copy
import importlib
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.config import load_config
from agent.provider import LLMProvider
from agent.tools.registry import ToolRegistry
from bootstrap.memory import build_memory_runtime
from core.net.http import (
    SharedHttpResources,
    clear_default_shared_http_resources,
    configure_default_shared_http_resources,
)
from feeds.base import FeedItem
from proactive.composer import Composer
from proactive.config import ProactiveConfig
from proactive.decide import DefaultDecidePort
from proactive.event import GenericContentEvent
from proactive.item_id import compute_item_id, compute_source_key
from proactive.judge import Judge
from proactive.loop_helpers import _semantic_text
from proactive.ports import DefaultMemoryRetrievalPort
from proactive.state import ProactiveStateStore
from proactive.tick import DecisionContext, ProactiveEngine

_WORKSPACE = Path("/home/huashen/.akasic/workspace")
_OBSERVE_DB = _WORKSPACE / "observe" / "observe.db"
_TRACE_TICK_ID = "b2e3a5f46b544ff8a839fcd333aae3ae"


def _resolve_real_inputs() -> tuple[object | None, Path | None, str]:
    if not bool(int(__import__("os").environ.get("AKASIC_RUN_REAL_MEMORY_TEST", "0"))):
        return None, None, "AKASIC_RUN_REAL_MEMORY_TEST 未开启"
    db_path = Path(
        __import__("os").environ.get(
            "AKASIC_REAL_MEMORY_DB",
            str(_WORKSPACE / "memory" / "memory2.db"),
        )
    )
    if not db_path.exists():
        return None, None, f"memory2.db 不存在: {db_path}"
    if not _OBSERVE_DB.exists():
        return None, None, f"observe.db 不存在: {_OBSERVE_DB}"
    cfg = load_config("config.json")
    cfg.memory_v2.db_path = str(db_path)
    return cfg, db_path, ""


def _patch_real_openai() -> None:
    # 1. pytest 环境里 tests/conftest.py 会先塞入 openai stub，这里显式切回真实包。
    for name in list(sys.modules):
        if name == "openai" or name.startswith("openai."):
            del sys.modules[name]
    real_openai = importlib.import_module("openai")
    import agent.provider as provider_mod

    provider_mod.AsyncOpenAI = real_openai.AsyncOpenAI


def _load_trace_row() -> dict:
    # 1. 从 observe.db 读取上次真实 trace 的候选和关键快照。
    conn = sqlite3.connect(str(_OBSERVE_DB))
    try:
        cur = conn.cursor()
        row = cur.execute(
            """
            select session_key, ts, candidates_json, sense_result_json,
                   proactive_sent_24h, fresh_items_24h
            from proactive_decisions
            where tick_id = ?
            limit 1
            """,
            (_TRACE_TICK_ID,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise AssertionError(f"未找到 tick_id={_TRACE_TICK_ID} 的真实 trace")
    return {
        "session_key": row[0],
        "ts": row[1],
        "candidates": json.loads(row[2] or "[]"),
        "sense": json.loads(row[3] or "{}"),
        "sent_24h": int(row[4] or 0),
        "fresh_items_24h": int(row[5] or 0),
    }


def _to_events(candidates: list[dict]) -> tuple[list[GenericContentEvent], list[tuple[str, str]]]:
    # 1. 把 observe.db 里的 candidates_json 还原成当前主动链路可消费的事件。
    events: list[GenericContentEvent] = []
    entries: list[tuple[str, str]] = []
    for payload in candidates:
        event = GenericContentEvent.from_mcp_payload(payload)
        item = event.to_feed_item()
        events.append(event)
        entries.append((compute_source_key(item), compute_item_id(item)))
    return events, entries


def _build_provider(cfg: object, *, use_light: bool) -> LLMProvider:
    return LLMProvider(
        api_key=(cfg.light_api_key if use_light else cfg.api_key) or cfg.api_key,
        base_url=(cfg.light_base_url if use_light else cfg.base_url) or cfg.base_url,
        system_prompt=cfg.system_prompt,
        extra_body=cfg.extra_body,
    )


def _format_items(items: list[FeedItem]) -> str:
    lines: list[str] = []
    for item in items[:4]:
        lines.append(f"- {item.title or '(无标题)'}（{item.source_name or 'unknown'}）")
        if item.content:
            lines.append((item.content or "").strip()[:220])
    return "\n".join(lines)


def _format_recent(recent: list[dict]) -> str:
    return "\n".join(
        f"- {str(msg.get('role', 'user'))}: {str(msg.get('content', '')).strip()[:120]}"
        for msg in recent[-5:]
        if str(msg.get("content", "")).strip()
    )


def _build_decide_port(cfg: object, provider: LLMProvider) -> DefaultDecidePort:
    composer = Composer(
        provider=provider,
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        format_items=_format_items,
        format_recent=_format_recent,
    )
    judge = Judge(
        provider=provider,
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        format_recent=_format_recent,
        cfg=cfg.proactive,
    )
    return DefaultDecidePort(
        randomize_fn=lambda decision: (decision, 0.0),
        source_key_fn=compute_source_key,
        item_id_fn=compute_item_id,
        semantic_text_fn=_semantic_text,
        semantic_text_max_chars=cfg.proactive.semantic_dedupe_text_max_chars,
        composer=composer,
        judge=judge,
    )


def _build_ctx(trace_row: dict) -> DecisionContext:
    ctx = DecisionContext()
    sense = ctx.ensure_sense()
    fetch = ctx.ensure_fetch()
    score = ctx.ensure_score()
    act = ctx.ensure_act()

    # 1. 还原 tick 基础状态与 sense/score 快照。
    ctx.state.tick_id = _TRACE_TICK_ID
    ctx.state.session_key = trace_row["session_key"]
    ctx.state.now_utc = datetime.fromisoformat(trace_row["ts"])
    sense.sleep_ctx = SimpleNamespace(
        state=trace_row["sense"].get("sleep_state", "awake"),
        available=trace_row["sense"].get("sleep_available", True),
        probability=trace_row["sense"].get("sleep_prob"),
        prob=trace_row["sense"].get("sleep_prob"),
        data_lag_min=trace_row["sense"].get("sleep_data_lag_min", 0),
    )
    sense.recent = []
    sense.health_events = []
    sense.energy = float(trace_row["sense"].get("energy", 0.5) or 0.5)
    sense.interruptibility = float(trace_row["sense"].get("interruptibility", 1.0) or 1.0)
    sense.interrupt_factor = float(trace_row["sense"].get("interrupt_factor", 1.0) or 1.0)
    sense.sleep_mod = float(trace_row["sense"].get("sleep_mod", 1.0) or 1.0)
    sense.interrupt_detail = {"f_reply": 1.0, "f_activity": 1.0, "f_fatigue": 1.0}
    sense.de = 0.0
    sense.dr = 1.0
    score.pre_score = 0.0
    score.base_score = 0.0
    score.draw_score = 0.0
    score.dc = 1.0
    score.is_crisis = False
    score.sent_24h = trace_row["sent_24h"]
    score.fresh_items_24h = trace_row["fresh_items_24h"]
    act.high_events = []

    # 2. 注入真实 trace 里的候选事件。
    fetch.new_items, fetch.new_entries = _to_events(trace_row["candidates"])
    return ctx


@pytest.mark.asyncio
async def test_real_memory_retrieval_with_trace_replay_writes_diagnostic_artifact(tmp_path: Path):
    cfg, db_path, reason = _resolve_real_inputs()
    if cfg is None or db_path is None:
        pytest.skip(reason)

    _patch_real_openai()
    trace_row = _load_trace_row()
    resources = SharedHttpResources()
    configure_default_shared_http_resources(resources)
    try:
        # 1. 用真实 config / provider / memory2.db 还原主动链路运行时。
        provider = _build_provider(cfg, use_light=False)
        light_provider = _build_provider(cfg, use_light=True)
        memory_runtime = build_memory_runtime(
            config=cfg,
            workspace=_WORKSPACE,
            tools=ToolRegistry(),
            provider=provider,
            light_provider=light_provider,
            http_resources=resources,
            observe_writer=None,
        )

        # 2. 手工拼一个最小 engine，只跑 memory/compose/judge，不进入真实发送。
        engine = ProactiveEngine.__new__(ProactiveEngine)
        engine._cfg = copy.deepcopy(cfg.proactive)
        engine._decide = _build_decide_port(cfg, provider)
        engine._memory_retrieval = DefaultMemoryRetrievalPort(
            cfg=engine._cfg,
            memory=memory_runtime.port,
            item_id_fn=compute_item_id,
            light_provider=light_provider,
            light_model=cfg.light_model or cfg.model,
        )
        engine._state = ProactiveStateStore(tmp_path / "state.json")
        engine._prefetch_fetcher = None
        engine._sense = SimpleNamespace(collect_recent_proactive=lambda n=5: [])

        ctx = _build_ctx(trace_row)
        memory_result = await engine._retrieve_memory(ctx)
        compose_result = await engine._compose(ctx)
        if compose_result.proceed:
            guard_result = await engine._judge_and_guard(ctx)
            reason_code = guard_result.reason_code
            should_send = guard_result.should_send
        else:
            reason_code = compose_result.reason_code
            should_send = False
        decide = ctx.ensure_decide()
        act = ctx.ensure_act()

        payload = {
            "tick_id": _TRACE_TICK_ID,
            "artifact_path": str(tmp_path / "real_trace_replay_result.json"),
            "db_path": str(db_path),
            "feed_titles": [e.title for e in act.compose_items] or [it.title for it in engine._feed_items(ctx.ensure_fetch().new_items)],
            "memory_query": decide.memory_query,
            "preference_block": decide.preference_block,
            "selected_titles": [item.title for item in act.compose_items],
            "compose_entries": act.compose_entries,
            "final_message": decide.decision_message,
            "should_send": should_send,
            "reason_code": reason_code,
            "decision_mode": "compose_judge",
            "memory_fallback_reason": memory_result.fallback_reason,
            "judge_final_score": decide.judge_final_score,
            "judge_vetoed_by": decide.judge_vetoed_by,
        }
        result_path = tmp_path / "real_trace_replay_result.json"
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\n===== real_trace_replay_result =====")
        print(f"artifact_path: {result_path}")
        print(f"tick_id: {_TRACE_TICK_ID}")
        print(f"db_path: {db_path}")
        print("\n[selected_titles]")
        for title in payload["selected_titles"]:
            print(f"- {title}")
        print("\n[memory_query]")
        print(payload["memory_query"])
        print("\n[preference_block]")
        print(payload["preference_block"])
        print("\n[compose_entries]")
        for entry in payload["compose_entries"]:
            print(f"- {entry}")
        print("\n[final_message]")
        print(payload["final_message"])
        print(f"\n[should_send]\n{payload['should_send']}")
        print(f"\n[reason_code]\n{payload['reason_code']}")
        print("===== end real_trace_replay_result =====")

        assert result_path.exists()
        assert ctx.ensure_fetch().new_items, "真实 trace 候选为空"
    finally:
        clear_default_shared_http_resources(resources)
        await resources.aclose()
        if "memory_runtime" in locals():
            await memory_runtime.aclose()
