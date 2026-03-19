"""
Akasic Benchmark on LongMemEval Dataset

使用 Akasic memory2（SQLite + 向量搜索）对 LongMemEval 数据集进行基准测试。
与 LoCoMo 的主要区别：
  - 对话格式为 user-assistant（贴近生产环境）
  - 每个 question 自带独立的 haystack_sessions
  - ingest 走主链路 consolidation（Phase 1）
  - retrieve 当前覆盖 event + profile + preference（Phase 1 主要写入 event）

下载数据：
  pip install huggingface_hub
  huggingface-cli download xiaowu0162/longmemeval --repo-type dataset --local-dir benchmark/data/longmemeval

用法（从 benchmark/ 目录运行）：
  cd benchmark/
  python run_longmemeval.py [options]

选项：
  --config PATH           config.json 路径（默认: ../config.json）
  --workspace PATH        benchmark 专用 workspace（默认: /tmp/akasic_benchmark/lme_parity/workspace）
  --db-path PATH          benchmark 专用 DB 路径（默认: /tmp/akasic_benchmark/lme_parity/lme_parity.db）
  --data PATH             longmemeval JSON 路径（默认: data/longmemeval/longmemeval_s.json）
  --max-samples N         最多处理 N 个 question（默认: 全部）
  --question-type TYPE    只测指定 question_type（逗号分隔，默认: 全部）
  --max-workers N         并发线程数（默认: 2）
  --skip-ingest           跳过 ingest（使用已有 DB）
  --use-flash             response/evaluate 走 light model（更快）
  --output PATH           结果 JSON 路径（默认: /tmp/akasic_benchmark/lme_parity/result_lme.json）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import dotenv

dotenv.load_dotenv()

_HERE = Path(__file__).parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config_loader import BenchmarkComponents, load_benchmark_components
from evaluate_agent import EvaluateAgent
from lme_ingestor import LMEProductionIngestor
from agent.policies.history_route import HistoryRoutePolicy
from memory2.hyde_enhancer import HyDEEnhancer
from memory2.injection_planner import retrieve_episodic, retrieve_procedure_items
from agent.config import load_config
from memory2.query_rewriter import QueryRewriter
from memory2.store import MemoryStore2
from memory2.sufficiency_checker import SufficiencyChecker
from memu.utils import setup_logging

logger = setup_logging(__name__, enable_flush=True)

LME_SCOPE_CHANNEL = "lme_benchmark"
BENCHMARK_BASE_DIR = Path("/tmp/akasic_benchmark")
DEFAULT_LME_RUN_DIR = BENCHMARK_BASE_DIR / "lme_parity"
DEFAULT_LME_WORKSPACE = DEFAULT_LME_RUN_DIR / "workspace"
DEFAULT_LME_DB = DEFAULT_LME_RUN_DIR / "lme_parity.db"
DEFAULT_LME_OUTPUT = DEFAULT_LME_RUN_DIR / "result_lme.json"

QUESTION_TYPE_NAMES = {
    "single-session-user": "SS-User",
    "single-session-assistant": "SS-Asst",
    "single-session-preference": "SS-Pref",
    "multi-session": "Multi",
    "knowledge-update": "KnowUpd",
    "temporal-reasoning": "Temporal",
    "abstention": "Abstain",
}


@dataclass
class _ChatResponse:
    content: str


class _LightProviderAdapter:
    """将 openai 客户端包装成 QueryRewriter/HyDE/Sufficiency 可用的 chat 接口。"""

    def __init__(self, client, default_model: str) -> None:
        self._client = client
        self._model = default_model

    async def chat(
        self,
        messages: list[dict],
        tools: list | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        **_kwargs,
    ) -> _ChatResponse:
        resp = await asyncio.to_thread(
            self._client.chat.completions.create,
            model=model or self._model,
            messages=messages,
            max_tokens=max_tokens or 256,
            temperature=0.1,
        )
        return _ChatResponse(content=resp.choices[0].message.content or "")


class _MemoryPlannerAdapter:
    """把 Retriever 适配为 injection_planner 所需的最小 MemoryPort 接口。"""

    def __init__(
        self,
        retriever,
        *,
        scope_channel: str = "",
        scope_chat_id: str = "",
        force_scope: bool = False,
    ) -> None:
        self._retriever = retriever
        self._scope_channel = scope_channel
        self._scope_chat_id = scope_chat_id
        self._force_scope = force_scope

    async def retrieve_related(
        self,
        query: str,
        *,
        memory_types: list[str] | None = None,
        top_k: int | None = None,
        scope_channel: str | None = None,
        scope_chat_id: str | None = None,
        require_scope_match: bool = False,
    ) -> list[dict]:
        effective_scope_channel = scope_channel or self._scope_channel
        effective_scope_chat_id = scope_chat_id or self._scope_chat_id
        effective_require_scope = require_scope_match or (
            self._force_scope and bool(effective_scope_channel and effective_scope_chat_id)
        )
        return await self._retriever.retrieve(
            query=query,
            memory_types=memory_types,
            top_k=top_k,
            scope_channel=effective_scope_channel or None,
            scope_chat_id=effective_scope_chat_id or None,
            require_scope_match=effective_require_scope,
        )

    def select_for_injection(self, items: list[dict]) -> list[dict]:
        return self._retriever.select_for_injection(items)

    def build_injection_block(self, items: list[dict]) -> tuple[str, list[str]]:
        return self._retriever.build_injection_block(items)


def _resolve_production_db(config_path: str) -> Path:
    """解析主 Agent 的生产 memory2 DB 路径。"""
    config = load_config(config_path)
    if config.memory_v2.db_path:
        return Path(config.memory_v2.db_path).expanduser().resolve()
    return (Path.home() / ".akasic" / "workspace" / "memory" / "memory2.db").resolve()


def _prepare_phase0_guardrails(
    config_path: str,
    db_path: str,
    workspace: str,
    *,
    skip_ingest: bool,
) -> tuple[Path, Path]:
    benchmark_root = BENCHMARK_BASE_DIR.resolve()
    db = Path(db_path).expanduser().resolve()
    ws = Path(workspace).expanduser().resolve()
    prod_db = _resolve_production_db(config_path)

    # 1. 启动硬校验：禁止命中生产 DB。
    if db == prod_db:
        raise SystemExit(f"拒绝启动：benchmark db_path 指向生产库：{db}")

    # 2. 启动硬校验：专用目录必须位于 /tmp/akasic_benchmark 下，避免误删。
    if benchmark_root not in db.parents or benchmark_root not in ws.parents:
        raise SystemExit("拒绝启动：db_path 与 workspace 必须位于 /tmp/akasic_benchmark 下")

    run_root = ws.parent
    if run_root == benchmark_root or run_root != db.parent:
        raise SystemExit("拒绝启动：workspace 与 db_path 需位于同一专用子目录，且不能是 /tmp/akasic_benchmark 根目录")

    # 3. 按运行模式处理目录。
    # 3.1 skip-ingest=false：清理 run_root，保证本次运行全新隔离。
    # 3.2 skip-ingest=true：保留 run_root，复用已有 DB。
    if not skip_ingest:
        shutil.rmtree(run_root, ignore_errors=True)
    ws.mkdir(parents=True, exist_ok=True)
    db.parent.mkdir(parents=True, exist_ok=True)
    if skip_ingest and not db.exists():
        raise SystemExit(f"拒绝启动：--skip-ingest 需要已有 DB，但未找到：{db}")
    return db, ws


def _assert_output_in_run_root(output_path: str, run_root: Path) -> Path:
    out = Path(output_path).expanduser().resolve()
    if run_root not in out.parents:
        raise SystemExit(f"拒绝启动：output 必须位于本次 run 目录下：{run_root}")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


# ─── Ingest ───────────────────────────────────────────────────────────

class LMEMemAgent:
    def __init__(self, components: BenchmarkComponents, workspace: Path, memory_window: int) -> None:
        self._store = components.store
        self._ingestor = LMEProductionIngestor(
            workspace=workspace,
            store=components.store,
            embedder=components.embedder,
            retriever=components.retriever,
            light_client=components.light_llm_client,
            light_model=components.light_model,
            memory_window=memory_window,
        )

    def ingest_haystack(
        self,
        question_id: str,
        haystack_sessions: list[list[dict]],
        haystack_dates: list[str],
    ) -> int:
        """
        摄取一个 question 的所有 haystack sessions，返回该 question 的 event 条目数。
        """
        total_sessions = len(haystack_sessions)
        logger.info("  consolidation ingest %d sessions...", total_sessions)
        try:
            saved = self._ingestor.ingest_question_sync(
                question_id=question_id,
                haystack_sessions=haystack_sessions,
                haystack_dates=haystack_dates,
            )
            logger.info("  consolidation ingest done: qid=%s events=%d", question_id, saved)
            return saved
        except Exception as exc:
            logger.warning("consolidation ingest failed qid=%s: %s", question_id, exc)
            return 0

    def clear_question_memory(self, question_id: str) -> None:
        try:
            # 1. 清理 memory2 中该 question 的条目，确保重跑时范围干净。
            self._store._db.execute(
                """DELETE FROM memory_items
                   WHERE json_extract(extra_json,'$.scope_channel')=?
                     AND json_extract(extra_json,'$.scope_chat_id')=?""",
                (LME_SCOPE_CHANNEL, question_id),
            )
            # 2. 同步清理 consolidation_events，避免 source_ref 幂等索引残留影响重跑。
            self._store._db.execute(
                """DELETE FROM consolidation_events
                   WHERE source_ref LIKE ?""",
                (f'%"lme:{question_id}:%',),
            )
            # 3. 提交删除事务。
            self._store._db.commit()
        except Exception as exc:
            logger.warning("clear_question_memory failed qid=%s: %s", question_id, exc)

    def count_superseded(self, question_id: str) -> int:
        row = self._store._db.execute(
            """SELECT COUNT(*)
               FROM memory_items
               WHERE status='superseded'
                 AND json_extract(extra_json,'$.scope_channel')=?
                 AND json_extract(extra_json,'$.scope_chat_id')=?""",
            (LME_SCOPE_CHANNEL, question_id),
        ).fetchone()
        return int(row[0] if row else 0)


# ─── Response ────────────────────────────────────────────────────────

class LMEResponseAgent:
    """从记忆库检索后，生成对 LongMemEval 问题的回答。"""

    def __init__(self, components: BenchmarkComponents, runtime_cfg, use_hyde: bool = False) -> None:
        self._retriever = components.retriever
        self._llm = components.llm_client
        self._model = components.model
        self._top_k_procedure = runtime_cfg.memory_v2.top_k_procedure
        self._top_k_history = runtime_cfg.memory_v2.top_k_history
        self._light_provider = _LightProviderAdapter(
            components.light_llm_client,
            components.light_model,
        )
        self._query_rewriter = (
            QueryRewriter(
                llm_client=self._light_provider,
                model=components.light_model,
                max_tokens=runtime_cfg.memory_v2.gate_max_tokens,
                timeout_ms=runtime_cfg.memory_v2.gate_llm_timeout_ms,
            )
            if runtime_cfg.memory_v2.route_intention_enabled
            else None
        )
        self._history_route = HistoryRoutePolicy(
            light_provider=self._light_provider,
            light_model=components.light_model,
            enabled=runtime_cfg.memory_v2.route_intention_enabled,
            llm_timeout_ms=runtime_cfg.memory_v2.gate_llm_timeout_ms,
            max_tokens=runtime_cfg.memory_v2.gate_max_tokens,
        )
        self._sufficiency_checker = (
            SufficiencyChecker(
                llm_client=self._light_provider,
                model=components.light_model,
            )
            if runtime_cfg.memory_v2.sufficiency_check_enabled
            else None
        )
        self._hyde_enhancer = (
            HyDEEnhancer(
                light_provider=self._light_provider,
                light_model=components.light_model,
                timeout_s=max(0.5, runtime_cfg.memory_v2.hyde_timeout_ms / 1000.0),
            )
            if use_hyde
            else None
        )

    def answer_question(
        self,
        question: str,
        question_id: str,
        haystack_sessions: list[list[dict]],
    ) -> dict:
        try:
            logger.info("  retrieving for: %s", question[:80])
            recent_history = self._format_gate_history(haystack_sessions, max_turns=3)
            runtime_metadata = self._build_runtime_metadata()
            items, context_text, retrieve_trace = asyncio.run(
                self._retrieve(
                    question=question,
                    question_id=question_id,
                    recent_history=recent_history,
                    runtime_metadata=runtime_metadata,
                )
            )
            logger.info("  retrieved %d items (selected)", len(items))
            if not context_text:
                context_text = "\n".join(item.get("summary", "") for item in items)
            logger.info("  context_text len=%d", len(context_text))
            logger.info("  calling LLM for answer...")
            answer = self._generate_answer(question, context_text)
            logger.info("  answer: %s", answer[:100])
            return {
                "answer": answer,
                "retrieved_content": context_text,
                "retrieved_count": len(items),
                "retrieve_trace": retrieve_trace,
            }
        except Exception as exc:
            logger.error("answer_question failed qid=%s: %s", question_id, exc)
            return {
                "answer": "",
                "retrieved_content": "",
                "retrieved_count": 0,
                "retrieve_trace": {},
            }

    async def _retrieve(
        self,
        *,
        question: str,
        question_id: str,
        recent_history: str,
        runtime_metadata: dict[str, object],
    ) -> tuple[list[dict], str, dict[str, object]]:
        # 1. 先做 gate 决策，并始终并发检索 procedure/preference 规则记忆。
        gate = await self._decide_gate(
            question=question,
            recent_history=recent_history,
            runtime_metadata=runtime_metadata,
        )
        procedure_memory = _MemoryPlannerAdapter(self._retriever)
        episodic_memory = _MemoryPlannerAdapter(
            self._retriever,
            scope_channel=LME_SCOPE_CHANNEL,
            scope_chat_id=question_id,
            force_scope=True,
        )
        p_items = await retrieve_procedure_items(
            procedure_memory,
            query=question,
            top_k=self._top_k_procedure,
        )
        # 2. 再按 gate 判定检索 event/profile，并和规则记忆合并注入。
        h_items: list[dict] = []
        hyde_context = self._build_hyde_context(recent_history)
        if gate["route_decision"] == "RETRIEVE":
            h_items, _scope_mode, _hypothesis = await retrieve_episodic(
                episodic_memory,
                gate["episodic_query"],
                memory_types=["event", "profile"],
                top_k=self._top_k_history,
                context=hyde_context,
                hyde_enhancer=self._hyde_enhancer,
            )
        selected, block = self._build_injection_payload(procedure_memory, p_items, h_items)
        # 3. 最后仅在空召回场景触发 sufficiency retry（与主链路一致）。
        if gate["route_decision"] == "RETRIEVE" and self._sufficiency_checker and not block:
            refined = await self._retry_with_sufficiency(
                gate_query=gate["episodic_query"],
                selected_items=selected,
                recent_history=recent_history,
            )
            if refined:
                extra_h_items, _scope_mode, _hypothesis = await retrieve_episodic(
                    episodic_memory,
                    refined,
                    memory_types=["event", "profile"],
                    top_k=self._top_k_history,
                    context=hyde_context,
                    hyde_enhancer=self._hyde_enhancer,
                )
                selected, block = self._build_injection_payload(
                    procedure_memory,
                    p_items,
                    h_items + extra_h_items,
                )
                h_items = h_items + extra_h_items
        trace = self._build_retrieve_trace(
            gate=gate,
            procedure_items=p_items,
            history_items=h_items,
            selected_items=selected,
        )
        return selected, block, trace

    async def _decide_gate(
        self,
        *,
        question: str,
        recent_history: str,
        runtime_metadata: dict[str, object],
    ) -> dict[str, str]:
        if self._query_rewriter is not None:
            decision = await self._query_rewriter.decide(
                user_msg=question,
                recent_history=recent_history,
            )
            return {
                "gate_type": "query_rewriter",
                "route_decision": "RETRIEVE" if decision.needs_episodic else "NO_RETRIEVE",
                "episodic_query": decision.episodic_query or question,
            }
        route = await self._history_route.decide(
            user_msg=question,
            metadata=runtime_metadata,
            recent_history=recent_history,
        )
        return {
            "gate_type": "history_route",
            "route_decision": "RETRIEVE" if route.needs_history else "NO_RETRIEVE",
            "episodic_query": route.rewritten_query or question,
        }

    def _build_injection_payload(
        self,
        memory: _MemoryPlannerAdapter,
        procedure_items: list[dict],
        history_items: list[dict],
    ) -> tuple[list[dict], str]:
        merged = self._merge_memory_items(procedure_items + history_items)
        selected = memory.select_for_injection(merged)
        block, _ids = memory.build_injection_block(merged)
        return selected, block

    @staticmethod
    def _build_retrieve_trace(
        *,
        gate: dict[str, str],
        procedure_items: list[dict],
        history_items: list[dict],
        selected_items: list[dict],
    ) -> dict[str, object]:
        # 1. 统计 type 命中数（基于最终 selected，贴近注入有效命中）。
        type_hits: dict[str, int] = {}
        for item in selected_items:
            mtype = str(item.get("memory_type", "") or "")
            if not mtype:
                continue
            type_hits[mtype] = type_hits.get(mtype, 0) + 1
        # 2. 统计 path 命中数（procedure 固定 procedure，history 按 raw/hyde）。
        path_hits: dict[str, int] = {}
        if procedure_items:
            path_hits["procedure"] = len(procedure_items)
        for item in history_items:
            path = str(item.get("_retrieval_path", "history_raw") or "history_raw")
            path_hits[path] = path_hits.get(path, 0) + 1
        # 3. 返回可聚合 trace 字段。
        return {
            "gate_type": gate.get("gate_type", ""),
            "route_decision": gate.get("route_decision", ""),
            "type_hits": type_hits,
            "path_hits": path_hits,
        }

    async def _retry_with_sufficiency(
        self,
        *,
        gate_query: str,
        selected_items: list[dict],
        recent_history: str,
    ) -> str:
        result = await self._sufficiency_checker.check(
            query=gate_query,
            items=selected_items,
            context=recent_history,
        )
        if result.is_sufficient:
            return ""
        return result.refined_query or ""

    @staticmethod
    def _merge_memory_items(items: list[dict]) -> list[dict]:
        seen: set[str] = set()
        merged: list[dict] = []
        for item in items:
            item_id = str(item.get("id", "") or "")
            if item_id and item_id in seen:
                continue
            if item_id:
                seen.add(item_id)
            merged.append(item)
        return merged

    @staticmethod
    def _build_runtime_metadata() -> dict[str, object]:
        # benchmark 无真实工具调用运行态，这里给出与主链路结构兼容的最小 metadata。
        return {
            "last_turn_had_task_tool": False,
            "recent_task_tools": [],
            "last_turn_tool_calls_count": 0,
        }

    @staticmethod
    def _format_gate_history(
        haystack_sessions: list[list[dict]],
        max_turns: int = 3,
        max_content_len: int | None = 100,
    ) -> str:
        flat_msgs: list[dict] = []
        for sess in haystack_sessions:
            for turn in sess:
                role = str(turn.get("role", "")).strip().lower()
                content = str(turn.get("content", "")).strip()
                if role in {"user", "assistant"} and content:
                    flat_msgs.append({"role": role, "content": content})
        turns: list[str] = []
        for msg in reversed(flat_msgs):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if max_content_len is not None:
                content = content[:max_content_len]
            turns.append(f"[{role}] {content}")
            if len(turns) >= max_turns * 2:
                break
        return "\n".join(reversed(turns))

    @staticmethod
    def _build_hyde_context(recent_history: str) -> str:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d %H:%M")
        if recent_history.strip():
            return f"当前时间：{date_str}\n{recent_history.strip()}"
        return f"当前时间：{date_str}"

    def _generate_answer(self, question: str, context: str) -> str:
        prompt = (
            "Based on the conversation history below, answer the question concisely.\n"
            "Rules:\n"
            "- Use the EXACT numbers, names, and values from the history. "
            "Never substitute with values from your general knowledge.\n"
            "- You may make reasonable inferences (e.g., if someone visits a city "
            "to meet their sister, that city is likely where the sister lives).\n"
            "- Pay attention to completion status: 'preparing to attend' or 'planning' "
            "means not done yet; 'recently completed' or 'already did' means done.\n"
            "- Only say 'I don't know' if there is truly no relevant information.\n\n"
            f"History:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer (≤30 words):"
        )
        try:
            resp = self._llm.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.error("LLM answer failed: %s", exc)
            return ""


# ─── Main Tester ─────────────────────────────────────────────────────

class AkasicLMETester:
    def __init__(
        self,
        config_path: str,
        db_path: str,
        workspace: str,
        max_workers: int = 2,
        question_type_filter: list[str] | None = None,
        use_flash: bool = False,
        use_hyde: bool = False,
    ) -> None:
        runtime_cfg = load_config(config_path)
        self.workspace = Path(workspace).expanduser().resolve()
        self.components = load_benchmark_components(
            config_path=config_path,
            db_path=db_path,
        )

        if use_flash:
            from dataclasses import replace
            qa_components = replace(
                self.components,
                llm_client=self.components.light_llm_client,
                model=self.components.light_model,
            )
        else:
            qa_components = self.components

        self.mem_agent = LMEMemAgent(
            components=self.components,
            workspace=self.workspace,
            memory_window=runtime_cfg.memory_window,
        )
        self.response_agent = LMEResponseAgent(
            components=qa_components,
            runtime_cfg=runtime_cfg,
            use_hyde=use_hyde,
        )
        self.evaluate_agent = EvaluateAgent(
            chat_deployment=qa_components.model,
            api_key=str(qa_components.llm_client.api_key),
            azure_endpoint=str(qa_components.llm_client.base_url),
        )

        self.max_workers = max_workers
        self.question_type_filter = question_type_filter
        self.processing_time = 0.0

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.error_log_file = str(self.workspace.parent / f"lme_error_log_{ts}.txt")
        with open(self.error_log_file, "w", encoding="utf-8") as f:
            f.write(f"Akasic LongMemEval Benchmark Error Log - {datetime.now()}\n{'='*80}\n")

    def _process_question(self, item: dict, skip_ingest: bool, idx: int, total: int) -> dict:
        qid = str(item.get("question_id", f"q{idx}"))
        qtype = str(item.get("question_type", ""))
        question = str(item.get("question", ""))
        answer = item.get("answer")

        if not question or answer is None:
            return {"skip": True}

        if self.question_type_filter and qtype not in self.question_type_filter:
            return {"skip": True}

        answer = str(answer)
        haystack_sessions: list[list[dict]] = item.get("haystack_sessions", [])
        haystack_dates: list[str] = item.get("haystack_dates", [])
        superseded_before = self.mem_agent.count_superseded(qid)

        # 日期列表不足时用空字符串补齐
        while len(haystack_dates) < len(haystack_sessions):
            haystack_dates.append("")

        # Ingest
        if not skip_ingest:
            logger.info(f"[{idx+1}/{total}] {qid} ({qtype}) ingest start — {len(haystack_sessions)} sessions")
            fact_count = self.mem_agent.ingest_haystack(
                question_id=qid,
                haystack_sessions=haystack_sessions,
                haystack_dates=haystack_dates,
            )
            logger.info(f"[{idx+1}/{total}] {qid} ingest done — {fact_count} facts total")
        else:
            logger.info(f"[{idx+1}/{total}] {qid} ({qtype}) skip ingest")

        # Answer
        logger.info(f"[{idx+1}/{total}] {qid} answering...")
        resp = self.response_agent.answer_question(
            question,
            qid,
            haystack_sessions,
        )
        generated_answer = resp.get("answer", "")
        retrieved_content = resp.get("retrieved_content", "")
        retrieve_trace = resp.get("retrieve_trace", {}) if isinstance(resp, dict) else {}
        superseded_after = self.mem_agent.count_superseded(qid)

        # Evaluate
        logger.info(f"[{idx+1}/{total}] {qid} evaluating...")
        eval_result = self.evaluate_agent.evaluate_answer_accuracy(
            question=question,
            generated_answer=generated_answer,
            standard_answer=answer,
        )
        is_correct = eval_result.get("is_correct", False)
        explanation = eval_result.get("explanation", "")

        mark = "✓" if is_correct else "✗"
        logger.info(
            f"[{idx+1}/{total}] {qid} {mark} type={qtype} "
            f"gen=\"{generated_answer[:60]}\" std=\"{answer[:60]}\""
        )

        if not is_correct:
            self._log_error(
                qid=qid,
                qtype=qtype,
                question=question,
                generated_answer=generated_answer,
                standard_answer=answer,
                retrieved_content=retrieved_content,
                explanation=explanation,
            )

        return {
            "skip": False,
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "generated_answer": generated_answer,
            "standard_answer": answer,
            "is_correct": is_correct,
            "retrieved_count": resp.get("retrieved_count", 0),
            "retrieve_trace": retrieve_trace,
            "superseded_delta": max(0, superseded_after - superseded_before),
            "explanation": explanation,
        }

    def _log_error(self, **kw) -> None:
        try:
            lines = [
                f"\n{'='*80}",
                f"QID: {kw['qid']}  TYPE: {kw['qtype']}",
                f"QUESTION:\n{kw['question']}\n",
                f"RETRIEVED CONTENT:\n{kw.get('retrieved_content','')}\n",
                f"GENERATED ANSWER:\n{kw['generated_answer']}\n",
                f"STANDARD ANSWER:\n{kw['standard_answer']}\n",
                f"EXPLANATION:\n{kw.get('explanation','')}\n",
                f"{'='*80}\n",
            ]
            with open(self.error_log_file, "a", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception:
            pass

    def run(self, data: list[dict], skip_ingest: bool = False) -> dict:
        t0 = time.time()
        total = len(data)
        results: list[dict] = []

        # 顺序处理（ingest 和 query 绑定 question_id，不互相干扰）
        for idx, item in enumerate(data):
            r = self._process_question(item, skip_ingest=skip_ingest, idx=idx, total=total)
            if not r.get("skip"):
                results.append(r)

        self.processing_time = time.time() - t0
        return self._compile_results(results)

    def _compile_results(self, results: list[dict]) -> dict:
        total = len(results)
        correct = sum(1 for r in results if r.get("is_correct"))
        overall_acc = correct / total if total else 0.0
        phase4_stats = self._build_phase4_stats(results)

        by_type: dict[str, dict] = {}
        for r in results:
            qt = r.get("question_type", "unknown")
            if qt not in by_type:
                by_type[qt] = {"total": 0, "correct": 0}
            by_type[qt]["total"] += 1
            if r.get("is_correct"):
                by_type[qt]["correct"] += 1

        type_stats: dict = {}
        for qt, counts in sorted(by_type.items()):
            acc = counts["correct"] / counts["total"] if counts["total"] else 0.0
            label = QUESTION_TYPE_NAMES.get(qt, qt)
            type_stats[label] = {
                "correct": counts["correct"],
                "total": counts["total"],
                "accuracy": round(acc, 4),
            }

        summary = {
            "overall_accuracy": round(overall_acc, 4),
            "total_questions": total,
            "correct_answers": correct,
            "processing_time_s": round(self.processing_time, 1),
            "config_model": self.components.model,
            "config_embed_model": self.components.embedder._model,
            "config_top_k": self.components.retriever._top_k,
            "type_stats": type_stats,
            "phase4_stats": phase4_stats,
        }

        logger.info("\n" + "=" * 60)
        logger.info(f"OVERALL ACCURACY: {overall_acc:.1%}  ({correct}/{total})")
        logger.info("=" * 60)
        for label, stats in type_stats.items():
            logger.info(
                f"  {label}: {stats['accuracy']:.1%} "
                f"({stats['correct']}/{stats['total']})"
            )
        logger.info(f"Processing time: {self.processing_time:.1f}s")
        logger.info("Gate distribution: %s", phase4_stats.get("gate_distribution", {}))
        logger.info("Supersede: %s", phase4_stats.get("supersede_stats", {}))
        logger.info("=" * 60)

        return {"summary": summary, "details": results}

    def _build_phase4_stats(self, results: list[dict]) -> dict[str, object]:
        total = len(results)
        gate_distribution: dict[str, int] = {}
        type_hit_questions: dict[str, int] = {}
        path_hit_questions: dict[str, int] = {}
        superseded_total = 0
        superseded_questions = 0

        # 1. 聚合 gate/type/path 命中与 supersede 增量。
        for row in results:
            trace = row.get("retrieve_trace", {}) if isinstance(row, dict) else {}
            route = str(trace.get("route_decision", "UNKNOWN") or "UNKNOWN")
            gate_distribution[route] = gate_distribution.get(route, 0) + 1
            for mtype, count in (trace.get("type_hits", {}) or {}).items():
                if int(count or 0) > 0:
                    type_hit_questions[str(mtype)] = type_hit_questions.get(str(mtype), 0) + 1
            for path, count in (trace.get("path_hits", {}) or {}).items():
                if int(count or 0) > 0:
                    path_hit_questions[str(path)] = path_hit_questions.get(str(path), 0) + 1
            delta = max(0, int(row.get("superseded_delta", 0) or 0))
            superseded_total += delta
            if delta > 0:
                superseded_questions += 1

        # 2. 计算命中率（按 question 维度）。
        denom = total if total > 0 else 1
        type_hit_rate = {k: round(v / denom, 4) for k, v in sorted(type_hit_questions.items())}
        path_hit_rate = {k: round(v / denom, 4) for k, v in sorted(path_hit_questions.items())}

        # 3. 输出 Phase4 统计切面。
        return {
            "gate_distribution": gate_distribution,
            "type_hit_rate": type_hit_rate,
            "path_hit_rate": path_hit_rate,
            "supersede_stats": {
                "questions_with_supersede": superseded_questions,
                "superseded_items_delta": superseded_total,
            },
        }


# ─── CLI ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Akasic LongMemEval Benchmark")
    p.add_argument("--config", default=str(_PROJECT_ROOT / "config.json"))
    p.add_argument("--workspace", default=str(DEFAULT_LME_WORKSPACE))
    p.add_argument("--db-path", default=str(DEFAULT_LME_DB))
    p.add_argument("--data", default=str(_HERE / "data" / "longmemeval" / "longmemeval_s"))
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--question-type", default=None, help="逗号分隔的 question_type（默认全部）")
    p.add_argument("--skip-ingest", action="store_true")
    p.add_argument("--use-flash", action="store_true", help="response/evaluate 走 light model")
    p.add_argument("--use-hyde", action="store_true", help="检索时启用 HyDE 增强")
    p.add_argument("--question-ids", default=None, help="只测指定 question_id（逗号分隔）")
    p.add_argument("--output", default=str(DEFAULT_LME_OUTPUT))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    resolved_db, resolved_workspace = _prepare_phase0_guardrails(
        config_path=args.config,
        db_path=args.db_path,
        workspace=args.workspace,
        skip_ingest=args.skip_ingest,
    )
    resolved_output = _assert_output_in_run_root(args.output, resolved_workspace.parent)
    question_type_filter = (
        [qt.strip() for qt in args.question_type.split(",")]
        if args.question_type
        else None
    )

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.error(
            "Download with:\n"
            "  pip install huggingface_hub\n"
            "  huggingface-cli download xiaowu0162/longmemeval "
            "--repo-type dataset --local-dir benchmark/data/longmemeval"
        )
        sys.exit(1)

    data: list[dict] = json.loads(data_path.read_text(encoding="utf-8"))
    if args.max_samples:
        data = data[: args.max_samples]
    if args.question_ids:
        qid_set = {q.strip() for q in args.question_ids.split(",")}
        data = [q for q in data if q.get("question_id", "") in qid_set]
        logger.info(f"Filtered to {len(data)} questions by --question-ids")
    logger.info(f"Loaded {len(data)} questions from {data_path}")

    tester = AkasicLMETester(
        config_path=args.config,
        db_path=str(resolved_db),
        workspace=str(resolved_workspace),
        question_type_filter=question_type_filter,
        use_flash=args.use_flash,
        use_hyde=args.use_hyde,
    )
    results = tester.run(data, skip_ingest=args.skip_ingest)

    resolved_output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Results saved to {resolved_output}")


if __name__ == "__main__":
    main()
