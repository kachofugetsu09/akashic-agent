"""
akasha 记忆引擎 — 生产运行时层

依赖 agent-framework 的基础设施（Config、Embedder、EventBus 等）。
纯算法逻辑委托到 akasha.core。
"""

from __future__ import annotations

import json
import hashlib
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from plugins.akasha.core import (
    # Types shared with core
    ActivationEventRow,
    ActivationTrace,
    AkashaActivationSnapshot,
    ActivationUpdate,
    AkashaCandidate,
    AkashaNode,
    CoreConfig,
    EdgeUpdate,
    SourceMessage,
    build_dense_message_index,
    turn_key,
    # Algorithm functions (aliased with _ prefix for internal convention)
    activation_edge_updates as _activation_edge_updates,
    local_residual as _local_residual,
    activation_updates as _activation_updates,
    compute_candidates as _core_compute_candidates,
    compute_candidates_from_snapshot as _core_compute_candidates_from_snapshot,
    dense_message_candidates as _dense_message_candidates,
    edges_by_src as _edges_by_src,
    effective_edge_weight as _effective_edge_weight,
    fan_counts as _fan_counts,
    graph_seed_keys_from_snapshot as _graph_seed_keys_from_snapshot,
    parse_turn_key as _parse_turn_key,
    bounded_add as _bounded_add,
    reinforce_boost_from_payload as _reinforce_boost_from_payload,
)
from agent.config_models import Config  # noqa: F401
from bus.events_lifecycle import TurnCommitted
from core.memory.engine import (
    EngineProfile,
    EvidenceRef,
    MemoryCapability,
    MemoryEngineDescriptor,
    MemoryIngestRequest,
    MemoryIngestResult,
    MemoryMutation,
    MemoryMutationResult,
    MemoryQuery,
    MemoryQueryResult,
    MemoryRecord,
    MemoryScope,
    MemoryToolProfile,
    MemoryToolSpec,
)
from memory2.embedder import Embedder
from plugins.akasha.config import AkashaConfig, resolve_akasha_db_path
from plugins.akasha.store import AkashaStore

if TYPE_CHECKING:
    from bus.event_bus import EventBus
    from core.net.http import SharedHttpResources


@dataclass(frozen=True)
class AkashaCard:
    key: str
    source_ref: str
    user_message: str
    assistant_preview: str
    happened_at: str
    score: float
    lane: str
    signals: dict[str, object]


@dataclass(frozen=True)
class PendingActivation:
    query_id: str
    seq: int          # 仅作 query_log 标识用
    ts: float         # 用于 EdgeUpdate / last_used_ts
    items: list[AkashaCandidate]
    query_vec: np.ndarray


class AkashaMemoryEngine:
    DESCRIPTOR = MemoryEngineDescriptor(
        name="akasha",
        profile=EngineProfile.RICH_MEMORY_ENGINE,
        capabilities=frozenset(
            {
                MemoryCapability.INGEST_MESSAGES,
                MemoryCapability.RETRIEVE_SEMANTIC,
                MemoryCapability.RETRIEVE_CONTEXT_BLOCK,
                MemoryCapability.RETRIEVE_STRUCTURED_HITS,
                MemoryCapability.SEMANTICS_RICH_MEMORY,
            }
        ),
        notes={"owner": "plugins.akasha.engine", "truth": "sessions.db/messages"},
    )

    def __init__(
        self,
        *,
        config: Config,
        akasha_config: AkashaConfig,
        workspace: Path,
        http_resources: "SharedHttpResources",
        event_publisher: "EventBus | None" = None,
    ) -> None:
        # 1. 初始化 sidecar store、原始消息库路径和 embedding 客户端。
        self._config = config
        self._akasha_config = akasha_config
        self._workspace = workspace
        self._session_db_path = workspace / "sessions.db"
        self._store = AkashaStore(
            resolve_akasha_db_path(
                workspace=workspace,
                akasha_config=akasha_config,
            )
        )
        embedding = config.memory.embedding
        self._embedder = Embedder(
            base_url=embedding.base_url
            or config.light_base_url
            or config.base_url
            or "",
            api_key=embedding.api_key
            or config.light_api_key
            or config.api_key,
            model=embedding.model,
            output_dimensionality=embedding.output_dimensionality,
            requester=http_resources.external_default,
        )
        self._event_bus = event_publisher
        self._pending_by_session: dict[str, PendingActivation] = {}
        self._graph_lock = threading.RLock()
        self._nodes: dict[str, AkashaNode] = {}
        self._edges: dict[tuple[str, str], float] = {}
        self._edges_by_src: dict[str, dict[str, float]] = {}
        self._fan: dict[str, int] = {}
        self._message_embeddings: dict[str, np.ndarray] = {}
        self._message_turn_keys: dict[str, str] = {}
        self._message_index = build_dense_message_index({})
        self._load_graph_cache()
        self._ensure_idf_table()
        self.closeables: list[object] = [self._store, self._embedder]
        self._wire_events()

    # 启动时自动检查 / 建 FTS IDF 表。缺失或漂移过大时重建。
    def _ensure_idf_table(self) -> None:
        from plugins.akasha.core import (
            build_idf_table, idf_table_is_stale, load_idf_from_db, set_idf_table,
        )
        sessions_db = str(self._session_db_path)
        conn = self._store.db
        try:
            stale = idf_table_is_stale(sessions_db, conn)
        except Exception:
            stale = True
        if stale and self._session_db_path.exists():
            try:
                idf = build_idf_table(sessions_db, conn)
                print(f"[akasha] built FTS IDF table: {len(idf)} tokens")
            except Exception as exc:  # noqa: BLE001
                print(f"[akasha] IDF build failed, falling back to no-filter: {exc}")
                set_idf_table({})
                return
        idf = load_idf_from_db(conn)
        set_idf_table(idf)

    # 创建 Akasha sidecar 数据库。
    @classmethod
    def ensure_workspace_storage(
        cls,
        *,
        akasha_config: AkashaConfig,
        workspace: Path,
    ) -> None:
        # 1. 启动前只确保数据库和 schema 存在。
        store = AkashaStore(
            resolve_akasha_db_path(
                workspace=workspace,
                akasha_config=akasha_config,
            )
        )
        store.close()

    # 注册 after-turn 增量写入。
    def _wire_events(self) -> None:
        # 1. Akasha 不接 consolidation，只关心每轮真实消息提交。
        if self._event_bus is not None:
            self._event_bus.on(TurnCommitted, self._on_turn_committed)

    # 返回 Akasha 工具描述。
    def tool_profile(self) -> MemoryToolProfile:
        # 1. Akasha 只开放 recall_memory；事实写入来自 sessions.db 生命周期。
        return MemoryToolProfile(
            recall=MemoryToolSpec(
                description=(
                    "从 Akasha message-as-truth 记忆引擎召回原始对话。"
                    "返回 Dense 精确命中和 Ripple 联想命中；回答具体事实前继续用 fetch_messages(source_ref) 取原文。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "要召回的历史对话主题"},
                        "limit": {
                            "type": "integer",
                            "description": "最多返回条数",
                            "minimum": 1,
                            "maximum": 20,
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
                search_hint="历史对话 原始消息 Akasha 右脑联想",
            ),
            tools=(
                MemoryToolSpec(
                    name="reinforce_memory",
                    description=(
                        "加强当前轮的 Akasha 记忆信号。仅当用户纠正你刚才的回答/工具选择,或明确说以后要记住某条做法时调用。"
                        "本工具强化的是当前对话轮,不需要也不能填写上一轮 source_ref;不要为了普通闲聊、新事实、已知偏好重复调用。"
                        "常见用法:用户说'昨晚睡眠要用 fitbit_health_snapshot/snapshot,不是 sleep_report'时,加强正确工具选择。"
                        "也适合强化流程纠正:例如'要先 fetch_messages(source_ref) 看原文再下结论'。调用后仍要正常回复用户。"
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "note": {"type": "string", "description": "为什么加强(简述纠正/强调的要点)"},
                        },
                        "required": [],
                    },
                    risk="write",
                    search_hint=(
                        "纠正 强调 记牢 加强记忆 当前轮 source_ref "
                        "snapshot fitbit_health_snapshot sleep_report fetch_messages"
                    ),
                ),
            ),
        )

    # 根据 MemoryQuery 执行 Akasha 检索。
    async def query(
        self,
        request: MemoryQuery,
    ) -> MemoryQueryResult:
        # 1. timeline 不是 Akasha MVP 范围，时间线事实仍应走 fetch/search messages。
        if request.intent == "timeline":
            return MemoryQueryResult(
                trace={"engine": self.DESCRIPTOR.name, "intent": "timeline_unsupported"}
            )

        # 2. 空 query 不触发状态更新。
        query_text = request.text.strip()
        if not query_text:
            return MemoryQueryResult(trace={"engine": self.DESCRIPTOR.name, "hit_count": 0})

        # 3. 检索旧 turn 图，并在 context 入口记录本轮激活。
        now_ts = _query_timestamp_unix(request)
        if now_ts is None:
            return MemoryQueryResult(
                trace={
                    "engine": self.DESCRIPTOR.name,
                    "intent": request.intent,
                    "error": "missing_query_timestamp",
                }
            )
        query_vec = np.array(await self._embedder.embed(query_text), dtype=np.float32)
        stateful = request.effect != "read_only"
        result = self._retrieve(
            query_text,
            query_vec,
            request,
            now_ts=now_ts,
            update_state=stateful,
        )
        if stateful and request.intent in {"context", "answer"}:
            self._remember_pending_activation(request, result.activation_items, query_vec, now_ts=now_ts)

        # 4. context 注入按 Akasha 配置展示 topK；工具查询继续尊重调用方 limit。
        dense_limit = self._akasha_config.dense_top_k
        ripple_limit = self._akasha_config.ripple_top_k
        if request.intent != "context":
            dense_limit = min(request.limit, dense_limit)
            ripple_limit = min(request.limit, ripple_limit)
        dense_cards = self._cards_from_keys(
            [(item.key, item.score, "dense", _candidate_signals(item)) for item in result.dense_items],
            limit=dense_limit,
        )
        dense_keys = {card.key for card in dense_cards}
        dense_pairs = {_card_dedupe_key(card) for card in dense_cards}
        ripple_cards = self._cards_from_keys(
            [
                (item.key, item.score, "ripple", _candidate_signals(item))
                for item in result.ripple_items
                if item.key not in dense_keys
            ],
            limit=ripple_limit,
            skip_pairs=dense_pairs,
        )
        text_block = (
            self._format_context_block(dense_cards, ripple_cards, now_ts=now_ts)
            if request.intent == "context"
            else ""
        )
        cards = [*dense_cards, *ripple_cards]

        # 5. 记录检索诊断日志（context/answer intent 才有意义）。
        if stateful and request.intent in {"context", "answer"} and request.scope.session_key:
            self._write_query_log(
                request=request,
                result=result,
                seq=result.seq,
                dense_cards=dense_cards,
                ripple_cards=ripple_cards,
                text_block=text_block,
            )

        return MemoryQueryResult(
            text_block=text_block,
            records=[_card_to_record(card, injected=bool(text_block)) for card in cards],
            trace={
                "engine": self.DESCRIPTOR.name,
                "profile": self.DESCRIPTOR.profile.value,
                "intent": request.intent,
                "effect": request.effect,
                "dense_count": len(dense_cards),
                "ripple_count": len(ripple_cards),
                "seed_count": result.trace.seed_count,
                "pool_count": result.trace.pool_count,
            },
            raw={"items": [_card_to_raw(card) for card in cards]},
        )

    # 接收外部批量 ingest 请求。
    async def ingest(self, request: MemoryIngestRequest) -> MemoryIngestResult:
        # 1. Akasha 的主写入来自 TurnCommitted；ingest 只支持显式 conversation_turn。
        if request.source_kind not in {"conversation_turn", "conversation_batch"}:
            return MemoryIngestResult(
                accepted=False,
                summary="unsupported source_kind",
                raw={"reason": "unsupported_source_kind"},
            )
        content: dict[str, object] = {}
        payload: object = request.content
        if isinstance(payload, dict):
            raw_content = cast(dict[object, object], payload)
            content = {str(key): value for key, value in raw_content.items()}
        user_text = str(content.get("user_message") or "")
        assistant_text = str(content.get("assistant_response") or "")
        if not user_text and not assistant_text:
            return MemoryIngestResult(
                accepted=False,
                summary="empty conversation",
                raw={"reason": "empty_conversation"},
            )

        # 2. 没有稳定 message id 时不伪造事实来源。
        return MemoryIngestResult(
            accepted=False,
            summary="akasha ingest requires persisted sessions.db messages",
            raw={"reason": "requires_persisted_messages"},
        )

    # Akasha 不通过 memorize/forget 改写事实。
    async def mutate(self, request: MemoryMutation) -> MemoryMutationResult:
        # 1. sidecar 不是事实库，显式写删不在 MVP 中提供。
        if request.kind == "forget":
            return MemoryMutationResult(
                accepted=False,
                status="unsupported",
                missing_ids=list(request.ids),
            )
        return MemoryMutationResult(accepted=False, status="unsupported")

    # 强化已引用项。
    def reinforce_items_batch(self, ids: list[str]) -> None:
        # 1. 引用强化暂不改变 Akasha 图，图状态由真实 query 激活驱动。
        return None

    # 撤销由指定消息生成的 Akasha 状态。
    def undo_by_message_sources(
        self,
        message_ids: list[str],
        *,
        dry_run: bool = False,
    ) -> dict[str, object]:
        clean_ids = [str(item).strip() for item in message_ids if str(item).strip()]
        if not clean_ids:
            return {"affected_ids": [], "restored_ids": [], "rollback_source_ids": []}
        affected_keys = self._affected_turn_keys(clean_ids)
        rollback_source_ids = list(clean_ids) if affected_keys else []
        if not dry_run and affected_keys:
            keys = sorted(affected_keys)
            turns = [
                parsed for key in keys if (parsed := _parse_turn_key(key)) is not None
            ]
            _ = self._store.delete_items_batch(keys)
            self._store.delete_query_state_for_turns(turns)
            _ = self._store.delete_cached_embeddings(clean_ids)
            self._remove_cached_nodes(keys)
            self._remove_cached_messages(clean_ids)
        return {
            "affected_ids": sorted(affected_keys),
            "restored_ids": [],
            "rollback_source_ids": sorted(rollback_source_ids),
        }

    # 返回引擎描述。
    def describe(self) -> MemoryEngineDescriptor:
        # 1. runtime 和 dashboard 都通过 descriptor 识别当前 engine。
        return self.DESCRIPTOR

    # Akasha 不提供 procedure 关键词规则。
    def keyword_match_procedures(
        self,
        action_tokens: list[str],
    ) -> list[dict[str, object]]:
        # 1. procedure 语义属于 default memory，不混进 message-as-truth 引擎。
        return []

    # Akasha 不提供 event timeline 查询。
    def list_events_by_time_range(
        self,
        time_start: datetime,
        time_end: datetime,
        *,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        # 1. 时间线问题使用 search_messages/fetch_messages 更贴近原始事实。
        return []

    # dashboard 列出 Akasha turn 节点。
    def list_items_for_dashboard(
        self,
        *,
        q: str = "",
        memory_type: str = "",
        status: str = "",
        source_ref: str = "",
        scope_channel: str = "",
        scope_chat_id: str = "",
        has_embedding: bool | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> tuple[list[dict[str, object]], int]:
        # 1. 过滤项先保持 MVP，只支持 q 和分页排序。
        _ = (memory_type, status, source_ref, scope_channel, scope_chat_id, has_embedding)
        return self._store.list_items_for_dashboard(
            q=q,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        )

    # dashboard 读取 Akasha 节点详情。
    def get_item_for_dashboard(
        self,
        item_id: str,
        *,
        include_embedding: bool = False,
    ) -> dict[str, object] | None:
        # 1. include_embedding 由 store MVP 忽略，避免把大向量塞进 dashboard。
        _ = include_embedding
        return self._store.get_item_for_dashboard(item_id)

    # dashboard 更新 Akasha 节点。
    def update_item_for_dashboard(
        self,
        item_id: str,
        *,
        status: str | None = None,
        extra_json: dict[str, object] | None = None,
        source_ref: str | None = None,
        happened_at: str | None = None,
        emotional_weight: int | None = None,
    ) -> dict[str, object] | None:
        # 1. Akasha sidecar 不支持手改事实或状态，直接返回当前详情。
        _ = (status, extra_json, source_ref, happened_at, emotional_weight)
        return self._store.get_item_for_dashboard(item_id)

    # 删除 Akasha sidecar 节点。
    def delete_item(self, item_id: str) -> bool:
        # 1. 只删除 sidecar 索引，不删除原始 messages。
        deleted = self._store.delete_item(item_id)
        if deleted:
            self._remove_cached_nodes([item_id])
        return deleted

    # 批量删除 Akasha sidecar 节点。
    def delete_items_batch(self, ids: list[str]) -> int:
        # 1. 只删除 sidecar 索引，不删除原始 messages。
        deleted = self._store.delete_items_batch(ids)
        if deleted:
            self._remove_cached_nodes(ids)
        return deleted

    # 查找相似 Akasha 节点。
    def find_similar_items_for_dashboard(
        self,
        item_id: str,
        *,
        top_k: int = 8,
        memory_type: str = "",
        score_threshold: float = 0.0,
        include_superseded: bool = False,
    ) -> list[dict[str, object]]:
        # 1. MVP dashboard 不做相似节点展开。
        _ = (item_id, top_k, memory_type, score_threshold, include_superseded)
        return []

    # 检索 Akasha 图并更新旧节点激活。
    def _retrieve(
        self,
        query: str,
        query_vec: np.ndarray,
        request: MemoryQuery,
        *,
        now_ts: float,
        update_state: bool,
    ) -> "_AkashaRetrieval":
        # 1. 准备内存图和当前查询所在的预测 seq。
        snapshot = self._graph_snapshot()
        seq = _current_query_seq(self._session_db_path, request.scope)
        source_db = (
            sqlite3.connect(str(self._session_db_path))
            if self._session_db_path.exists()
            else None
        )

        # 2. 状态激活只取原始 replay 限额，展示候选单独放宽。
        try:
            source_cursor = source_db.cursor() if source_db is not None else None
            dense_items = _dense_message_candidates(
                query_vec,
                snapshot.nodes,
                snapshot.message_embeddings,
                snapshot.message_turn_keys,
                limit=max(self._akasha_config.dense_top_k, request.limit),
                message_index=snapshot.message_index,
            )
            graph_seed_keys = _graph_seed_keys_from_snapshot(
                query_vec,
                snapshot,
                limit=self._akasha_config.dense_top_k,
            )
            activation_items, _, _ = _compute_candidates_from_snapshot(
                query,
                query_vec,
                now_ts,
                snapshot=snapshot,
                config=self._akasha_config,
                source_cursor=source_cursor,
                soft_recall=False,
                return_limit=self._akasha_config.activate_limit,
                graph_seed_keys=graph_seed_keys,
            )
            display_limit = max(
                24,
                max(self._akasha_config.ripple_top_k, request.limit) * 3,
            )
            ripple_items, _, trace = _compute_candidates_from_snapshot(
                query,
                query_vec,
                now_ts,
                snapshot=snapshot,
                config=self._akasha_config,
                source_cursor=source_cursor,
                soft_recall=True,
                return_limit=display_limit,
                graph_seed_keys=graph_seed_keys,
            )
        finally:
            if source_db is not None:
                source_db.close()

        # 3. 只读查询没有记忆动力学副作用。
        if update_state:
            updates = _activation_updates(activation_items, snapshot.nodes, now_ts)
            self._store.update_activation_batch(updates)
            self._apply_activation_updates(updates)
        return _AkashaRetrieval(
            dense_items=dense_items,
            ripple_items=ripple_items,
            activation_items=activation_items,
            trace=trace,
            seq=seq,
        )

    # 暂存本轮激活，等待 after-turn 拿到真实 message id 后建边。
    def _remember_pending_activation(
        self,
        request: MemoryQuery,
        items: list[AkashaCandidate],
        query_vec: np.ndarray,
        *,
        now_ts: float,
    ) -> None:
        # 1. 没有 session_key 时仍可检索，但不建当前 turn 的共激活边。
        if not request.scope.session_key or not items:
            return
        seq = _current_query_seq(self._session_db_path, request.scope)
        self._pending_by_session[request.scope.session_key] = PendingActivation(
            query_id=f"{request.scope.session_key}:{seq}",
            seq=seq,
            ts=now_ts,
            items=list(items),
            query_vec=query_vec,
        )

    # TurnCommitted 后把真实 user/assistant 写入 sidecar，并补本轮共激活边。
    async def _on_turn_committed(self, event: TurnCommitted) -> None:
        # 1. 跳过不应进入记忆的系统轮次。
        if event.session_key.startswith("scheduler:") or bool((event.extra or {}).get("skip_post_memory")):
            return
        messages = _load_committed_turn_messages(self._session_db_path, event)
        if not messages:
            return

        # 2. 分别 embed user 和 assistant，再按 cross CLI 规则合并到 turn 节点。
        embeddings = await self._embedder.embed_batch([message.content for message in messages])
        current_key = ""
        for message, embedding in zip(messages, embeddings, strict=False):
            self._store.upsert_cached_embedding(
                message=message,
                model=self._config.memory.embedding.model,
                embedding=embedding,
            )
            current_key = self._store.upsert_message_node(message, embedding)
            self._refresh_cached_node(current_key)
            self._refresh_cached_message(message, embedding, current_key)

        # 3. 用真实 current_key 建边，并记录激活诊断。
        #    reinforce 标记 = 本轮调用了 reinforce_memory 工具(记在 tool_chain)或 extra 回填；
        #    与离线重建(build._load_reinforce_boosts)读同一来源，live 与重放一致。
        reinforced = _reinforce_boost_for_turn(
            event.extra,
            event.tool_chain_raw,
        ) > 1.0
        pending = self._pending_by_session.pop(event.session_key, None)
        if current_key and pending is not None:
            self._commit_pending_activation(current_key, pending, reinforced=reinforced)

    # 把 pending activation 转成边和事件。
    def _commit_pending_activation(
        self,
        current_key: str,
        pending: PendingActivation,
        reinforced: bool = False,
    ) -> None:
        query_residual = self._compute_query_residual(pending.query_vec, current_key)
        edge_updates = _activation_edge_updates(
            current_key,
            pending.items,
            pending.ts,
            query_residual=query_residual,
            reinforced=reinforced,
        )
        self._store.upsert_edges(edge_updates)
        self._apply_edge_updates(edge_updates)

        # 2. 记录本轮激活明细，便于之后诊断。
        self._store.insert_activation_events([
            ActivationEventRow(
                seq=pending.seq,
                query_id=pending.query_id,
                activated_key=item.key,
                source=item.source,
                score=item.score,
                direct_score=item.direct,
                state_score=item.state,
                edge_score=item.edge,
                long_score=item.long,
                resource=item.resource,
                fan=item.fan,
            )
            for item in pending.items
        ])

    # 启动时加载一次内存图。
    def _load_graph_cache(self) -> None:
        nodes = {node.key: node for node in self._store.list_nodes()}
        edges, edges_meta = self._store.load_edges_with_meta()
        message_embeddings = dict(
            self._store.list_cached_embeddings(model=self._config.memory.embedding.model)
        )
        message_turn_keys = _load_message_turn_keys(self._session_db_path)
        message_index = build_dense_message_index(message_embeddings)
        with self._graph_lock:
            self._nodes = nodes
            self._edges = edges
            self._edges_meta = edges_meta
            self._edges_by_src = _edges_by_src(edges)
            self._fan = _fan_counts(edges)
            self._message_embeddings = message_embeddings
            self._message_turn_keys = message_turn_keys
            self._message_index = message_index

    # 取查询使用的内存图快照。
    def _graph_snapshot(self) -> AkashaActivationSnapshot:
        if not hasattr(self, "_graph_lock"):
            self._graph_lock = threading.RLock()
            self._nodes = {}
            self._edges = {}
            self._edges_meta = {}
            self._edges_by_src = {}
            self._fan = {}
            self._message_embeddings = {}
            self._message_turn_keys = {}
            self._message_index = build_dense_message_index({})
            self._load_graph_cache()
        with self._graph_lock:
            if not hasattr(self, "_message_index"):
                self._message_index = build_dense_message_index(self._message_embeddings)
            return AkashaActivationSnapshot(
                nodes=dict(self._nodes),
                edges=dict(self._edges),
                edges_meta=dict(self._edges_meta),
                fan=dict(self._fan),
                edges_by_src={
                    key: dict(value)
                    for key, value in self._edges_by_src.items()
                },
                message_embeddings=dict(self._message_embeddings),
                message_turn_keys=dict(self._message_turn_keys),
                message_index=self._message_index,
            )

    # 把查询产生的状态更新同步进内存图。
    def _apply_activation_updates(self, updates: list[ActivationUpdate]) -> None:
        if not updates:
            return
        with self._graph_lock:
            for item in updates:
                node = self._nodes.get(item.key)
                if node is None:
                    continue
                self._nodes[item.key] = replace(
                    node,
                    strength=item.strength,
                    resource=item.resource,
                    recall_count=item.recall_count,
                    last_activated_ts=item.ts,
                    last_strength_ts=item.ts,
                    last_resource_ts=item.ts,
                )

    # 把新写入或合并的节点同步进内存图。
    def _refresh_cached_node(self, key: str) -> None:
        node = self._store.get_node(key)
        if node is None:
            return
        with self._graph_lock:
            self._nodes[key] = node

    # 把 message-level dense 缓存同步进内存图。
    def _refresh_cached_message(
        self,
        message: SourceMessage,
        embedding: list[float],
        turn_key_value: str,
    ) -> None:
        with self._graph_lock:
            self._message_embeddings[message.id] = np.array(embedding, dtype=np.float32)
            self._message_turn_keys[message.id] = turn_key_value
            self._message_index = build_dense_message_index(self._message_embeddings)

    # ν_turn = 1 − max_{j<i} cos(query, prior_j)²；当前 turn 自身排除。
    def _compute_query_residual(self, query_vec: np.ndarray, current_key: str) -> float:
        with self._graph_lock:
            embeddings = [
                node.embedding
                for key, node in self._nodes.items()
                if key != current_key and node.embedding.size > 0
            ]
        if not embeddings:
            return 1.0
        prior = np.stack(embeddings).astype(np.float32)
        norms = np.linalg.norm(prior, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        prior = prior / norms
        return _local_residual(query_vec, prior)

    # 把新增或增强的边同步进内存图。
    def _apply_edge_updates(self, updates: list[EdgeUpdate]) -> None:
        if not updates:
            return
        with self._graph_lock:
            for item in updates:
                if item.src_key == item.dst_key:
                    continue
                edge_key = (item.src_key, item.dst_key)
                old = self._edges.get(edge_key)
                if old is None:
                    weight = 0.12 * item.strength
                else:
                    decayed = _effective_edge_weight(
                        old,
                        self._edges_meta.get(edge_key, 0.0),
                        item.ts,
                    )
                    weight = _bounded_add(decayed, 0.12 * item.strength, 2.0)
                self._edges[edge_key] = weight
                self._edges_meta[edge_key] = item.ts
                self._edges_by_src.setdefault(item.src_key, {})[item.dst_key] = weight
            self._fan = _fan_counts(self._edges)

    # 删除 dashboard 移除的节点和相关边。
    def _remove_cached_nodes(self, ids: list[str]) -> None:
        remove_ids = set(ids)
        if not remove_ids:
            return
        with self._graph_lock:
            for item_id in remove_ids:
                _ = self._nodes.pop(item_id, None)
            self._edges = {
                key: weight
                for key, weight in self._edges.items()
                if key[0] not in remove_ids and key[1] not in remove_ids
            }
            self._edges_meta = {
                key: value
                for key, value in self._edges_meta.items()
                if key[0] not in remove_ids and key[1] not in remove_ids
            }
            self._edges_by_src = _edges_by_src(self._edges)
            self._fan = _fan_counts(self._edges)

    # 删除 undo 移除的 message-level dense 缓存。
    def _remove_cached_messages(self, message_ids: list[str]) -> None:
        remove_ids = {str(item).strip() for item in message_ids if str(item).strip()}
        if not remove_ids:
            return
        with self._graph_lock:
            for message_id in remove_ids:
                _ = self._message_embeddings.pop(message_id, None)
                _ = self._message_turn_keys.pop(message_id, None)
            self._message_index = build_dense_message_index(self._message_embeddings)

    # 根据 message id 找到本轮 Akasha turn 节点。
    def _affected_turn_keys(self, message_ids: list[str]) -> set[str]:
        affected: set[str] = set()
        message_id_set = set(message_ids)
        with self._graph_lock:
            cached_turn_keys = dict(self._message_turn_keys)
            existing_keys = set(self._nodes)
        for message_id in message_ids:
            cached = cached_turn_keys.get(message_id)
            if cached and cached in existing_keys:
                affected.add(cached)
                continue
            parsed = _parse_message_id(message_id)
            if parsed is None:
                continue
            session_key, seq = parsed
            for candidate in _possible_turn_keys(session_key, seq, message_id_set):
                if candidate in existing_keys:
                    affected.add(candidate)
        return affected

    # 把 turn key 列表转成可注入 card。
    def _cards_from_keys(
        self,
        items: list[tuple[str, float, str, dict[str, object]]],
        *,
        limit: int,
        skip_pairs: set[tuple[str, str]] | None = None,
    ) -> list[AkashaCard]:
        # 1. 每个 card 的正文都回 sessions.db 取，sidecar 不充当事实来源。
        cards: list[AkashaCard] = []
        seen_keys: set[str] = set()
        seen_pairs = set(skip_pairs or set())
        for key, score, lane, signals in items:
            if key in seen_keys:
                continue
            seen_keys.add(key)
            card = _load_turn_card(
                self._session_db_path,
                key,
                assistant_preview_chars=self._akasha_config.assistant_preview_chars,
                score=score,
                lane=lane,
                signals=signals,
            )
            if card is None:
                continue
            pair_key = _card_dedupe_key(card)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            cards.append(card)
            if len(cards) >= limit:
                break
        return cards

    # 序列化并写入本次检索诊断日志。
    def _write_query_log(
        self,
        *,
        request: "MemoryQuery",
        result: "_AkashaRetrieval",
        seq: int,
        dense_cards: list[AkashaCard],
        ripple_cards: list[AkashaCard],
        text_block: str,
    ) -> None:
        # 1. 收集 source_ref 去重统计。
        all_source_refs: set[str] = set()
        for card in dense_cards:
            try:
                for ref in json.loads(card.source_ref):
                    all_source_refs.add(str(ref))
            except Exception:
                pass
        for card in ripple_cards:
            try:
                for ref in json.loads(card.source_ref):
                    all_source_refs.add(str(ref))
            except Exception:
                pass

        # 2. 批量读取 activation_items 的消息内容，填充 user_message / assistant_preview。
        session_db_path = getattr(self, "_session_db_path", None) or Path("")
        activation_content = _load_messages_batch(
            session_db_path,
            [item.key for item in result.activation_items],
            assistant_preview_chars=self._akasha_config.assistant_preview_chars,
        )

        def _candidate_to_dict(item: AkashaCandidate) -> dict[str, object]:
            user_msg, asst_preview = activation_content.get(item.key, ("", ""))
            return {
                "key": item.key,
                "user_message": user_msg,
                "assistant_preview": asst_preview,
                "score": item.score,
                "source": item.source,
                "path_type": item.path_type,
                "fan": item.fan,
                "direct": item.direct,
                "state": item.state,
                "edge": item.edge,
                "long": item.long,
                "resource": item.resource,
                "ripple": item.ripple,
                "seed_key": item.seed_key,
                "bridge_key": item.bridge_key,
                "suppressed": item.suppressed,
            }

        def _card_to_dict(card: AkashaCard) -> dict[str, object]:
            d: dict[str, object] = {
                "key": card.key,
                "user_message": card.user_message,
                "assistant_preview": card.assistant_preview,
                "score": card.score,
                "source_ref": card.source_ref,
            }
            d.update(card.signals)
            return d

        activation_items_json = json.dumps(
            [_candidate_to_dict(item) for item in result.activation_items],
            ensure_ascii=False,
        )
        dense_items_json = json.dumps(
            [_card_to_dict(card) for card in dense_cards],
            ensure_ascii=False,
        )
        ripple_items_json = json.dumps(
            [_card_to_dict(card) for card in ripple_cards],
            ensure_ascii=False,
        )

        # 3. text_block 截断到 500 chars 作为预览。
        preview = text_block[:500].rstrip() + ("..." if len(text_block) > 500 else "")

        store = getattr(self, "_store", None)
        if store is None:
            return
        store.insert_query_log(
            query_id=_query_log_id(request.scope.session_key or "", seq, request.intent, request.text),
            session_key=request.scope.session_key or "",
            seq=seq,
            query_text=request.text.strip(),
            intent=request.intent,
            ts=datetime.fromtimestamp(_query_timestamp_unix(request) or 0.0, timezone.utc).isoformat(),
            seed_count=result.trace.seed_count,
            pool_count=result.trace.pool_count,
            activated_count=len(result.activation_items),
            activation_threshold=self._akasha_config.activation_threshold,
            dense_count=len(dense_cards),
            ripple_count=len(ripple_cards),
            inject_chars=len(text_block),
            source_ref_count=len(all_source_refs),
            activation_items_json=activation_items_json,
            dense_items_json=dense_items_json,
            ripple_items_json=ripple_items_json,
            text_block_preview=preview,
        )

    # 格式化 agent 看到的 Dense/Ripple 双块。
    def _format_context_block(
        self,
        dense_cards: list[AkashaCard],
        ripple_cards: list[AkashaCard],
        *,
        now_ts: float,
    ) -> str:
        # 1. Dense 块优先展示重叠项，Ripple 块只展示 ripple-only。
        parts: list[str] = []
        if dense_cards or ripple_cards:
            date_label = datetime.fromtimestamp(now_ts, timezone.utc).astimezone().strftime("%Y-%m-%d")
            parts.append(f"# Akasha memory now={date_label}")
        if dense_cards:
            parts.append(_format_cards("## 左脑记忆：精确回忆", _sort_cards_by_time(dense_cards)))
        if ripple_cards:
            parts.append(_format_cards("## 右脑联想：潜意识第一反应", _sort_cards_by_time(ripple_cards)))

        # 2. 应用字符预算，避免历史消息过长撑爆上下文。
        text = "\n\n".join(part for part in parts if part.strip())
        max_chars = max(1, self._akasha_config.inject_max_chars)
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + f"\n...[Akasha 已截断 {len(text) - max_chars} 字]"


@dataclass(frozen=True)
class _AkashaRetrieval:
    dense_items: list[AkashaCandidate]
    ripple_items: list[AkashaCandidate]
    activation_items: list[AkashaCandidate]
    trace: ActivationTrace
    seq: int


def _reinforce_boost_for_turn(
    event_extra: dict[str, object] | None,
    tool_chain: list[dict[str, object]],
) -> float:
    return _reinforce_boost_from_payload(event_extra, tool_chain)


def _core_config(config: AkashaConfig) -> CoreConfig:
    return CoreConfig(
        dense_top_k=config.dense_top_k,
        dense_seed_threshold=config.dense_seed_threshold,
        activation_threshold=config.activation_threshold,
        cross_boost=config.cross_boost,
        nearby_time_seconds=config.nearby_time_seconds,
        nearby_dense_threshold=config.nearby_dense_threshold,
        soft_recall_threshold=config.soft_recall_threshold,
        soft_recall_direct_floor=config.soft_recall_direct_floor,
        activate_limit=config.activate_limit,
    )


def _compute_candidates(
    query: str,
    query_vec: np.ndarray,
    nodes: dict[str, AkashaNode],
    edges: dict[tuple[str, str], float],
    now_ts: float,
    *,
    config: AkashaConfig,
    fan: dict[str, int],
    source_cursor: sqlite3.Cursor | None = None,
    edges_by_src: dict[str, dict[str, float]] | None = None,
    edges_meta: dict[tuple[str, str], float] | None = None,
    soft_recall: bool = False,
    return_limit: int | None = None,
    graph_seed_keys: list[str] | None = None,
) -> tuple[list[AkashaCandidate], list[AkashaCandidate], ActivationTrace]:
    return _core_compute_candidates(
        query,
        query_vec,
        nodes,
        edges,
        now_ts,
        config=_core_config(config),
        fan=fan,
        source_cursor=source_cursor,
        edges_by_src=edges_by_src,
        edges_meta=edges_meta,
        soft_recall=soft_recall,
        return_limit=return_limit,
        graph_seed_keys=graph_seed_keys,
    )


def _compute_candidates_from_snapshot(
    query: str,
    query_vec: np.ndarray,
    now_ts: float,
    *,
    snapshot: AkashaActivationSnapshot,
    config: AkashaConfig,
    source_cursor: sqlite3.Cursor | None = None,
    soft_recall: bool = False,
    return_limit: int | None = None,
    graph_seed_keys: list[str] | None = None,
) -> tuple[list[AkashaCandidate], list[AkashaCandidate], ActivationTrace]:
    return _core_compute_candidates_from_snapshot(
        query,
        query_vec,
        snapshot,
        now_ts,
        config=_core_config(config),
        source_cursor=source_cursor,
        soft_recall=soft_recall,
        return_limit=return_limit,
        graph_seed_keys=graph_seed_keys,
    )


# ── sessions.db 特有辅助函数 ──────────────────────────────────────────


def _query_timestamp_unix(request: MemoryQuery) -> float | None:
    if request.timestamp is None:
        return None
    return float(request.timestamp.timestamp())


# 读取 message 到 turn key 的映射。
def _load_message_turn_keys(session_db_path: Path) -> dict[str, str]:
    # 1. sessions.db messages 表映射。
    if not session_db_path.exists():
        return {}
    with closing(sqlite3.connect(str(session_db_path))) as db:
        rows = db.execute("SELECT id, session_key, seq, role FROM messages").fetchall()
    result: dict[str, str] = {}
    for message_id, session_key, seq, role in rows:
        _, _, key = turn_key(str(session_key), int(seq), str(role or ""))
        result[str(message_id)] = key
    return result


# 获取当前 query 对应的预测 seq。
def _current_query_seq(
    session_db_path: Path,
    scope: MemoryScope,
) -> int:
    # 1. before-turn 检索时 user 尚未持久化，因此使用 sessions.db 当前 next seq。
    if not scope.session_key or not session_db_path.exists():
        return 0
    with closing(sqlite3.connect(str(session_db_path))) as db:
        row = db.execute(
            "SELECT COALESCE(MAX(seq) + 1, 0) FROM messages WHERE session_key = ?",
            (scope.session_key,),
        ).fetchone()
    return int(row[0] if row else 0)


# 从 sessions.db 反查 after-turn 已经持久化的本轮消息。
def _load_committed_turn_messages(
    session_db_path: Path,
    event: TurnCommitted,
) -> list[SourceMessage]:
    # 1. after-turn 发生在 append_messages 后，这里用内容和相邻 seq 反查稳定 id。
    if not session_db_path.exists() or not (event.persisted_user_message or event.input_message):
        return []
    user_text = event.persisted_user_message or event.input_message
    with closing(sqlite3.connect(str(session_db_path))) as db:
        db.row_factory = sqlite3.Row
        row = db.execute(
            """
            SELECT
                u.id AS user_id, u.session_key AS session_key, u.seq AS user_seq,
                u.content AS user_content, u.ts AS user_ts,
                a.id AS assistant_id, a.seq AS assistant_seq,
                a.content AS assistant_content, a.ts AS assistant_ts
            FROM messages u
            LEFT JOIN messages a
                ON a.session_key = u.session_key
               AND a.seq = u.seq + 1
               AND a.role = 'assistant'
               AND a.content = ?
            WHERE u.session_key = ?
              AND u.role = 'user'
              AND u.content = ?
            ORDER BY u.seq DESC
            LIMIT 1
            """,
            (event.assistant_response, event.session_key, user_text),
        ).fetchone()
    if row is None:
        return []
    messages = [
        SourceMessage(
            id=str(row["user_id"]),
            session_key=str(row["session_key"]),
            seq=int(row["user_seq"]),
            role="user",
            content=str(row["user_content"] or ""),
            ts=str(row["user_ts"] or ""),
        )
    ]
    if row["assistant_id"] is not None:
        messages.append(
            SourceMessage(
                id=str(row["assistant_id"]),
                session_key=str(row["session_key"]),
                seq=int(row["assistant_seq"]),
                role="assistant",
                content=str(row["assistant_content"] or ""),
                ts=str(row["assistant_ts"] or ""),
            )
        )
    return messages


# 从 sessions.db 读取 turn card。
def _load_turn_card(
    session_db_path: Path,
    key: str,
    *,
    assistant_preview_chars: int,
    score: float,
    lane: str,
    signals: dict[str, object],
) -> AkashaCard | None:
    # 1. 用 turn key 回源到 messages 表，user 全量、assistant 截断。
    parsed = _parse_turn_key(key)
    if parsed is None or not session_db_path.exists():
        return None
    session_key, seq = parsed
    with closing(sqlite3.connect(str(session_db_path))) as db:
        db.row_factory = sqlite3.Row
        user_row = db.execute(
            """
            SELECT id, content, ts
            FROM messages
            WHERE session_key = ? AND seq = ? AND role = 'user'
            """,
            (session_key, seq),
        ).fetchone()
        assistant_row = db.execute(
            """
            SELECT id, content, ts
            FROM messages
            WHERE session_key = ? AND seq = ? AND role = 'assistant'
            """,
            (session_key, seq + 1),
        ).fetchone()
    if user_row is None and assistant_row is None:
        return None
    source_ids = [
        str(row["id"])
        for row in (user_row, assistant_row)
        if row is not None and str(row["id"]).strip()
    ]
    assistant_text = str(assistant_row["content"] or "") if assistant_row is not None else ""
    user_text = str(user_row["content"] or "") if user_row is not None else ""
    happened_at = (
        str(user_row["ts"] or "")
        if user_row is not None
        else str(assistant_row["ts"] or "")
    )
    return AkashaCard(
        key=key,
        source_ref=json.dumps(source_ids, ensure_ascii=False),
        user_message=user_text,
        assistant_preview=_clip_assistant(assistant_text, assistant_preview_chars),
        happened_at=happened_at,
        score=score,
        lane=lane,
        signals=signals,
    )


def _parse_message_id(message_id: str) -> tuple[str, int] | None:
    return _parse_turn_key(message_id)


def _possible_turn_keys(
    session_key: str,
    seq: int,
    deleted_message_ids: set[str],
) -> tuple[str, ...]:
    current_id = f"{session_key}:{seq}"
    previous_id = f"{session_key}:{seq - 1}" if seq > 0 else ""
    next_id = f"{session_key}:{seq + 1}"
    if next_id in deleted_message_ids:
        return (current_id,)
    if previous_id in deleted_message_ids:
        return (previous_id,)
    keys = [current_id]
    if seq > 0:
        keys.append(previous_id)
    return tuple(key for key in keys if key)


# 截断 assistant 预览。
def _clip_assistant(text: str, limit: int) -> str:
    # 1. assistant 只作为 disambiguation，不能替代 fetch 原文。
    clean = " ".join(text.split())
    limit = max(0, int(limit))
    if limit <= 0 or len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "..."


# 归一化文本，避免多空格造成同文候选无法去重。
def _normalize_card_text(text: str) -> str:
    # 1. prompt 单行展示和去重都使用相同的空白规则。
    return " ".join(text.split()).strip()


# 生成注入去重键，只压掉 user 和助手预览都相同的候选。
def _card_dedupe_key(card: AkashaCard) -> tuple[str, str]:
    # 1. 同一句历史消息可能在不同 turn 重复出现，展示时只保留最高分。
    return _normalize_card_text(card.user_message), _normalize_card_text(card.assistant_preview)


def _card_ts(card: AkashaCard) -> float:
    try:
        return datetime.fromisoformat(card.happened_at.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _sort_cards_by_time(cards: list[AkashaCard]) -> list[AkashaCard]:
    return sorted(cards, key=_card_ts, reverse=True)


# 生成检索日志 ID，避免同一轮 context 和 recall_memory 互相覆盖。
def _query_log_id(session_key: str, seq: int, intent: str, query_text: str) -> str:
    # 1. 同一轮可能有预检索和显式 recall_memory，多条日志都要保留。
    digest = hashlib.sha1(f"{intent}\n{query_text}".encode("utf-8")).hexdigest()[:10]
    return f"{session_key}:{seq}:{intent}:{digest}"


# 格式化一个记忆块。
def _format_cards(title: str, cards: list[AkashaCard]) -> str:
    # 1. 每条 card 都带 source_ref，agent 需要事实时可继续 fetch_messages。
    lines = [title]
    for card in cards:
        user_text = json.dumps(_normalize_card_text(card.user_message), ensure_ascii=False)
        assistant_text = json.dumps(
            _normalize_card_text(card.assistant_preview),
            ensure_ascii=False,
        )
        happened_at = _format_card_time(card.happened_at)
        time_part = f" t={happened_at}" if happened_at else ""
        lines.append(
            f"- user={user_text} assistant={assistant_text}"
            f"{time_part} source_ref={card.source_ref}"
        )
    return "\n".join(lines)


def _format_card_time(value: str) -> str:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%m-%d")
    except ValueError:
        return ""


# 把 card 转成 MemoryRecord。
def _card_to_record(card: AkashaCard, *, injected: bool) -> MemoryRecord:
    # 1. summary 是展示 card，不是事实摘要；证据仍然指向 source_ref。
    summary = f"user_message: {card.user_message}"
    if card.assistant_preview:
        summary += f"\nassistant_message: {card.assistant_preview}"
    return MemoryRecord(
        id=card.key,
        kind="turn",
        summary=summary,
        score=card.score,
        engine_kind="akasha",
        evidence=[
            EvidenceRef(
                kind="message_range",
                refs=_source_ref_ids(card.source_ref),
                resolver="session",
                source_ref=card.source_ref,
            )
        ],
        signals=card.signals,
        injected=injected,
    )


# 把 card 转成 raw item。
def _card_to_raw(card: AkashaCard) -> dict[str, object]:
    # 1. raw item 兼容 recall_memory 和 dashboard 里的通用命名。
    return {
        "id": card.key,
        "memory_type": "turn",
        "summary": f"user_message: {card.user_message}",
        "score": card.score,
        "source_ref": card.source_ref,
        "extra_json": card.signals,
    }


# 批量从 sessions.db 读取多个 turn key 的消息内容，一次连接完成。
def _load_messages_batch(
    session_db_path: Path,
    keys: list[str],
    *,
    assistant_preview_chars: int,
) -> dict[str, tuple[str, str]]:
    # 1. 解析所有 key，跳过无法解析的。
    if not keys or not session_db_path.exists():
        return {}
    parsed: list[tuple[str, str, int]] = []
    for key in keys:
        result = _parse_turn_key(key)
        if result is not None:
            session_key, seq = result
            parsed.append((key, session_key, seq))
    if not parsed:
        return {}

    # 2. 一次性拉取 user seq 和 user_seq+1（assistant），避免每个 key 单独开连接。
    seq_pairs: list[tuple[str, int]] = []
    for _, sk, user_seq in parsed:
        seq_pairs.append((sk, user_seq))
        seq_pairs.append((sk, user_seq + 1))
    placeholders = ",".join("(?,?)" for _ in seq_pairs)
    flat_params: list[object] = [v for pair in seq_pairs for v in pair]
    with closing(sqlite3.connect(str(session_db_path))) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            f"""
            SELECT session_key, seq, role, content
            FROM messages
            WHERE (session_key, seq) IN ({placeholders})
              AND role IN ('user', 'assistant')
            """,
            flat_params,
        ).fetchall()

    # 3. 按 (session_key, seq) 分组成 {(sk, seq): {role: content}}。
    by_turn: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        sk = str(row["session_key"])
        seq_val = int(row["seq"])
        role = str(row["role"])
        content = str(row["content"] or "")
        by_turn.setdefault((sk, seq_val), {})[role] = content

    # 4. 为每个 key 映射 user_message / assistant_preview。
    result_map: dict[str, tuple[str, str]] = {}
    for key, sk, user_seq in parsed:
        user_msg = by_turn.get((sk, user_seq), {}).get("user", "")
        asst_raw = by_turn.get((sk, user_seq + 1), {}).get("assistant", "")
        result_map[key] = (user_msg, _clip_assistant(asst_raw, assistant_preview_chars))
    return result_map


# 从 source_ref JSON 数组中取消息 id。
def _source_ref_ids(source_ref: str) -> list[str]:
    # 1. source_ref 始终由 Akasha 生成，解析失败时回退空列表。
    try:
        value = cast(object, json.loads(source_ref))
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    items = cast(list[object], value)
    return [str(item) for item in items if str(item).strip()]


# 把 candidate 信号放进 record。
def _candidate_signals(item: AkashaCandidate) -> dict[str, object]:
    # 1. signals 只用于诊断，不参与事实判断。
    return {
        "lane": "akasha",
        "source": item.source,
        "ripple": item.ripple,
        "direct": item.direct,
        "state": item.state,
        "edge": item.edge,
        "long": item.long,
        "resource": item.resource,
        "fan": item.fan,
        "path_type": item.path_type,
        "seed_key": item.seed_key,
        "bridge_key": item.bridge_key,
        "path_value": item.path_value,
        "suppressed": item.suppressed,
    }
