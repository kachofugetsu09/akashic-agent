from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from plugins.akasha.config import AkashaConfig
from plugins.akasha.core import (
    ActivationEventRow,
    ActivationTrace,
    AkashaActivationSnapshot,
    AkashaCandidate,
    CoreConfig,
    SourceMessage,
    activation_edge_updates,
    activation_updates,
    compute_candidates_from_snapshot,
    dense_message_candidates,
    edges_by_src,
    fan_counts,
    graph_seed_keys_from_snapshot,
    parse_turn_key,
    parse_ts_unix,
    turn_key,
)
from plugins.akasha.engine import (
    AkashaCard,
    _candidate_signals,
    _card_dedupe_key,
    _format_cards,
    _load_messages_batch,
    _load_turn_card,
    _query_log_id,
    _sort_cards_by_time,
)
from plugins.akasha.store import AkashaStore

CONTEXT_QUERY_LIMIT = 8


@dataclass(frozen=True)
class ReplayMessage:
    message: SourceMessage
    embedding: list[float]


@dataclass(frozen=True)
class ReplayTurnResult:
    current_key: str
    activation_items: list[AkashaCandidate]


@dataclass(frozen=True)
class ReplayActivation:
    activation_items: list[AkashaCandidate]
    dense_items: list[AkashaCandidate]
    ripple_items: list[AkashaCandidate]
    trace: ActivationTrace


class AkashaReplayRuntime:
    def __init__(
        self,
        *,
        store: AkashaStore,
        config: AkashaConfig,
        source_db_path: Path,
        source_cursor: sqlite3.Cursor,
        message_embeddings: dict[str, np.ndarray],
        message_turn_keys: dict[str, str],
    ) -> None:
        self._store = store
        self._config = config
        self._core_config = _core_config(config)
        self._source_db_path = source_db_path
        self._source_cursor = source_cursor
        self._message_embeddings = dict(message_embeddings)
        self._message_turn_keys = dict(message_turn_keys)

    # 按线上状态机回放一轮：先激活旧图，再提交当前 turn。
    def replay_turn(
        self,
        items: Sequence[ReplayMessage],
    ) -> ReplayTurnResult:
        if not items:
            return ReplayTurnResult(current_key="", activation_items=[])
        trigger = next((item for item in items if item.message.role == "user"), None)
        has_query = trigger is not None and bool(trigger.message.content.strip())
        activation = (
            self._activate_before_turn(trigger.message, trigger.embedding)
            if has_query and trigger is not None
            else _empty_activation()
        )
        current_key = self.commit_turn(items, activation.activation_items)
        if has_query and trigger is not None and trigger.message.session_key:
            self._write_query_log(trigger.message, activation)
        return ReplayTurnResult(
            current_key=current_key,
            activation_items=activation.activation_items,
        )

    # 使用当前 user embedding 检索历史节点，并更新旧节点状态。
    def activate_before_turn(
        self,
        message: SourceMessage,
        embedding: list[float],
    ) -> list[AkashaCandidate]:
        return self._activate_before_turn(message, embedding).activation_items

    def _activate_before_turn(
        self,
        message: SourceMessage,
        embedding: list[float],
    ) -> ReplayActivation:
        if message.role != "user":
            return _empty_activation()
        query_text = message.content.strip()
        if not query_text:
            return _empty_activation()
        if query_text != message.content:
            raise ValueError(f"Akasha replay 缺少 strip 后 query embedding: {message.id}")
        nodes = {node.key: node for node in self._store.list_nodes()}
        if not nodes:
            return _empty_activation()

        edges, edges_meta = self._store.load_edges_with_meta()
        query_vec = np.array(embedding, dtype=np.float32)
        snapshot = AkashaActivationSnapshot(
            nodes=nodes,
            edges=edges,
            edges_meta=edges_meta,
            fan=fan_counts(edges),
            edges_by_src=edges_by_src(edges),
            message_embeddings=self._message_embeddings,
            message_turn_keys=self._message_turn_keys,
        )
        graph_seed_keys = graph_seed_keys_from_snapshot(
            query_vec,
            snapshot,
            limit=self._config.dense_top_k,
        )
        now_ts = parse_ts_unix(message.ts)
        dense_items = dense_message_candidates(
            query_vec,
            snapshot.nodes,
            snapshot.message_embeddings,
            snapshot.message_turn_keys,
            limit=max(self._config.dense_top_k, CONTEXT_QUERY_LIMIT),
        )
        candidates, _, trace = compute_candidates_from_snapshot(
            query_text,
            query_vec,
            snapshot,
            now_ts,
            config=self._core_config,
            source_cursor=self._source_cursor,
            soft_recall=False,
            return_limit=self._config.activate_limit,
            graph_seed_keys=graph_seed_keys,
        )
        display_limit = max(
            24,
            max(self._config.ripple_top_k, CONTEXT_QUERY_LIMIT) * 3,
        )
        ripple_items, _, trace = compute_candidates_from_snapshot(
            query_text,
            query_vec,
            snapshot,
            now_ts,
            config=self._core_config,
            source_cursor=self._source_cursor,
            soft_recall=True,
            return_limit=display_limit,
            graph_seed_keys=graph_seed_keys,
        )
        self._store.update_activation_batch(activation_updates(candidates, nodes, now_ts))
        return ReplayActivation(candidates, dense_items, ripple_items, trace)

    # 提交当前 turn，并把本轮激活转成共激活边和诊断事件。
    def commit_turn(
        self,
        items: Sequence[ReplayMessage],
        activation_items: list[AkashaCandidate],
    ) -> str:
        current_key = ""
        for item in items:
            current_key = self._store.upsert_message_node(item.message, item.embedding)
            self._message_embeddings[item.message.id] = np.array(item.embedding, dtype=np.float32)
            self._message_turn_keys[item.message.id] = turn_key(
                item.message.session_key,
                item.message.seq,
                item.message.role,
            )[2]
        if current_key and activation_items:
            trigger = next((item.message for item in items if item.message.role == "user"), items[0].message)
            ts = parse_ts_unix(trigger.ts)
            self._store.upsert_edges(activation_edge_updates(current_key, activation_items, ts))
            self._store.insert_activation_events(_activation_events(trigger, activation_items))
        return current_key

    def _write_query_log(
        self,
        message: SourceMessage,
        activation: ReplayActivation,
    ) -> None:
        dense_cards = _cards_from_candidates(
            self._source_db_path,
            self._config,
            activation.dense_items,
            lane="dense",
            limit=self._config.dense_top_k,
        )
        dense_keys = {card.key for card in dense_cards}
        dense_pairs = {_card_dedupe_key(card) for card in dense_cards}
        ripple_cards = _cards_from_candidates(
            self._source_db_path,
            self._config,
            [item for item in activation.ripple_items if item.key not in dense_keys],
            lane="ripple",
            limit=self._config.ripple_top_k,
            skip_pairs=dense_pairs,
        )
        text_block = _format_context_block(
            self._config,
            dense_cards,
            ripple_cards,
            now_ts=parse_ts_unix(message.ts),
        )
        all_source_refs = _source_refs(dense_cards, ripple_cards)
        activation_content = _load_messages_batch(
            self._source_db_path,
            [item.key for item in activation.activation_items],
            assistant_preview_chars=self._config.assistant_preview_chars,
        )
        activation_items_json = json.dumps(
            [
                _candidate_to_log_item(
                    activation_content,
                    item,
                )
                for item in activation.activation_items
            ],
            ensure_ascii=False,
        )
        self._store.insert_query_log(
            query_id=_query_log_id(message.session_key, message.seq, "context", message.content),
            session_key=message.session_key,
            seq=message.seq,
            query_text=message.content.strip(),
            intent="context",
            ts=datetime.fromtimestamp(parse_ts_unix(message.ts), timezone.utc).isoformat(),
            seed_count=activation.trace.seed_count,
            pool_count=activation.trace.pool_count,
            activated_count=len(activation.activation_items),
            activation_threshold=self._config.activation_threshold,
            dense_count=len(dense_cards),
            ripple_count=len(ripple_cards),
            inject_chars=len(text_block),
            source_ref_count=len(all_source_refs),
            activation_items_json=activation_items_json,
            dense_items_json=json.dumps(
                [_card_to_log_item(card) for card in dense_cards],
                ensure_ascii=False,
            ),
            ripple_items_json=json.dumps(
                [_card_to_log_item(card) for card in ripple_cards],
                ensure_ascii=False,
            ),
            text_block_preview=_preview_text_block(text_block),
        )


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


def _empty_activation() -> ReplayActivation:
    return ReplayActivation([], [], [], ActivationTrace(seed_count=0, pool_count=0))

def _activation_events(
    message: SourceMessage,
    candidates: list[AkashaCandidate],
) -> list[ActivationEventRow]:
    query_id = f"{message.session_key}:{message.seq}"
    return [
        ActivationEventRow(
            seq=message.seq,
            query_id=query_id,
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
        for item in candidates
    ]


def _candidate_to_log_item(
    activation_content: dict[str, tuple[str, str]],
    item: AkashaCandidate,
) -> dict[str, object]:
    user_message, assistant_preview = activation_content.get(item.key, ("", ""))
    return {
        "key": item.key,
        "user_message": user_message,
        "assistant_preview": assistant_preview,
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


def _cards_from_candidates(
    source_db_path: Path,
    config: AkashaConfig,
    items: list[AkashaCandidate],
    *,
    lane: str,
    limit: int,
    skip_pairs: set[tuple[str, str]] | None = None,
) -> list[AkashaCard]:
    cards: list[AkashaCard] = []
    seen_keys: set[str] = set()
    seen_pairs = set(skip_pairs or set())
    for item in items:
        if item.key in seen_keys:
            continue
        seen_keys.add(item.key)
        card = _load_turn_card(
            source_db_path,
            item.key,
            assistant_preview_chars=config.assistant_preview_chars,
            score=item.score,
            lane=lane,
            signals=_candidate_signals(item),
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


def _format_context_block(
    config: AkashaConfig,
    dense_cards: list[AkashaCard],
    ripple_cards: list[AkashaCard],
    *,
    now_ts: float,
) -> str:
    parts: list[str] = []
    if dense_cards or ripple_cards:
        date_label = datetime.fromtimestamp(now_ts, timezone.utc).astimezone().strftime("%Y-%m-%d")
        parts.append(f"# Akasha memory now={date_label}")
    if dense_cards:
        parts.append(_format_cards("## 左脑记忆：精确回忆", _sort_cards_by_time(dense_cards)))
    if ripple_cards:
        parts.append(_format_cards("## 右脑联想：潜意识第一反应", _sort_cards_by_time(ripple_cards)))

    text = "\n\n".join(part for part in parts if part.strip())
    max_chars = max(1, config.inject_max_chars)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + f"\n...[Akasha 已截断 {len(text) - max_chars} 字]"


def _source_refs(
    dense_cards: list[AkashaCard],
    ripple_cards: list[AkashaCard],
) -> set[str]:
    refs: set[str] = set()
    for card in [*dense_cards, *ripple_cards]:
        for item in json.loads(card.source_ref):
            refs.add(str(item))
    return refs


def _card_to_log_item(card: AkashaCard) -> dict[str, object]:
    item: dict[str, object] = {
        "key": card.key,
        "user_message": card.user_message,
        "assistant_preview": card.assistant_preview,
        "score": card.score,
        "source_ref": card.source_ref,
    }
    item.update(card.signals)
    return item


def _preview_text_block(text_block: str) -> str:
    return text_block[:500].rstrip() + ("..." if len(text_block) > 500 else "")


def _turn_messages(
    cursor: sqlite3.Cursor,
    key: str,
    *,
    assistant_preview_chars: int,
) -> tuple[str, str]:
    parsed = parse_turn_key(key)
    if parsed is None:
        raise ValueError(f"Akasha turn key 无法解析: {key}")
    session_key, turn_seq = parsed
    rows = cursor.execute(
        """
        SELECT seq, role, content
        FROM messages
        WHERE session_key = ? AND seq IN (?, ?)
        ORDER BY seq
        """,
        (session_key, turn_seq, turn_seq + 1),
    ).fetchall()
    user_message = ""
    assistant_preview = ""
    has_user = False
    for row in rows:
        seq = int(row[0])
        role = str(row[1] or "")
        content = str(row[2] or "")
        if seq == turn_seq and role == "user":
            has_user = True
            user_message = content
        elif seq == turn_seq + 1 and role == "assistant":
            assistant_preview = _clip_text(content, assistant_preview_chars)
    if not has_user:
        raise LookupError(f"Akasha replay 找不到 turn 内容: {key}")
    return user_message, assistant_preview


def _clip_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."
