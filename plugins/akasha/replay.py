from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from plugins.akasha.config import AkashaConfig
from plugins.akasha.core import (
    ActivationEventRow,
    AkashaCandidate,
    CoreConfig,
    EdgeUpdate,
    SourceMessage,
    activation_updates,
    compute_candidates,
    edges_by_src,
    fan_counts,
    parse_ts_unix,
)
from plugins.akasha.store import AkashaStore


@dataclass(frozen=True)
class ReplayMessage:
    message: SourceMessage
    embedding: list[float]


@dataclass(frozen=True)
class ReplayTurnResult:
    current_key: str
    activation_items: list[AkashaCandidate]


class AkashaReplayRuntime:
    def __init__(
        self,
        *,
        store: AkashaStore,
        config: AkashaConfig,
        source_cursor: sqlite3.Cursor,
    ) -> None:
        self._store = store
        self._config = config
        self._core_config = _core_config(config)
        self._source_cursor = source_cursor

    # 按线上状态机回放一轮：先激活旧图，再提交当前 turn。
    def replay_turn(
        self,
        items: Sequence[ReplayMessage],
    ) -> ReplayTurnResult:
        if not items:
            return ReplayTurnResult(current_key="", activation_items=[])
        trigger = next((item for item in items if item.message.role == "user"), None)
        activation_items = (
            self.activate_before_turn(trigger.message, trigger.embedding)
            if trigger is not None
            else []
        )
        current_key = self.commit_turn(items, activation_items)
        return ReplayTurnResult(
            current_key=current_key,
            activation_items=activation_items,
        )

    # 使用当前 user embedding 检索历史节点，并更新旧节点状态。
    def activate_before_turn(
        self,
        message: SourceMessage,
        embedding: list[float],
    ) -> list[AkashaCandidate]:
        if message.role != "user":
            return []
        nodes = {node.key: node for node in self._store.list_nodes()}
        if not nodes:
            return []

        edges = self._store.load_edges()
        now_ts = parse_ts_unix(message.ts)
        candidates, _, _ = compute_candidates(
            message.content,
            np.array(embedding, dtype=np.float32),
            nodes,
            edges,
            now_ts,
            config=self._core_config,
            fan=fan_counts(edges),
            source_cursor=self._source_cursor,
            edges_by_src=edges_by_src(edges),
            soft_recall=False,
            return_limit=self._config.activate_limit,
        )
        self._store.update_activation_batch(activation_updates(candidates, nodes, now_ts))
        return candidates

    # 提交当前 turn，并把本轮激活转成共激活边和诊断事件。
    def commit_turn(
        self,
        items: Sequence[ReplayMessage],
        activation_items: list[AkashaCandidate],
    ) -> str:
        current_key = ""
        for item in items:
            current_key = self._store.upsert_message_node(item.message, item.embedding)
        if current_key and activation_items:
            trigger = next((item.message for item in items if item.message.role == "user"), items[0].message)
            ts = parse_ts_unix(trigger.ts)
            self._store.upsert_edges(_edge_updates(current_key, activation_items, ts))
            self._store.insert_activation_events(_activation_events(trigger, activation_items))
        return current_key


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


def _edge_updates(
    current_key: str,
    candidates: list[AkashaCandidate],
    ts: float,
) -> list[EdgeUpdate]:
    updates: list[EdgeUpdate] = []
    key_to_score = {item.key: item.score for item in candidates}
    for item in candidates:
        edge_strength = key_to_score.get(item.key, 1.0)
        updates.append(EdgeUpdate(current_key, item.key, edge_strength, ts))
        updates.append(EdgeUpdate(item.key, current_key, edge_strength, ts))
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1:]:
            edge_strength = math.sqrt(key_to_score[left.key] * key_to_score[right.key])
            updates.append(EdgeUpdate(left.key, right.key, edge_strength, ts))
            updates.append(EdgeUpdate(right.key, left.key, edge_strength, ts))
    return updates


def _activation_events(
    message: SourceMessage,
    candidates: list[AkashaCandidate],
) -> list[ActivationEventRow]:
    return [
        ActivationEventRow(
            seq=message.seq,
            query_id=message.id,
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
