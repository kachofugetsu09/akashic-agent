from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from typing import TYPE_CHECKING

from session.log import MessageCatalog
from session.message import Message, Output
from session.message import Input
from session.message_codec import encode_body
from session.embedding_store import EmbeddingRecords
import numpy as np

from .domain.model import Turn, TurnFeedback
from .infrastructure.consumption import Applied
from .infrastructure.sparse_index.encoding import tokenize

if TYPE_CHECKING:
    from plugins.turn_projection.plugin import TurnProjection


type CausalKey = tuple[datetime, str, int, str]


@dataclass(frozen=True, slots=True)
class Sample:
    """本次读取的学习材料；只有实际消费的出处引用会随学习结果保存。"""

    ending: Message
    messages: tuple[Message, ...]
    observations: tuple[Message, ...]

    @property
    def key(self) -> CausalKey:
        return (
            self.ending.recorded_at.astimezone(UTC), self.ending.session_id,
            self.ending.seq, self.ending.message_id,
        )


def project_samples(
    catalog: MessageCatalog,
    projection: TurnProjection,
    *,
    include: Callable[[str, str], bool],
    heads: Mapping[str, int] | None = None,
) -> tuple[Sample, ...]:
    """在线与重放共享固定 heads 的读取与全局排序，不持有 SQL 或第二份 Turn 库。"""
    # 1. 先固定所有来源可见的日志上界，再分页取每个 Session 的完整前缀。
    heads = catalog.snapshot_heads() if heads is None else dict(heads)
    samples: list[Sample] = []
    for session_id, head in heads.items():
        messages = catalog.reader(session_id).snapshot(through_seq=head)
        by_id = {message.message_id: message for message in messages}
        for source in sorted({message.source for message in messages}):
            if not include(session_id, source):
                continue
            # 2. 只有插件给出的闭段成为样本；open/abandoned 仍可读取但不学习。
            for turn in projection.project(messages, source):
                if turn.status not in {"complete", "quiet"}:
                    continue
                if turn.ending_message_id is None:
                    raise ValueError("已结束投影缺少消息身份")
                ending = by_id[turn.ending_message_id]
                if not isinstance(ending.body, Output) or ending.body.finish != turn.status:
                    raise ValueError("学习投影结束点不匹配实际 Output")
                samples.append(Sample(
                    ending,
                    tuple(by_id[message_id] for message_id in turn.message_ids),
                    tuple(by_id[message_id] for _, message_id in turn.observations),
                ))
    # 3. seq 只在会话内排序；跨会话采用真实接纳时间，不修改倒退的源时间。
    return tuple(sorted(samples, key=lambda sample: sample.key))


def source_digest(sample: Sample) -> str:
    """校验实际学习成员及观察，包含身份、来源、接纳时间和正文。"""
    rows: list[list[object]] = [
        [message.session_id, message.seq, message.message_id, message.author,
         message.source, message.recorded_at.isoformat(), encode_body(message.body)]
        for message in (*sample.messages, *sample.observations)
    ]
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def applied_source(sample: Sample, *, learning_binding: str) -> Applied:
    return Applied(
        learning_binding=learning_binding,
        session_id=sample.ending.session_id,
        ending=(sample.ending.seq, sample.ending.message_id),
        members=tuple((message.seq, message.message_id) for message in sample.messages),
        observations=tuple((message.seq, message.message_id) for message in sample.observations),
        source_digest=source_digest(sample),
    )


def restore_sample(catalog: MessageCatalog, projection: TurnProjection, entry: Applied) -> Sample:
    """调用者打开出处中的精确 projection 后，重算并核对唯一学习样本。"""
    samples = project_samples(
        catalog, projection, heads={entry.session_id: entry.ending[0]},
        include=lambda session_id, source: True,
    )
    selected = next((sample for sample in samples if sample.ending.message_id == entry.ending[1]), None)
    if selected is None:
        raise ValueError("已学习结束消息不再产生原投影样本")
    if applied_source(selected, learning_binding=entry.learning_binding).model_dump() != entry.model_dump():
        raise ValueError("已学习消息的投影出处发生改变")
    return selected


def dialogue_turn(
    sample: Sample, *, node_id: int, previous: datetime | None,
    text: Callable[[Message], str], embeddings: EmbeddingRecords,
    embedding_model: str, dimension: int, feedback: TurnFeedback = TurnFeedback(),
) -> Turn | None:
    """默认学习完整问答；多输入共享一个样本，向量缺口明确失败。"""
    # 1. 选择学习材料的职责属于 Akasha，投影插件仍返回所有闭段。
    users = [message for message in sample.messages if isinstance(message.body, Input)]
    ending = sample.ending
    if not isinstance(ending.body, Output) or ending.body.finish != "complete" or not users:
        return None
    answer = text(ending)
    if not any(text(message).strip() for message in users) or not answer.strip():
        return None

    # 2. 在线 query 与完整学习使用同一种多输入文本和向量聚合。
    joined, user_dense = input_features(users, text=text, embeddings=embeddings,
                                        embedding_model=embedding_model, dimension=dimension)
    assistant_dense = fixed_vector(ending, embeddings, embedding_model, dimension)
    committed = ending.recorded_at.astimezone(UTC)
    gap = None if previous is None else (committed - previous).total_seconds()
    if gap is not None and gap < 0:
        raise ValueError("学习时间倒退；需要显式核对来源，不能自动重排已学习历史")
    return Turn(
        node_id=node_id, turn_id=f"{users[0].message_id}::{ending.message_id}",
        session_key=ending.session_id, user_seq=users[0].seq,
        user_message_id=users[0].message_id, assistant_message_id=ending.message_id,
        started_at=users[0].recorded_at.astimezone(UTC).isoformat(),
        committed_at=committed.isoformat(), user_text=joined, assistant_text=answer,
        user_dense=user_dense, assistant_dense=assistant_dense,
        user_terms=tuple(sorted(tokenize(joined).items())),
        assistant_terms=tuple(sorted(tokenize(answer).items())),
        inter_gap_seconds=gap, feedback=feedback,
    )


def fixed_vector(message: Message, embeddings: EmbeddingRecords, model: str, dimension: int) -> np.ndarray:
    values = embeddings.read(message, model=model, dimension=dimension)
    if values is None:
        raise ValueError(f"学习消息缺少固定 embedding: {message.message_id}")
    return np.asarray(values, dtype=np.float32)


def input_features(
    inputs: Sequence[Message], *, text: Callable[[Message], str], embeddings: EmbeddingRecords,
    embedding_model: str, dimension: int,
) -> tuple[str, np.ndarray | None]:
    """完整输入文本按顺序连接，多条输入的固定向量取均值后归一化。"""
    values = tuple(text(message) for message in inputs)
    vectors = [fixed_vector(message, embeddings, embedding_model, dimension)
               for message, value in zip(inputs, values) if value.strip()]
    dense = None if not vectors else vectors[0]
    if len(inputs) > 1 and vectors:
        mean = np.mean(np.stack(vectors), axis=0)
        norm = float(np.linalg.norm(mean))
        dense = mean / norm if norm > 0.0 else None
    return "\n\n".join(values), dense
