from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from typing import Protocol

from plugins.ledger.contract import MessageCatalog
from plugins.ledger.contract import Input, Message, Output, encode_body
import numpy as np

from .domain.model import Turn, TurnFeedback
from .infrastructure.consumption import Applied
from .infrastructure.lexical import tokenize
from ._boundaries import Turn as ProjectedTurn, TurnProjection


type CausalKey = tuple[datetime, str, int, str]


class EmbeddingRecords(Protocol):
    def read(self, message: Message, *, model: str, dimension: int) -> tuple[float, ...] | None: ...


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


@dataclass(frozen=True, slots=True)
class SessionCursor:
    """一个 Session 已投影到的固定 head，以及每个学习来源最后一个闭段的结束 seq。

    TurnProjection 合同允许输入"闭合 Turn 后的尾部"：工具结果只归属同段调用，
    闭段之后的投影不依赖更早的消息，所以从结束点续读与从头投影得到同样的后续
    闭段。续读点只是调用者的内存进度，缺失时退回从头投影。

    续读成立的前提是已提交前缀只追加：Message 不可改写，软删保留正文，compaction
    不改写 Message。head 只能发现倒退，发现不了同长度的改写或删后再追加；若将来
    允许在 consumer 生命周期内硬改写前缀，须由 Ledger 提供前缀 revision 让续读点
    失效。在那之前，任何硬修复都要停止并重启 consumer，重启后自然从头投影。
    """

    head: int
    boundaries: Mapping[str, int]


def project_session(
    catalog: MessageCatalog, projection: TurnProjection, session_id: str, head: int, *,
    include: Callable[[str, str], bool], cursor: SessionCursor | None = None,
) -> tuple[list[Sample], SessionCursor]:
    """只读取每个学习来源上次闭段之后的尾部；返回新闭段和前进后的续读点。"""
    # 1. 倒退的 head 不可信，退回从头投影。
    if cursor is not None and cursor.head > head:
        cursor = None
    reader = catalog.reader(session_id)
    boundaries = {} if cursor is None else dict(cursor.boundaries)
    samples: list[Sample] = []
    for source in sorted(reader.source_names()):
        if not include(session_id, source):
            continue
        # 2. 上次读取之后没有本来源消息时，它的闭段和开放尾部都没有变化。
        if cursor is not None and reader.head(source=source) <= cursor.head:
            continue
        after = boundaries.get(source, -1)
        messages = reader.scan(tuple, after_seq=after, through_seq=head, source=source)
        turns = projection.project(messages, source, after_seq=after)
        samples.extend(closed_samples(messages, turns))
        # 3. 任一闭段（含 abandoned）的结束点都是投影合同允许的续读起点。
        boundaries[source] = next(
            (turn.through_seq for turn in reversed(turns) if turn.status != "open"), after,
        )
    return samples, SessionCursor(head, boundaries)


def closed_samples(messages: Sequence[Message], turns: Sequence[ProjectedTurn]) -> list[Sample]:
    """只有插件给出的 complete/quiet 闭段成为样本；open/abandoned 不学习。"""
    by_id = {message.message_id: message for message in messages}
    samples: list[Sample] = []
    for turn in turns:
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
    return samples


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


def restore_sample(catalog: MessageCatalog, entry: Applied) -> Sample:
    """按已学出处的精确引用还原样本，不重投影整段历史。

    出处已经固定了成员、观察与结束点，所以恢复只需要按引用读取这些消息，
    再用 `source_digest` 核对它们没有变化。重投影会让一次恢复的代价随
    Session 前缀长度增长，而它在每次装载和每次 `recall_memory` 都要付。
    """

    reader = catalog.reader(entry.session_id)

    def fetch(ref: tuple[int, str]) -> Message:
        seq, message_id = ref
        message = reader.get(message_id)
        if message is None:
            raise ValueError(f"已学习出处的消息缺失: {message_id}")
        if message.session_id != entry.session_id or message.seq != seq:
            raise ValueError(f"已学习出处的 Session 或 seq 不一致: {message_id}")
        return message

    messages = tuple(fetch(ref) for ref in entry.members)
    observations = tuple(fetch(ref) for ref in entry.observations)
    sample = Sample(ending=messages[-1], messages=messages, observations=observations)
    if applied_source(sample, learning_binding=entry.learning_binding).model_dump() != entry.model_dump():
        raise ValueError("已学习消息的投影出处发生改变")
    return sample


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
