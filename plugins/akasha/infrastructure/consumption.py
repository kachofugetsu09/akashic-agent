"""随学习快照保存出处；不保存第二份消息正文或运行任务。"""
from __future__ import annotations

from typing import Annotated, Literal, Self
from dataclasses import asdict
from collections.abc import Sequence
from pathlib import Path
from contextlib import closing
import sqlite3
import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..domain.model import Turn

Text = Annotated[str, Field(min_length=1)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Seq = Annotated[int, Field(ge=0)]
Head = Annotated[int, Field(ge=-1)]
Ref = tuple[Seq, Text]


class LegacyPrefix(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    count: Annotated[int, Field(ge=0)]
    index_state_sha256: Digest
    turns_digest: Digest


class Applied(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    learning_binding: Text
    session_id: Text
    ending: Ref
    members: tuple[Ref, ...]
    observations: tuple[Ref, ...]
    source_digest: Digest

    @model_validator(mode="after")
    def check_refs(self) -> Self:
        """在持久化边界拒绝重复、乱序或越过结束点的出处。"""
        all_refs = self.members + self.observations
        for refs in (self.members, self.observations):
            if any(left[0] >= right[0] for left, right in zip(refs, refs[1:])):
                raise ValueError("消费出处必须按 seq 严格递增")
        if len({ref[0] for ref in all_refs}) != len(all_refs) or len({ref[1] for ref in all_refs}) != len(all_refs):
            raise ValueError("消费出处不能重复")
        if not self.members or self.members[-1] != self.ending:
            raise ValueError("消费成员必须以 ending 结束")
        if any(ref[0] >= self.ending[0] for ref in self.observations):
            raise ValueError("工具观察必须先于 ending")
        return self


class Consumption(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[1] = 1
    legacy_prefix: LegacyPrefix
    # tuple 避免 frozen model 中仍可原位修改 dict。
    cutover_heads: tuple[tuple[Text, Head], ...]
    applied: tuple[Applied, ...] = ()

    @model_validator(mode="after")
    def check_order(self) -> Self:
        """同一结束消息只能学习一次；切换以前的闭段不能进入新规则。"""
        heads = dict(self.cutover_heads)
        if len(heads) != len(self.cutover_heads):
            raise ValueError("切换 heads 包含重复 Session")
        if tuple(sorted(self.cutover_heads)) != self.cutover_heads:
            raise ValueError("切换 heads 必须按 Session 排序")
        seen: set[str] = set()
        for entry in self.applied:
            if entry.ending[1] in seen:
                raise ValueError("同一结束消息不能重复学习")
            if entry.ending[0] <= heads.get(entry.session_id, -1):
                raise ValueError("新消费不能重新学习切换前的结束消息")
            seen.add(entry.ending[1])
        return self

    def append(self, entry: Applied) -> Consumption:
        return Consumption(
            legacy_prefix=self.legacy_prefix,
            cutover_heads=self.cutover_heads,
            applied=(*self.applied, entry),
        )

    def check_count(self, count: int) -> None:
        if self.legacy_prefix.count + len(self.applied) != count:
            raise ValueError("消费进度与学习图节点数不一致")

    def check_turns(self, turns: list[Turn]) -> None:
        """验证旧学习前缀不变，后缀逐项对应同一节点上的消息出处。"""
        self.check_count(len(turns))
        count = self.legacy_prefix.count
        if turns_digest(turns[:count]) != self.legacy_prefix.turns_digest:
            raise ValueError("旧学习前缀的固定材料不一致")
        for node_id, turn in enumerate(turns):
            if turn.node_id != node_id:
                raise ValueError("学习节点必须连续且有序")
        for entry, turn in zip(self.applied, turns[count:]):
            if (turn.session_key != entry.session_id
                or turn.assistant_message_id != entry.ending[1]
                or (turn.user_seq, turn.user_message_id) not in entry.members):
                raise ValueError("学习节点与消费出处不一致")


def turns_digest(turns: list[Turn]) -> str:
    """固定旧前缀全部学习材料，向量按真实浮点字节比较。"""
    digest = hashlib.sha256()
    for turn in turns:
        row = asdict(turn)
        for field in ("user_dense", "assistant_dense"):
            vector = row[field]
            row[field] = None if vector is None else {
                "dtype": vector.dtype.str, "shape": vector.shape,
                "bytes": vector.tobytes().hex(),
            }
        digest.update(json.dumps(
            row, sort_keys=True, ensure_ascii=False, allow_nan=False,
            separators=(",", ":"),
        ).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def load_legacy_prefix(state: Consumption, index: Path | None) -> list[Turn]:
    """读取固定旧索引，不把当前索引的后缀补入旧学习图。"""
    from .loader import load_turns
    from .sparse_index import sparse_index_state_sha256

    if state.legacy_prefix.index_state_sha256 == "0" * 64:
        return []
    if index is None or not index.exists():
        raise ValueError("学习恢复缺少固定旧索引")
    if sparse_index_state_sha256(index) != state.legacy_prefix.index_state_sha256:
        raise ValueError("学习恢复的旧索引身份不一致")
    turns = load_turns(index, max_turns=state.legacy_prefix.count) if state.legacy_prefix.count else []
    if turns_digest(turns) != state.legacy_prefix.turns_digest:
        raise ValueError("学习恢复的旧前缀材料不一致")
    return turns


def message_nodes(legacy: Sequence[Turn], applied: Sequence[Applied]) -> dict[str, int]:
    """旧前缀沿索引 Message ID；新节点包括投影中的全部输入和输出。"""
    targets = {identity: turn.node_id for turn in legacy
               for identity in (turn.user_message_id, turn.assistant_message_id)}
    for node, entry in enumerate(applied, start=len(legacy)):
        targets.update({identity: node for _, identity in entry.members})
    return targets


def load_message_nodes(memory: Path, legacy_index: Path | None) -> dict[str, int]:
    """反馈只读已发布出处，不取得图 writer 或自动重建。"""
    from .persistence import load_consumption

    state = load_consumption(memory)
    if state is None:
        raise ValueError("记忆反馈缺少已发布的学习消费状态")
    return message_nodes(load_legacy_prefix(state, legacy_index), state.applied)


def legacy_embedding_model(index: Path) -> str:
    """读取已验证旧索引固定的模型身份；维度由实际恢复向量决定。"""
    with closing(sqlite3.connect(f"file:{index}?mode=ro", uri=True)) as connection:
        row = connection.execute("SELECT value FROM metadata WHERE key='embedding_model'").fetchone()
    if row is None or not isinstance(row[0], str) or not row[0]:
        raise ValueError("旧学习索引缺少 embedding 空间身份")
    return row[0]
