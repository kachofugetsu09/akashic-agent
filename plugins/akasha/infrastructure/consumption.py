"""随学习快照保存出处；不保存第二份消息正文或运行任务。"""
from __future__ import annotations

from typing import Annotated, Literal, Self
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..domain.model import Turn

Text = Annotated[str, Field(min_length=1)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Seq = Annotated[int, Field(ge=0)]
Head = Annotated[int, Field(ge=-1)]
Ref = tuple[Seq, Text]


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
    """在线学习与完整重建共用的唯一进度：每个节点都有一个已学出处。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[2] = 2
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
            cutover_heads=self.cutover_heads,
            applied=(*self.applied, entry),
        )

    def check_count(self, count: int) -> None:
        if len(self.applied) != count:
            raise ValueError("消费进度与学习图节点数不一致")

    def check_turns(self, turns: list[Turn]) -> None:
        """验证每个节点都逐项对应同一份已学出处。"""
        self.check_count(len(turns))
        for node_id, turn in enumerate(turns):
            if turn.node_id != node_id:
                raise ValueError("学习节点必须连续且有序")
        for entry, turn in zip(self.applied, turns):
            if (turn.session_key != entry.session_id
                or turn.assistant_message_id != entry.ending[1]
                or (turn.user_seq, turn.user_message_id) not in entry.members):
                raise ValueError("学习节点与消费出处不一致")


def message_nodes(applied: Sequence[Applied]) -> dict[str, int]:
    """节点身份来自已学投影的全部成员。"""
    targets: dict[str, int] = {}
    for node, entry in enumerate(applied):
        targets.update({identity: node for _, identity in entry.members})
    return targets


def load_message_nodes(memory: Path) -> dict[str, int]:
    """反馈只读已发布出处，不取得图 writer 或自动重建。"""
    from .persistence import load_consumption

    state = load_consumption(memory)
    if state is None:
        raise ValueError("记忆反馈缺少已发布的学习消费状态")
    return message_nodes(state.applied)
