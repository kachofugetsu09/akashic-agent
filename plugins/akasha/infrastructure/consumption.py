"""随学习快照保存出处；不保存第二份消息正文或运行任务。"""
from __future__ import annotations

from typing import Annotated, Literal, Self
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

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


class Skipped(BaseModel):
    """一个明确不学习的闭段；跳过也是必须持久的消费事实。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    session_id: Text
    ending: Ref
    reason: Text


class Consumption(BaseModel):
    """在线学习与完整重建共用的唯一进度：每个闭段要么已学，要么已明确跳过。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[2] = 2
    # tuple 避免 frozen model 中仍可原位修改 dict。
    cutover_heads: tuple[tuple[Text, Head], ...]
    applied: tuple[Applied, ...] = ()
    skipped: tuple[Skipped, ...] = ()
    # 结束消息 → 已学下标（跳过记为 -1）；构造时整段校验后建立，追加只校验新条目。
    _endings: dict[str, int] = PrivateAttr(default_factory=dict)
    # 每条出处的规范 JSON 片段；整段文本由片段拼接，追加时只序列化新条目。
    _applied_json: list[str] | None = PrivateAttr(default=None)
    _skipped_json: list[str] | None = PrivateAttr(default=None)
    _text: str | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def check_order(self) -> Self:
        """同一结束消息只能学习一次；切换以前的闭段不能进入新规则。"""
        heads = dict(self.cutover_heads)
        if len(heads) != len(self.cutover_heads):
            raise ValueError("切换 heads 包含重复 Session")
        if tuple(sorted(self.cutover_heads)) != self.cutover_heads:
            raise ValueError("切换 heads 必须按 Session 排序")
        seen: dict[str, int] = {}
        for index, entry in enumerate(self.applied):
            if entry.ending[1] in seen:
                raise ValueError("同一结束消息不能重复学习")
            if entry.ending[0] <= heads.get(entry.session_id, -1):
                raise ValueError("新消费不能重新学习切换前的结束消息")
            seen[entry.ending[1]] = index
        for item in self.skipped:
            if item.ending[1] in seen:
                raise ValueError("同一结束消息不能既学习又跳过")
            if item.ending[0] <= heads.get(item.session_id, -1):
                raise ValueError("跳过记事不能指向切换前的结束消息")
            seen[item.ending[1]] = -1
        self._endings = seen
        return self

    def append(self, entry: Applied) -> Consumption:
        # 前缀已在构造时整段校验；归纳地只校验新条目，与整段 check_order 的结论相同。
        self._check_new(entry.ending, entry.session_id, "同一结束消息不能重复学习", "新消费不能重新学习切换前的结束消息")
        return self._extend((*self.applied, entry), self.skipped, entry.ending[1], len(self.applied))

    def mark_skipped(self, entry: Skipped) -> Consumption:
        """记录一次明确跳过；图与节点身份不因跳过改变。"""

        self._check_new(entry.ending, entry.session_id, "同一结束消息不能既学习又跳过", "跳过记事不能指向切换前的结束消息")
        return self._extend(self.applied, (*self.skipped, entry), entry.ending[1], -1)

    # 私有缓存（索引、JSON 片段）不属于消费事实；相等只比较公开字段。
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Consumption):
            return NotImplemented
        return (self.version, self.cutover_heads, self.applied, self.skipped) == (
            other.version, other.cutover_heads, other.applied, other.skipped)

    __hash__ = None  # type: ignore[assignment]

    def canonical_text(self) -> str:
        """与 canonical_json(self.model_dump(mode="json")) 逐字节相同：键序 applied、cutover_heads、skipped、version。"""
        if self._text is None:
            if self._applied_json is None:
                self._applied_json = [_canonical(entry.model_dump(mode="json")) for entry in self.applied]
            if self._skipped_json is None:
                self._skipped_json = [_canonical(entry.model_dump(mode="json")) for entry in self.skipped]
            heads = _canonical([list(item) for item in self.cutover_heads])
            self._text = (
                '{"applied":[' + ",".join(self._applied_json) + '],"cutover_heads":' + heads
                + ',"skipped":[' + ",".join(self._skipped_json) + '],"version":' + str(self.version) + "}"
            )
        return self._text

    def check_suffix(self, turns: list[Turn], start: int) -> None:
        """前 start 个节点已在上次发布时逐项核对，只核对之后的新节点。"""
        self.check_count(len(turns))
        for node_id in range(start, len(turns)):
            turn, entry = turns[node_id], self.applied[node_id]
            if turn.node_id != node_id:
                raise ValueError("学习节点必须连续且有序")
            if (turn.session_key != entry.session_id
                or turn.assistant_message_id != entry.ending[1]
                or (turn.user_seq, turn.user_message_id) not in entry.members):
                raise ValueError("学习节点与消费出处不一致")

    def applied_at(self, message_id: str) -> Applied | None:
        index = self._endings.get(message_id, -1)
        return None if index < 0 else self.applied[index]

    def _check_new(self, ending: Ref, session_id: str, duplicate: str, cutover: str) -> None:
        if ending[1] in self._endings:
            raise ValueError(duplicate)
        if ending[0] <= dict(self.cutover_heads).get(session_id, -1):
            raise ValueError(cutover)

    def _extend(self, applied: tuple[Applied, ...], skipped: tuple[Skipped, ...], ending: str, index: int) -> Consumption:
        extended = Consumption.model_construct(
            version=self.version, cutover_heads=self.cutover_heads, applied=applied, skipped=skipped,
        )
        extended._endings = {**self._endings, ending: index}
        if index >= 0 and self._applied_json is not None:
            extended._applied_json = [*self._applied_json, _canonical(applied[-1].model_dump(mode="json"))]
            extended._skipped_json = self._skipped_json
        elif index < 0 and self._skipped_json is not None:
            extended._skipped_json = [*self._skipped_json, _canonical(skipped[-1].model_dump(mode="json"))]
            extended._applied_json = self._applied_json
        return extended

    def check_count(self, count: int) -> None:
        if len(self.applied) != count:
            raise ValueError("消费进度与学习图节点数不一致")

    def check_next(self, turns: list[Turn]) -> None:
        """前缀已逐项对应时，只核对新追加的最后一个节点。"""
        self.check_count(len(turns))
        if not turns:
            return
        turn, entry = turns[-1], self.applied[-1]
        if turn.node_id != len(turns) - 1:
            raise ValueError("学习节点必须连续且有序")
        if (turn.session_key != entry.session_id
            or turn.assistant_message_id != entry.ending[1]
            or (turn.user_seq, turn.user_message_id) not in entry.members):
            raise ValueError("学习节点与消费出处不一致")

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


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


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
