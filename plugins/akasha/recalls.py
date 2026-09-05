"""实际查询的耐久出处独立于学习图，正文仍从 Message 读取。"""
from __future__ import annotations

import json
from datetime import datetime
import numpy as np
from typing import TYPE_CHECKING, Annotated, Literal, Self, cast
from collections.abc import Mapping

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from plugins.content.api import Reference
from plugins.context.api import Materials
from .application.cycle import MemoryCycle, RetrievalTicket
from .domain.model import Turn
from .infrastructure.consumption import Consumption
from .infrastructure.sparse_index.encoding import tokenize
from session.log import MessageCatalog, OwnerStore
from session.message import CallRef, ContentPart
from session.message_codec import json_value

if TYPE_CHECKING:
    from .learning import Learning

Text = Annotated[str, Field(min_length=1)]


class ContextSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    kind: Literal["context"] = "context"
    session_id: Text
    source: Text
    through_seq: Annotated[int, Field(ge=0)]


class ToolSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    kind: Literal["tool"] = "tool"
    session_id: Text
    call_ref: CallRef


class ProgramSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    kind: Literal["program"] = "program"
    key: Text
    query: Annotated[str, Field(min_length=1, max_length=32000)]


class Hit(BaseModel):
    """记录查询实际选中的来源及顺序，不复制学习材料正文。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    node_id: Annotated[int, Field(ge=0)]
    session_id: Text
    message_ids: tuple[Text, ...] = Field(min_length=1, max_length=10000)
    score: FiniteFloat
    lane: Literal["dense", "completion"]
    sources: tuple[Text, ...]
    basin_ids: tuple[Text, ...] = ()


class Recall(BaseModel):
    """这是发生过的查询，不证明模型已接收或输出已送达。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    version: Literal[1] = 1
    learning_binding: Text
    graph_version: Annotated[int, Field(ge=0)]
    source: Annotated[ContextSource | ToolSource | ProgramSource, Field(discriminator="kind")]
    timestamp: AwareDatetime
    limit: Annotated[int, Field(ge=1, le=40)]
    max_chars: Annotated[int, Field(gt=0)] = 12000
    strong: bool = False
    time_start: AwareDatetime | None = None
    time_end: AwareDatetime | None = None
    hits: tuple[Hit, ...] = Field(max_length=45)
    presented_message_ids: tuple[Text, ...] = ()
    active_basin_count: Annotated[int, Field(ge=0)]
    pushes: Annotated[int, Field(ge=0)]
    residual_l1: Annotated[FiniteFloat, Field(ge=0)]


    @model_validator(mode="after")
    def check_hits(self) -> Self:
        """一次查询不能声称命中未来节点或把相同节点重复计入两条展示通道。"""
        nodes = [hit.node_id for hit in self.hits]
        if any(node >= self.graph_version for node in nodes) or len(set(nodes)) != len(nodes):
            raise ValueError("召回命中不属于查询时的唯一已学习节点")
        members = {identity for hit in self.hits for identity in hit.message_ids}
        if (len(set(self.presented_message_ids)) != len(self.presented_message_ids)
            or not set(self.presented_message_ids) <= members):
            raise ValueError("实际呈现的消息必须是本次命中的唯一成员")
        if self.time_start is not None and self.time_end is not None and self.time_start > self.time_end:
            raise ValueError("召回时间窗口倒置")
        return self


class RecallRecords:
    """Akasha 自有只增查询记录；对外只发布 read，不暴露 OwnerStore。"""

    def __init__(self, state: OwnerStore):
        self._state = state

    def save(self, identity: str, recall: Recall) -> str:
        """查询完成后创建；同一 key 不能覆盖另一张图上的查询结果。"""
        payload = recall.model_dump(mode="json")
        if len(json.dumps(payload, ensure_ascii=False).encode()) > 1_048_576:
            raise ValueError("召回出处超过单条记录上限，请缩小查询范围")
        _ = self._state.transact(lambda transaction: transaction.save(
            "recall:" + identity, cast(Mapping[str, object], payload), expected_version=None,
        ))
        return identity

    def read(self, identity: str) -> Recall | None:
        record = self._state.read("recall:" + identity)
        if record is None:
            return None
        return Recall.model_validate_json(json.dumps(json_value(record.value)))

    def list(self) -> tuple[tuple[str, Recall], ...]:
        """按实际查询时间返回最新记录，不读取或重算学习图。"""
        records = tuple(
            (key.removeprefix("recall:"), Recall.model_validate_json(json.dumps(json_value(record.value))))
            for key, record in self._state.list() if key.startswith("recall:")
        )
        return tuple(sorted(records, key=lambda item: (item[1].timestamp, item[0]), reverse=True))


def select_hits(
    turns: list[Turn], state: Consumption, ticket: RetrievalTicket, cue: Turn,
    *, inhibited: set[int], limit: int, strong: bool = False,
    time_start: datetime | None = None, time_end: datetime | None = None,
) -> tuple[Hit, ...]:
    """保留精确向量和联想两条通道，输出引用覆盖已学习的全部成员。"""
    if ticket.state_version != len(turns):
        raise ValueError("召回票据与用于选择来源的图版本不一致")
    if not 1 <= limit <= 40:
        raise ValueError("召回 limit 必须在 1 到 40 之间")

    def matches(turn: Turn, sources: tuple[str, ...]) -> bool:
        started = datetime.fromisoformat(turn.started_at)
        return (
            (time_start is None or started >= time_start)
            and (time_end is None or started <= time_end)
            and (not strong or any(source != "relative_tail" for source in sources))
        )

    def hit(turn: Turn, score: float, lane: Literal["dense", "completion"],
            sources: tuple[str, ...], basins: tuple[str, ...] = ()) -> Hit:
        suffix = turn.node_id - state.legacy_prefix.count
        members = ((turn.user_message_id, turn.assistant_message_id) if suffix < 0
                   else tuple(identity for _, identity in state.applied[suffix].members))
        return Hit(node_id=turn.node_id, session_id=turn.session_key, message_ids=members,
                   score=score, lane=lane, sources=sources, basin_ids=basins)

    # 1. 向量分数选出最多五项；显式遗忘的节点不重新进入精确通道。
    dense: list[tuple[float, int]] = []
    if cue.user_dense is not None:
        for turn in turns:
            if turn.node_id in inhibited or not matches(turn, ("direct_dense",)):
                continue
            scores = [float(np.dot(cue.user_dense, vector))
                      for vector in (turn.user_dense, turn.assistant_dense) if vector is not None]
            if scores:
                dense.append((max(scores), turn.node_id))
    dense.sort(key=lambda item: (-item[0], item[1]))
    selected = [hit(turns[node], score, "dense", ("direct_dense",)) for score, node in dense[:min(5, limit)]]
    direct = {item.node_id for item in selected}

    # 2. 联想沿真实 ticket 顺序填充，与精确通道去重后才应用展示时间排序。
    completion: list[Hit] = []
    for item in ticket.completion.items:
        turn = turns[item.node_id]
        if item.node_id in direct or not matches(turn, item.sources):
            continue
        completion.append(hit(turn, item.score, "completion", item.sources, item.basin_ids))
        if len(completion) == limit:
            break
    def newest(item: Hit) -> tuple[datetime, bytes]:
        turn = turns[item.node_id]
        return datetime.fromisoformat(turn.started_at), turn.turn_id.encode()
    return tuple(sorted(selected, key=newest, reverse=True) + sorted(completion, key=newest, reverse=True))


def query_memory(
    cycle: MemoryCycle, state: Consumption, *, learning_binding: str,
    text: str, dense: np.ndarray | None, stamp: datetime,
    source: ContextSource | ToolSource | ProgramSource, limit: int,
) -> Recall:
    """Context、工具与独立程序使用相同图查询；临时 cue 不写回学习状态。"""
    gap = None if not cycle.turns else (stamp - datetime.fromisoformat(cycle.turns[-1].committed_at)).total_seconds()
    if gap is not None and gap < 0.0:
        raise ValueError("查询时间早于已发布学习图，需要核对时钟")
    cue = Turn(
        node_id=cycle.state_version, turn_id=f"query:{stamp.isoformat()}",
        session_key="" if isinstance(source, ProgramSource) else source.session_id,
        user_seq=source.through_seq if isinstance(source, ContextSource) else -1,
        user_message_id="", assistant_message_id="", started_at=stamp.isoformat(),
        committed_at=stamp.isoformat(), user_text=text, assistant_text="", user_dense=dense,
        assistant_dense=None, user_terms=tuple(sorted(tokenize(text).items())), assistant_terms=(),
        inter_gap_seconds=gap,
    )
    ticket = cycle.retrieve(cue)
    hits = select_hits(cycle.turns, state, ticket, cue, inhibited=cycle.inhibited_nodes, limit=limit)
    return Recall(
        learning_binding=learning_binding, graph_version=ticket.state_version,
        source=source, timestamp=stamp, limit=limit, hits=hits,
        active_basin_count=ticket.completion.active_basin_count,
        pushes=ticket.completion.pushes, residual_l1=ticket.completion.residual_l1,
    )


def render_materials(
    identity: str, recall: Recall, learning: Learning, catalog: MessageCatalog, *, max_chars: int,
) -> Materials:
    """正文从原消息读取；只有预算内实际呈现的消息获得本地引用证据。"""
    rows: list[str] = []
    references: list[Reference] = []
    used = 0
    for hit in recall.hits:
        reader = catalog.reader(hit.session_id)
        for message_id in hit.message_ids:
            message = reader.get(message_id)
            if message is None:
                raise ValueError(f"召回出处消息缺失: {message_id}")
            text = learning.text(message)
            if not text.strip():
                continue
            row = json.dumps({"message_id": message_id, "lane": hit.lane, "text": text}, ensure_ascii=False)
            if used + len(row) + bool(rows) > max_chars:
                continue
            used += len(row) + bool(rows)
            rows.append(row)
            references.append(Reference(message_id, resolved_ref=message_id, retrieval_ref=identity))
    parts = () if not rows else (ContentPart("text", "\n".join(rows)),)
    return Materials("", context=parts, references=tuple(references))
