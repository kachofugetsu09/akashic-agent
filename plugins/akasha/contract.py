"""akasha 发布的只读查询与结算合同。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Annotated, Any, Literal, Protocol, Self
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, FiniteFloat, model_validator
from plugins.ledger.contract import CallRef

from agent.plugin_composition import ServiceKey


class SemanticInterest(Protocol):
    def decision(self) -> bool | None: ...
    def status(self) -> str | None: ...
    async def score(
        self, texts: Sequence[str], *, cutoff: str
    ) -> tuple[float, ...]: ...


SEMANTIC_INTEREST = ServiceKey[SemanticInterest]("akasha.semantic-interest.v1")



# 互斥角色标记：第二个 provider 在 Core provide 唯一性检查处冲突，不需要读取消费者。
EMBEDDING_MEMORY_PLUGIN = ServiceKey[Any]("plugin.claim.embedding_memory")


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


class RecallRecordsRead(Protocol):
    def read(self, identity: str) -> Recall | None: ...
    def list(self) -> tuple[tuple[str, Recall], ...]: ...
    def legacy_page(self, before: str = "g", *, limit: int = 64) -> tuple[tuple[tuple[str, Recall], ...], str | None]: ...


# Fleet Observe 与 Akasha dashboard 读取查询出处，不取得学习 writer。
AKASHA_RECORDS_VIEW = ServiceKey[Callable[[], RecallRecordsRead]]("akasha.recall-records.v1")
