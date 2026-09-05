from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition import ServiceKey
from agent.plugin_composition.bindings import Bindings
from agent.turn_effects import PostCommitEffect
from plugins.content.api import legacy_post_commit_effect
from plugins.tools.plugin import TOOLS
from plugins.turn_projection.plugin import TurnProjection
from session.embedding_store import MessageEmbeddings
from session.log import MessageCatalog
from session.message import ContentPart, Input, Message, Output, ToolCall, ToolResult
from .domain.model import Turn, TurnFeedback
from .infrastructure.consumption import Applied, Consumption, message_nodes
from .projection import Sample, dialogue_turn, project_samples, restore_sample


class LearningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    embedding_model: Annotated[str, Field(min_length=1)]
    dimension: Annotated[int, Field(gt=0)]
    sources: tuple[Annotated[str, Field(min_length=1)], ...]


class Feedback(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    action: Literal["remember", "forget"]
    target_message_ids: tuple[Annotated[str, Field(min_length=1)], ...] = Field(min_length=1, max_length=20)
    reason: str = Field(default="", max_length=500)


class Learning:
    """固定学习材料的纯规则；实际消息、向量和学习图由调用者提供。"""

    def __init__(self, projection: TurnProjection, *, owner: str):
        self.projection = projection
        self.owner = owner

    def text(self, message: Message) -> str:
        """只连接可见正文；控制、工具协议和内部模型事实不成为问答文本。"""
        if not isinstance(message.body, (Input, Output)):
            return ""
        return "".join(
            part.value for part in message.body.parts
            if isinstance(part, ContentPart) and part.kind == "text" and isinstance(part.value, str)
        )

    def samples(
        self, catalog: MessageCatalog, config: LearningConfig, *, heads: Mapping[str, int],
    ) -> tuple[Sample, ...]:
        samples = project_samples(
            catalog, self.projection, heads=heads,
            include=lambda session, source: source in config.sources,
        )
        return tuple(sample for sample in samples if self.accepts(sample))

    def accepts(self, sample: Sample) -> bool:
        """一条历史成员被禁止沉淀时，整个问答样本不成为学习材料。"""
        effects = tuple(legacy_post_commit_effect(message)
                        for message in (*sample.messages, *sample.observations))
        return PostCommitEffect.SUPPRESS not in effects

    def feedback(
        self, sample: Sample, previous: Sequence[Turn], state: Consumption, bindings: Bindings,
    ) -> TurnFeedback:
        """从本样本实际成功的 Akasha 调用读取反馈，按完整成员映射学习节点。"""
        # 1. 旧前缀沿原索引身份；新节点包含所有输入，不只首个输入。
        cutover = state.legacy_prefix.count
        targets = message_nodes(previous[:cutover], state.applied[:len(previous) - cutover])
        current = {message.message_id for message in sample.messages if isinstance(message.body, Input)}
        return resolve_feedback(self.read_feedback(sample, bindings), targets, current, len(previous))

    def read_feedback(self, sample: Sample, bindings: Bindings) -> tuple[Feedback, ...]:
        """只有实际 Akasha 调用的成功结果可贡献反馈。"""
        messages = {message.message_id: message for message in sample.messages}
        feedback: list[Feedback] = []
        for observation in sample.observations:
            body = observation.body
            if not isinstance(body, ToolResult) or body.outcome != "success":
                continue
            parts = [part for part in body.parts if part.kind == "akasha.feedback"]
            if not parts:
                continue
            request = messages[body.call_ref.message_id]
            if not isinstance(request.body, Output):
                raise ValueError("反馈出处不是工具调用")
            call = request.body.parts[body.call_ref.part_index]
            if not isinstance(call, ToolCall):
                raise ValueError("反馈出处没有指向 ToolCall")
            tool = cast(Mapping[str, object], bindings.describe(call.binding_id, TOOLS)["tool"])
            if tool["owner"] != self.owner:
                raise ValueError("其他工具不能伪造 Akasha 反馈")
            for part in parts:
                value = part.value
                if not isinstance(value, Mapping):
                    raise ValueError("Akasha 反馈必须是对象")
                feedback.append(Feedback.model_validate(dict(cast(Mapping[str, object], value))))
        return tuple(feedback)

    def make_turn(
        self, sample: Sample, config: LearningConfig, embeddings: MessageEmbeddings,
        *, previous: Sequence[Turn], state: Consumption, bindings: Bindings,
    ) -> Turn | None:
        return dialogue_turn(
            sample, node_id=len(previous),
            previous=datetime.fromisoformat(previous[-1].committed_at) if previous else None,
            text=self.text, embeddings=embeddings.bind(self.text),
            embedding_model=config.embedding_model, dimension=config.dimension,
            feedback=self.feedback(sample, previous, state, bindings),
        )

    def restore(
        self, catalog: MessageCatalog, embeddings: MessageEmbeddings,
        config: LearningConfig, entry: Applied, *, previous: Sequence[Turn],
        state: Consumption, bindings: Bindings,
    ) -> Turn:
        """只还原已学习材料，不打开模型、写图或重放学习事件。"""
        sample = restore_sample(catalog, self.projection, entry)
        if not self.accepts(sample):
            raise ValueError("已学习样本包含禁止沉淀的历史成员")
        if sample.ending.source not in config.sources:
            raise ValueError("已学习来源不属于原学习绑定")
        turn = self.make_turn(sample, config, embeddings, previous=previous, state=state, bindings=bindings)
        if turn is None:
            raise ValueError("原学习规则不再接纳已经学习的样本")
        return turn


def resolve_feedback(
    feedback: Sequence[Feedback], targets: Mapping[str, int], current: set[str], node_id: int,
) -> TurnFeedback:
    """同一组消息反馈只有一种动作，当前输入只能标记记住。"""
    remember: set[int] = set()
    forget: set[int] = set()
    for item in feedback:
        for identity in item.target_message_ids:
            if identity in current:
                if item.action == "forget":
                    raise ValueError("不能遗忘当前尚未学习的输入")
                node = node_id
            elif identity in targets:
                node = targets[identity]
            else:
                raise ValueError("反馈目标没有对应的已学习消息")
            (remember if item.action == "remember" else forget).add(node)
    if remember & forget:
        raise ValueError("同一样本不能同时记住和遗忘同一节点")
    return TurnFeedback(tuple(sorted(remember)), tuple(sorted(forget)), 3.0 if remember else 1.0)


AKASHA_LEARNING = ServiceKey[Learning]("akasha.learning.v1")
