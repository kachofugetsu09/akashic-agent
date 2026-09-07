"""记忆反馈是普通工具结果，消费时才与完整学习节点一起生效。"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.plugin_composition.bindings import Bindings
from plugins.tools.api import CallSource, InvalidArguments, Result
from plugins.tools.plugin import TOOLS
from session.message import ContentPart, ContentReferences, Input, Output, ToolCall
from session.message_codec import json_value
from .learning import Feedback, Learning, resolve_feedback
from .projection import Sample


class FeedbackArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    message_ids: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1, max_length=20)
    reason: str = Field(default="", max_length=500)


def check_feedback(part: ContentPart) -> ContentReferences:
    _ = Feedback.model_validate_json(json.dumps(json_value(part.value)))
    return ContentReferences()


class FeedbackTool:
    """参数准备固定真实目标，执行只返回标记，不另存 staged feedback。"""

    idempotent = True

    def __init__(
        self, action: Literal["remember", "forget"], learning: Learning,
        bindings: Bindings, targets: Callable[[], Mapping[str, int]],
    ):
        self._action: Literal["remember", "forget"] = action
        self._learning = learning
        self._bindings = bindings
        self._targets = targets

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object]:
        """当前输入由请求前缀解析；无效反馈是可修正的工具错误。"""
        try:
            request = FeedbackArguments.model_validate(json_value(arguments))
        except ValidationError as error:
            raise InvalidArguments(str(error)) from error
        if source is None:
            raise InvalidArguments("记忆反馈需要实际对话中的 ToolCall")
        # 1. 只投影实际调用当时的来源；其他来源和后来输入不参与别名解析。
        messages = source.messages
        by_id = {message.message_id: message for message in messages}
        calling = by_id[source.call_ref.message_id]
        turns = self._learning.projection.project(messages, calling.source)
        turn = turns[-1]
        if turn.status != "open" or calling.message_id not in turn.message_ids:
            raise InvalidArguments("记忆反馈没有未闭合的对话来源")
        members = tuple(by_id[identity] for identity in turn.message_ids)
        inputs = [message for message in members if isinstance(message.body, Input)]
        if not inputs:
            raise InvalidArguments("记忆反馈没有对应输入")
        def resolve(action: Literal["remember", "forget"], request: FeedbackArguments) -> Feedback:
            identities = tuple(dict.fromkeys(
                inputs[-1].message_id if identity == "current_user_message" else identity
                for identity in request.message_ids
            ))
            return Feedback(action=action, target_message_ids=identities, reason=request.reason)
        feedback = resolve(self._action, request)
        # 2. 已完成的反馈共同校验到学习节点，U2/U3 不会成为两个相互冲突的目标。
        sample = Sample(calling, members, tuple(by_id[identity] for _, identity in turn.observations))
        previous = self._learning.read_feedback(sample, self._bindings)
        targets = self._targets()
        planned = [feedback]
        # 3. 同一 Output 的结果在调用前缀之外，必须联合检查其原始反馈请求。
        actions: dict[str, Literal["remember", "forget"]] = {
            "remember_memory": "remember", "forget_memory": "forget",
        }
        peers: list[tuple[int, ToolCall, Mapping[str, object], Literal["remember", "forget"]]] = []
        for index, part in enumerate(cast(Output, calling.body).parts):
            if not isinstance(part, ToolCall):
                continue
            metadata = self._bindings.describe(part.binding_id, TOOLS)
            tool = cast(Mapping[str, object], metadata["tool"])
            if tool["owner"] == self._learning.owner and tool["name"] in actions:
                peers.append((index, part, metadata, actions[cast(str, tool["name"])]))
        if len(peers) > 1 and any(metadata["prepare"] is not None for _, _, metadata, _ in peers):
            raise InvalidArguments("带参数转换的反馈需要分开调用，无法按原请求联合校验")
        for index, part, _, action in peers:
            if index == source.call_ref.part_index:
                continue
            try:
                peer = FeedbackArguments.model_validate(json_value(part.arguments))
            except ValidationError as error:
                raise InvalidArguments(str(error)) from error
            planned.append(resolve(action, peer))
        try:
            _ = resolve_feedback((*previous, *planned), targets,
                                 {message.message_id for message in inputs}, max(targets.values(), default=-1) + 1)
        except ValueError as error:
            raise InvalidArguments(str(error)) from error
        return feedback.model_dump(mode="json")

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        """使用已经耐久接纳的最终参数，不再读取当前消息或学习图。"""
        feedback = Feedback.model_validate_json(json.dumps(json_value(arguments)))
        return Result("success", (
            ContentPart("text", json.dumps({"recorded": True, **feedback.model_dump(mode="json")}, ensure_ascii=False)),
            ContentPart("akasha.feedback", cast(Mapping[str, object], feedback.model_dump(mode="json"))),
        ))

    async def query(self, key: str) -> Result | None:
        return None
