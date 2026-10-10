"""MessageProjection.estimate 对同一不可变请求对象只估算一次。"""

from plugins.models.contract import (
    LLMResponse,
    ModelRequest,
)
from plugins.models.contract import BoundModelDescriptor
from session.message import ContentReferences
from plugins.models.projection import MessageProjection


class _CountingModel:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def descriptor(self) -> BoundModelDescriptor:
        raise AssertionError("估算测试不读取模型描述")

    async def complete(self, request: ModelRequest) -> LLMResponse:
        raise AssertionError("估算测试不调用模型")

    def estimate_appended_message_tokens(self, messages) -> int:
        raise AssertionError("估算测试不计算增量")

    @property
    def max_tool_schemas(self) -> int | None:
        raise AssertionError("估算测试不读取工具上限")

    def key_recovery(self, request_key: str) -> str:
        raise AssertionError("估算测试不读取恢复状态")

    def estimate_context_tokens(self, messages, tools=()):
        self.calls += 1
        return 42


def _projection(model: _CountingModel) -> MessageProjection:
    return MessageProjection(
        model,
        source="conversation",
        render_content=lambda part: (),
        tool_name=lambda binding: binding,
        read_call=lambda call_id: {},
        check_summary=lambda part: ContentReferences(),
    )


def test_estimate_reuses_result_for_same_request_object() -> None:
    model = _CountingModel()
    projection = _projection(model)
    request = ModelRequest(messages=[{"role": "user", "content": "hi"}])
    assert projection.estimate(request) == 42
    assert projection.estimate(request) == 42
    assert model.calls == 1
    # 新请求对象不命中身份备忘，重新估算。
    other = ModelRequest(messages=[{"role": "user", "content": "again"}])
    assert projection.estimate(other) == 42
    assert model.calls == 2
