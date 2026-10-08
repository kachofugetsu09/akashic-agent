"""MessageProjection.estimate 对同一不可变请求对象只估算一次。"""

from agent.plugin_composition.models import ModelRequest
from plugins.models.projection import MessageProjection


class _CountingModel:
    def __init__(self) -> None:
        self.calls = 0

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
        check_summary=lambda part: None,
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
