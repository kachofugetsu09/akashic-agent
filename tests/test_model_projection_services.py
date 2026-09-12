from __future__ import annotations

import pytest

from agent.plugin_composition.models import (
    BoundModelDescriptor,
    CapabilitySources,
    LLMResponse,
    ModelCapabilities,
    ModelRole,
)
from agent.plugin_contracts import ContentPart
from plugins.context.api import check_summary
from plugins.models.content import MODEL_CONTENT, ContentOwner
from plugins.models.projection import (
    MODEL_MESSAGE_CHECKS,
    MODEL_PROJECTION,
    MessageChecksOwner,
    ProjectionOwner,
)
from plugins.models.selection import MODEL_SELECTION, SelectionOwner


class _Model:
    def __init__(self) -> None:
        self.descriptor = BoundModelDescriptor(
            binding_id="binding",
            plugin_snapshot_id="snapshot",
            model_revision=1,
            model_id="model",
            connection_id="connection",
            driver_id="driver",
            driver_contract_version="1",
            auth_identity="identity",
            model="model",
            role=ModelRole.AGENT,
            reasoning_effort=None,
            capabilities=ModelCapabilities(context_window=123),
            capability_sources=CapabilitySources(),
            capability_digest="digest",
        )
        self.max_tool_schemas = 4

    def estimate_context_tokens(self, messages, tools=()):
        return len(messages) + len(tools)


def test_model_services_have_separate_keys_and_replacement_boundaries() -> None:
    assert MODEL_CONTENT.name == "models.content.v1"
    assert MODEL_PROJECTION.name == "models.projection.v1"
    assert MODEL_MESSAGE_CHECKS.name == "models.message-checks.v1"
    assert MODEL_SELECTION.name == "models.selection.v1"

    selection = SelectionOwner()
    assert selection.read(()) is None
    assert selection.check(ContentPart("model.selection", {"model_id": None, "reasoning_effort": None}))

    checks = MessageChecksOwner()
    assert checks.check_tool_rejection(
        ContentPart("model.tool_rejection", {"name": "tool", "arguments": {}, "error": "bad"})
    )

    content = ContentOwner()
    assert content.render(ContentPart("text", "hello"), artifacts={}) == (
        {"type": "text", "text": "hello"},
    )

    projection = ProjectionOwner().create(
        _Model(),
        source="conversation",
        render_content=lambda part: (),
        tool_name=lambda binding: "tool",
        read_call=lambda call_id: {"state": "success", "binding": {"binding_id": "binding"}},
        check_summary=check_summary,
    )
    assert projection.context_window == 123
    assert projection.max_tool_schemas == 4
    assert projection.render((), after_seq=-1).messages == ()
    assert projection.facts(LLMResponse("done", call_record_id="call"), ()).kind == "model.facts"


@pytest.mark.asyncio
async def test_model_content_owner_keeps_artifact_loading_read_only() -> None:
    loaded = await ContentOwner().load_artifacts(object(), (), accepts_images=True)  # type: ignore[arg-type]
    assert loaded == {}
