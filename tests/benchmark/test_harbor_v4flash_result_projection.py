from types import SimpleNamespace

from benchmark.harbor_v4flash.result_projection import project_agent_context


def test_project_agent_context_initializes_optional_harbor_metadata() -> None:
    context = SimpleNamespace(
        n_input_tokens=None,
        n_cache_tokens=None,
        n_output_tokens=None,
        metadata=None,
    )
    result = {
        "thread_id": "programmatic:test",
        "turn_id": "turn:test",
        "status": "completed",
        "terminal_source": "turn/read_recovery",
        "event_count": 26,
        "terminal": {
            "usage": {
                "inputTokens": 162497,
                "cachedInputTokens": 148224,
                "outputTokens": 3992,
            }
        },
    }

    project_agent_context(
        context,
        result,
        harness_name="akasic-v4flash",
        harness_version="0.1.0",
        source_digest="sha256:test",
    )

    assert context.n_input_tokens == 162497
    assert context.n_cache_tokens == 148224
    assert context.n_output_tokens == 3992
    assert context.metadata["terminal_source"] == "turn/read_recovery"
    assert context.metadata["event_count"] == 26
    assert context.metadata["usage_available"] is True


def test_project_agent_context_allows_failed_turn_without_usage() -> None:
    context = SimpleNamespace(
        n_input_tokens=None,
        n_cache_tokens=None,
        n_output_tokens=None,
        metadata=None,
    )
    result = {
        "thread_id": "programmatic:failed",
        "turn_id": "turn:failed",
        "status": "failed",
        "terminal_source": "event",
        "event_count": 4,
        "terminal": {"usage": None},
    }

    project_agent_context(
        context,
        result,
        harness_name="akasic-v4flash",
        harness_version="0.1.0",
        source_digest="sha256:test",
    )

    assert context.n_input_tokens is None
    assert context.n_cache_tokens is None
    assert context.n_output_tokens is None
    assert context.metadata["turn_status"] == "failed"
    assert context.metadata["usage_available"] is False
