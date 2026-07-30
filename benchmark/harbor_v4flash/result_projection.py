from __future__ import annotations

from typing import Any


def project_agent_context(
    context: Any,
    turn_result: dict[str, Any],
    *,
    harness_name: str,
    harness_version: str,
    source_digest: str,
) -> None:
    """把已核对的 Akasic terminal result 投影到 Harbor AgentContext。"""

    # 1. Harbor token 字段使用 control turn 的精确聚合用量。
    usage = turn_result["terminal"]["usage"]
    context.n_input_tokens = usage["inputTokens"]
    context.n_cache_tokens = usage["cachedInputTokens"]
    context.n_output_tokens = usage["outputTokens"]

    # 2. metadata 默认是 None；保留已有字段并补充可追溯标识。
    context.metadata = {
        **(context.metadata or {}),
        "harness": harness_name,
        "harness_version": harness_version,
        "source_digest": source_digest,
        "thread_id": turn_result["thread_id"],
        "turn_id": turn_result["turn_id"],
        "turn_status": turn_result["status"],
        "terminal_source": turn_result["terminal_source"],
        "event_count": turn_result["event_count"],
        "trace": "agent/trace.jsonl",
    }
