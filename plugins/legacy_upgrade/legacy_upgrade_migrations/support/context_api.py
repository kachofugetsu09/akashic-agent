"""Frozen context source range check used by legacy summary import."""

from agent.plugin_contracts import Message


def summary_range(snapshot: tuple[Message, ...], source_message_ids: tuple[str, ...]) -> range:
    """Locate the exact contiguous message span covered by a historical summary."""
    identities = tuple(message.message_id for message in snapshot)
    if not source_message_ids or source_message_ids[0] not in identities:
        raise ValueError("摘要来源缺少实际消息")
    start = identities.index(source_message_ids[0])
    end = start + len(source_message_ids)
    if identities[start:end] != source_message_ids:
        raise ValueError("摘要来源不等于实际连续消息范围")
    return range(start, end)
