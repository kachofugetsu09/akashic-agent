from __future__ import annotations


def build_reply_inbound_text(
    user_text: str,
    reply_text: str,
    *,
    sender_label: str | None = None,
) -> str:
    """把引用目标和当前消息组织为统一的 Agent 入站上下文。"""

    source = reply_text.strip()
    if not source:
        raise ValueError("引用上下文缺少被回复消息文本")
    sender = f"（来自 {sender_label}）" if sender_label else ""
    return (
        "【你正在回复一条历史消息】\n"
        f"被回复消息{sender}：\n"
        f"{source}\n\n"
        "【你当前新消息】\n"
        f"{user_text.strip()}"
    ).strip()


__all__ = ["build_reply_inbound_text"]
