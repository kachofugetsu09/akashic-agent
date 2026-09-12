"""Frozen Wake phase completion predicate used by the legacy migration."""

from agent.plugin_composition.messages import MessageReader
from agent.plugin_contracts import Control, Input, Message, Output

from .wake_request import Request, Stage


def finished(reader: MessageReader, request: Request, stage: Stage) -> Message | None:
    """Find the first terminal Wake message after the requested phase input."""
    start = reader.get(request.phase_id(stage))
    if start is None:
        return None
    for message in reader.snapshot():
        if message.seq <= start.seq or message.source != "wake":
            continue
        if isinstance(message.body, Output) and message.body.finish != "continue":
            return message
        if isinstance(message.body, Control) and message.body.action in {"pause", "failure"}:
            return message
        if isinstance(message.body, Input):
            raise ValueError("Wake 前一阶段未结束便出现下一输入")
    return None
