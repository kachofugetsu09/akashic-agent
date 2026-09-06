from __future__ import annotations

from plugins.turn_projection.plugin import TurnProjection
from session.log import MessageReader
from session.message import Control, Input


def read_result(reader: MessageReader, input_id: str, projection: TurnProjection) -> dict[str, object]:
    """从同一日志快照定位原 Input 的结果，不把空闲或晚到工具当作成功。"""
    # 1. 投影和控制判定共用一个前缀，避免读到不同时间的结束边界。
    messages = reader.snapshot()
    target = next((message for message in messages if message.message_id == input_id), None)
    if target is None or target.source != "programmatic" or not isinstance(target.body, Input):
        raise ValueError("结果查询必须引用当前 Session 的 programmatic Input")
    turn = next(turn for turn in projection.project(messages, target.source)
                if input_id in turn.message_ids)
    result: dict[str, object] = {
        "version": 2, "session_id": reader.session_id, "input_id": input_id,
        "status": turn.status, "ending_message_id": turn.ending_message_id,
        "through_seq": messages[-1].seq,
    }
    if turn.status != "open":
        result["ending_seq"] = next(message.seq for message in messages
                                    if message.message_id == turn.ending_message_id)
        return result

    # 2. pause/failure 不关闭 Turn，但明确结束本次等待；resume 恢复等待。
    controls = [message for message in messages if message.source == target.source
                and isinstance(message.body, Control) and message.body.through_seq >= target.seq]
    if controls:
        latest = controls[-1]
        assert isinstance(latest.body, Control)
        newest_input = max(message.seq for message in messages
                           if message.source == target.source and isinstance(message.body, Input))
        if latest.body.action in {"pause", "failure"} and latest.body.through_seq >= newest_input:
            result.update(status=latest.body.action, ending_message_id=latest.message_id,
                          ending_seq=latest.seq, reason=latest.body.reason)
    return result
