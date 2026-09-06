from plugins.programmatic.result import read_result
from plugins.turn_projection.plugin import TurnProjection
from session.log import MessageLog
from session.message import ContentPart, Control, Input, Output
from tests.test_message_log import writer


def test_result_recovers_exact_input_across_pause_new_input_and_other_source(tmp_path):
    path = tmp_path / "messages.db"
    log = MessageLog(path)
    projection = TurnProjection()
    try:
        source = writer(log, source="programmatic", bodies=(Input, Output, Control))
        other = writer(log, source="conversation", bodies=(Output,))
        source.append("first", Input((ContentPart("text", "原请求"),)))
        source.append("paused", Control("pause", 0))
        assert read_result(log.reader("s"), "first", projection)["status"] == "pause"
        source.append("more", Input((ContentPart("text", "补充条件"),)))
        other.append("other-answer", Output((ContentPart("text", "其他来源"),), "complete"))
        source.append("step", Output((ContentPart("text", "继续工作"),), "continue"))
        assert read_result(log.reader("s"), "first", projection)["status"] == "open"
        source.append("answer", Output((ContentPart("text", "完整结果"),), "complete"))
        source.append("next", Input((ContentPart("text", "另一次请求"),)))
        source.append("failed", Control("failure", 6, "真实错误"))
        assert read_result(log.reader("s"), "next", projection)["status"] == "failure"
    finally:
        log.close()

    restored = MessageLog(path)
    try:
        for identity in ("first", "more"):
            result = read_result(restored.reader("s"), identity, TurnProjection())
            assert (result["status"], result["ending_message_id"]) == ("complete", "answer")
        assert len(restored.reader("s").snapshot()) == 8
    finally:
        restored.close()
