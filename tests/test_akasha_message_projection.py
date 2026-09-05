from plugins.akasha.projection import project_samples
from plugins.turn_projection.plugin import TurnProjection
from session.log import MessageLog
from session.message import Input, Output, Control, ToolCall, ToolResult, CallRef


def test_projection_consumes_interrupted_inputs_once_and_keeps_sources_independent(tmp_path):
    log = MessageLog(tmp_path / "sessions.db")
    def writer(body, source="conversation", session="s", ref=None):
        return log.writer(session, author="author", source=source, body_types=(body,),
                          content={}, call_ref=ref, check_call=lambda call: None)
    try:
        writer(Input).append("u1", Input(()))
        writer(Control).append("stop1", Control("pause", 0))
        writer(Input).append("u2", Input(()))
        writer(Output, "wake").append("wake", Output((), "complete"))
        writer(Control).append("stop2", Control("pause", 2))
        writer(Input, "timer").append("timer-input", Input(()))
        writer(Input).append("u3", Input(()))
        writer(Output, "timer").append("timer-answer", Output((), "complete"))
        writer(Output).append("a", Output((), "complete"))
        writer(Input).append("open-tail", Input(()))
        samples = project_samples(log.catalog(), TurnProjection(), include=lambda session, source: True)
        assert [sample.ending.message_id for sample in samples] == ["wake", "timer-answer", "a"]
        assert [message.message_id for message in samples[-1].messages] == ["u1", "u2", "u3", "a"]
        assert [message.message_id for message in samples[1].messages] == ["timer-input", "timer-answer"]
        assert project_samples(log.catalog(), TurnProjection(), include=lambda session, source: True) == samples
    finally:
        log.close()


def test_fixed_heads_page_full_prefix_and_exclude_future_sessions_and_late_results(tmp_path):
    log = MessageLog(tmp_path / "sessions.db")
    def writer(body, session="s", ref=None):
        return log.writer(session, author="author", source="conversation", body_types=(body,),
                          content={}, call_ref=ref, check_call=lambda call: None)
    try:
        log.save_binding("tool", {"version": 1})
        writer(Input).append("old", Input(()))
        writer(Output).append("call", Output((ToolCall("tool", {}),), "continue"))
        writer(Control).append("abandon", Control("abandon", 1))
        for index in range(1005):
            writer(Input).append(f"u{index}", Input(()))
        writer(ToolResult, ref=CallRef("call", 0)).append("late", ToolResult(CallRef("call", 0), "success", ()))
        writer(Output).append("a", Output((), "complete"))
        heads = log.catalog().snapshot_heads()
        writer(Output, session="new").append("future-session", Output((), "complete"))
        writer(Input).append("future-input", Input(()))
        writer(Output).append("future-answer", Output((), "complete"))
        (sample,) = project_samples(log.catalog(), TurnProjection(), heads=heads, include=lambda session, source: True)
        assert sample.ending.message_id == "a"
        assert len(sample.messages) == 1006
        assert sample.messages[0].message_id == "u0"
        assert sample.observations == ()
        assert project_samples(log.catalog(), TurnProjection(), heads=heads, include=lambda session, source: False) == ()
    finally:
        log.close()
