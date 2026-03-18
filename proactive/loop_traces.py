from proactive.loop import ProactiveLoop


class ProactiveLoopTraceMixin:
    _trace_proactive_memory_retrieve = ProactiveLoop._trace_proactive_memory_retrieve
    _trace_proactive_config_snapshot = ProactiveLoop._trace_proactive_config_snapshot
    _trace_proactive_rate_decision = ProactiveLoop._trace_proactive_rate_decision
    _append_trace_line = ProactiveLoop._append_trace_line


__all__ = ["ProactiveLoopTraceMixin"]
