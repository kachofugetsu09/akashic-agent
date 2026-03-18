from proactive.loop import ProactiveLoop


class ProactiveLoopTriggerMixin:
    trigger_skill_action = ProactiveLoop.trigger_skill_action
    _pick_skill_action = ProactiveLoop._pick_skill_action


__all__ = ["ProactiveLoopTriggerMixin"]
