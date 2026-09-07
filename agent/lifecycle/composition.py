"""Legacy event-key names kept for the programmatic control Gate fixture."""

from agent.plugin_composition import SerialEventKey


# The active turn runtime owns its composition context. These names remain
# importable only because the isolated control Gate still creates a plugin that
# declares one of the old event keys; no runtime phase forwards through here.
PROMPT_RENDER_EVENT = SerialEventKey[object, object]("turn.prompt_render")
AFTER_REASONING_PREPROCESS_EVENT = SerialEventKey[object, object](
    "turn.after_reasoning.preprocess"
)
AFTER_REASONING_CLEANUP_EVENT = SerialEventKey[object, object](
    "turn.after_reasoning.cleanup"
)
