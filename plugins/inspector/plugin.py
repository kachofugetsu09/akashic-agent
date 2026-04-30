from agent.plugins import (
    Plugin,
    on_before_turn,
    on_before_reasoning,
    on_before_step,
    on_after_step,
    on_after_reasoning,
    on_after_turn,
)
import logging

_log = logging.getLogger("plugin.inspector")


class Inspector(Plugin):
    name = "inspector"

    @on_before_turn()
    async def h1(self, event):
        _log.info("1.before_turn  session=%s skills=%s", event.session_key, event.skill_names)
        return event

    @on_before_reasoning()
    async def h2(self, event):
        _log.info("2.before_reasoning  hints=%s", event.extra_hints)
        return event

    @on_before_step()
    async def h3(self, event):
        _log.info("3.before_step  iter=%d tokens~=%d", event.iteration, event.input_tokens_estimate)
        return event

    @on_after_step()
    async def h4(self, event):
        _log.info("4.after_step  iter=%d tools=%s has_more=%s", event.iteration, event.tools_called, event.has_more)

    @on_after_reasoning()
    async def h5(self, event):
        _log.info("5.after_reasoning  tools=%s reply=%.60s", event.tools_used, event.reply)
        return event

    @on_after_turn()
    async def h6(self, event):
        _log.info("6.after_turn  dispatch=%s", event.will_dispatch)
