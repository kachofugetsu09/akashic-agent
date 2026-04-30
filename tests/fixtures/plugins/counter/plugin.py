from agent.plugins import Plugin, on_before_turn
import logging

_log = logging.getLogger("plugin.counter")


class Counter(Plugin):
    name = "counter"

    @on_before_turn()
    async def count(self, event):
        n = self.context.kv_store.increment("turn_count")
        _log.info("counter: turn_count=%d plugin_id=%s", n, self.context.plugin_id)
        event.extra_metadata["turn_count"] = n
        return event
