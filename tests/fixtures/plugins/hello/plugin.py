from agent.plugins import Plugin, on_before_turn, on_after_step
import logging

_log = logging.getLogger("plugin.hello")

after_step_calls: list[str] = []


class Hello(Plugin):
    name = "hello"
    version = "0.1.0"

    @on_before_turn()
    async def on_before(self, event):
        _log.info("hello: BeforeTurn session=%s content=%.40s",
                  event.session_key, event.content)
        event.extra_metadata["hello_touched"] = True
        return event

    @on_after_step()
    async def on_after(self, event):
        _log.info("hello: AfterStep iter=%d tools=%s",
                  event.iteration, event.tools_called)
        after_step_calls.append(event.session_key)
