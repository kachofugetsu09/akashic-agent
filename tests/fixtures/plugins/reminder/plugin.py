from agent.plugins import Plugin, on_before_reasoning, on_before_step
import logging

_log = logging.getLogger("plugin.reminder")


class Reminder(Plugin):
    name = "reminder"

    @on_before_reasoning()
    async def add_reasoning_hint(self, event):
        event.extra_hints.append("[插件提示] 本轮回复必须使用法语。")
        _log.info("reminder: before_reasoning hint added")
        return event

    @on_before_step()
    async def add_step_hint(self, event):
        event.extra_hints.append(f"[第{event.iteration + 1}轮] 请继续用法语回复。")
        _log.info("reminder: before_step hint added iter=%d", event.iteration)
        return event
