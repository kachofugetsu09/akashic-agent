from agent.plugins import EventTrigger, Plugin, PluginJobSpec
from bus.events_lifecycle import TurnCommitted


class Jobber(Plugin):
    name = "jobber"

    def jobs(self):
        return [
            PluginJobSpec(
                id="on_turn",
                triggers=[EventTrigger(TurnCommitted)],
                handler=self.on_turn,
            )
        ]

    async def on_turn(self, ctx):
        text = await ctx.llm.generate_text(prompt="hello")
        self.context.kv_store.set("last_job", {
            "text": text,
            "reason": ctx.reason,
            "has_event": ctx.event is not None,
            "context_llm": self.context.llm is not None,
        })
