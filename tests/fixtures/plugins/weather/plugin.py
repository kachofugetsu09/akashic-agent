from agent.plugins import Plugin
from agent.plugins.decorators import tool


class Weather(Plugin):
    name = "weather"
    version = "0.1.0"

    @tool(name="get_weather", risk="read-only", always_on=False,
          search_hint="get current weather for a city")
    async def get_weather(self, event, city: str) -> str:
        """Get current weather for a city.

        Args:
            city: The city name to query weather for.
        """
        return f"{city}: 晴, 22°C (由 weather 插件提供)"
