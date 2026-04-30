from agent.plugins import Plugin, on_tool_call, on_tool_result


class Audit(Plugin):
    name = "audit"

    def __init__(self) -> None:
        self.before_tool_calls: list[str] = []
        self.after_tool_results: list[tuple[str, str]] = []

    @on_tool_call()
    def record_call(self, event: object) -> None:
        self.before_tool_calls.append(event.tool_name)  # type: ignore[union-attr]

    @on_tool_result()
    def record_result(self, event: object) -> None:
        self.after_tool_results.append((event.tool_name, event.status))  # type: ignore[union-attr]
