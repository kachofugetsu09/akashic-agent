from __future__ import annotations

import json


class ProviderContextBudgetStub:
    """Provide the context-budget half of the test provider contract."""

    context_window = 1_000_000

    def estimate_context_tokens(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> int:
        return max(
            1,
            len(json.dumps([messages, tools], ensure_ascii=False)) // 3,
        )

    def estimate_appended_message_tokens(self, messages: list[dict]) -> int:
        if not messages:
            return 0
        return max(1, len(json.dumps(messages, ensure_ascii=False)) // 3)
