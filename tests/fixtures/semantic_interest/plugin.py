from __future__ import annotations

from collections.abc import Sequence

from agent.plugin_composition import CONVERSATION_SEMANTIC_INTEREST, Context


api_version = 3
name = "semantic_interest"
version = "1.0.0"
inject = ()


class ZeroSemanticInterest:
    async def score(
        self,
        texts: Sequence[str],
        *,
        cutoff: str,
    ) -> tuple[float, ...]:
        _ = cutoff
        return tuple(0.0 for _ in texts)


async def apply(ctx: Context, config: object) -> None:
    _ = config
    _ = await ctx.provide(CONVERSATION_SEMANTIC_INTEREST, ZeroSemanticInterest())
