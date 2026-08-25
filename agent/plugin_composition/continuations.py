from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime

from agent.plugin_composition.model import ServiceKey
from agent.turn_effects import PostCommitEffect, set_post_commit_effect

InboundPublisher = Callable[[object], Awaitable[None]]


class PluginContinuations:
    """Submit one internal follow-up Message without exposing MessageBus."""

    def __init__(self, publisher: InboundPublisher | None) -> None:
        self._publisher = publisher

    @classmethod
    def candidate_validation(cls) -> PluginContinuations:
        return cls(None)

    @property
    def formal(self) -> bool:
        return self._publisher is not None

    async def submit(
        self,
        *,
        channel: str,
        chat_id: str,
        sender: str,
        content: str,
        timestamp: datetime | None = None,
    ) -> None:
        """Publish one stateless internal Message to an existing conversation lane."""

        publisher = self._publisher
        if publisher is None:
            raise RuntimeError("candidate 验证期禁止提交 continuation")
        if not channel.strip() or not chat_id.strip() or not sender.strip():
            raise ValueError("continuation route 不能为空")
        if not content.strip():
            raise ValueError("continuation content 不能为空")
        from bus.events import InboundMessage

        metadata: dict[str, object] = {
            "omit_user_turn": True,
            "disabled_prompt_sections": ["memory"],
        }
        set_post_commit_effect(metadata, PostCommitEffect.SUPPRESS)
        await publisher(
            InboundMessage(
                channel=channel,
                sender=sender,
                chat_id=chat_id,
                content=content,
                timestamp=timestamp or datetime.now(),
                media=[],
                metadata=metadata,
            )
        )


CONTINUATIONS = ServiceKey[PluginContinuations]("core.continuations")
