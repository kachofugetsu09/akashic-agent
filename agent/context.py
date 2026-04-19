import base64
import logging
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from agent.core.types import ContextRenderResult, ContextRequest
from agent.core.prompt_block import (
    ActiveSkillsPromptBlock,
    BehaviorRulesPromptBlock,
    IdentityPromptBlock,
    LongTermMemoryPromptBlock,
    MemesPromptBlock,
    MemoryBlockPromptBlock,
    RecentContextPromptBlock,
    SelfModelPromptBlock,
    SessionContextPromptBlock,
    SkillsCatalogPromptBlock,
    SystemPromptBuildResult,
    SystemPromptBuilder,
    TurnContext,
)
from agent.memes.catalog import MemeCatalog
from agent.prompting import (
    PromptAssembler,
    PromptSectionMeta,
    build_turn_injection_message,
)
from agent.skills import SkillsLoader
from prompts.agent import (
    build_agent_static_identity_prompt,
    build_current_message_time_envelope,
    build_skills_catalog_prompt,
    build_telegram_rendering_prompt,
)

if TYPE_CHECKING:
    from core.memory.profile import ProfileReader

logger = logging.getLogger("agent.context")


class ChannelPolicy(Protocol):
    channel: str

    def augment_system_prompt(self, prompt: str) -> str: ...


class TelegramChannelPolicy:
    channel = "telegram"

    def augment_system_prompt(self, prompt: str) -> str:
        return prompt + build_telegram_rendering_prompt()


class MessageEnvelopeBuilder:
    def __init__(self, policies: dict[str, ChannelPolicy] | None = None):
        self._policies = policies or {}

    def build(
        self,
        *,
        history: list[dict[str, Any]],
        current_message: str,
        system_prompt: str,
        turn_injection_context: dict[str, str] | None,
        channel: str | None,
        message_timestamp: datetime | None,
        media: list[str] | None,
    ) -> list[dict[str, Any]]:
        prompt = system_prompt
        if channel:
            policy = self._policies.get(channel)
            if policy is not None:
                prompt = policy.augment_system_prompt(prompt)

        # 顺序是有意设计的：system prompt -> turn injection -> history -> 当前用户消息。
        messages: list[dict[str, Any]] = [{"role": "system", "content": prompt}]
        for text in (turn_injection_context or {}).values():
            if text.strip():
                messages.append(build_turn_injection_message(text))
        messages.extend(history)
        messages.append(
            {
                "role": "user",
                "content": self._build_user_content(
                    current_message,
                    media,
                    message_timestamp=message_timestamp,
                ),
            }
        )
        return messages

    def _build_user_content(
        self,
        text: str,
        media: list[str] | None,
        *,
        message_timestamp: datetime | None = None,
    ) -> str | list[dict[str, Any]]:
        text = self._stamp_current_message(text, message_timestamp=message_timestamp)
        if not media:
            return text

        images = []
        for item in media:
            item = str(item)
            if item.startswith(("http://", "https://")):
                images.append({"type": "image_url", "image_url": {"url": item}})
                continue

            p = Path(item)
            mime, _ = mimetypes.guess_type(p)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            with p.open("rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            images.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def _stamp_current_message(
        self,
        text: str,
        *,
        message_timestamp: datetime | None = None,
    ) -> str:
        stripped = text.lstrip()
        if not stripped:
            return text
        if stripped.startswith("[当前消息时间:"):
            return text
        stamp = build_current_message_time_envelope(message_timestamp=message_timestamp)
        return f"{stamp}\n{text}"


class ContextBuilder:
    def __init__(self, workspace: Path, memory: "ProfileReader"):
        self.workspace = workspace
        self.skills = SkillsLoader(workspace)
        self.memory = memory
        self._system_prompt_builder = SystemPromptBuilder(
            [
                IdentityPromptBlock(render_fn=build_agent_static_identity_prompt),
                BehaviorRulesPromptBlock(),
                MemoryBlockPromptBlock(),
                LongTermMemoryPromptBlock(),
                SelfModelPromptBlock(),
                RecentContextPromptBlock(),
                SessionContextPromptBlock(),
                ActiveSkillsPromptBlock(),
                MemesPromptBlock(MemeCatalog(workspace / "memes")),
                SkillsCatalogPromptBlock(render_fn=build_skills_catalog_prompt),
            ]
        )
        self._envelope_builder = MessageEnvelopeBuilder(
            policies={TelegramChannelPolicy.channel: TelegramChannelPolicy()}
        )
        self._assembler = PromptAssembler(self)
        self._last_debug_breakdown: list[PromptSectionMeta] = []
        self._last_assembled_contexts: dict[str, dict[str, str]] = {
            "turn_injection_context": {},
        }

    @property
    def last_debug_breakdown(self) -> list[PromptSectionMeta]:
        return list(self._last_debug_breakdown)

    @property
    def last_assembled_contexts(self) -> dict[str, dict[str, str]]:
        return {
            "turn_injection_context": dict(
                self._last_assembled_contexts["turn_injection_context"]
            ),
        }

    def build_turn_injection_context(
        self,
        *,
        turn_injection_prompt: str | None = None,
    ) -> dict[str, str]:
        if not turn_injection_prompt:
            return {}
        return {"turn_injection": turn_injection_prompt}

    def render(self, request: ContextRequest) -> ContextRenderResult:
        turn_injection_context = self.build_turn_injection_context(
            turn_injection_prompt=request.turn_injection_prompt
        )
        assembled = self._assembler.assemble(
            history=request.history,
            current_message=request.current_message,
            media=request.media,
            skill_names=request.skill_names,
            channel=request.channel,
            chat_id=request.chat_id,
            message_timestamp=request.message_timestamp,
            retrieved_memory_block=request.retrieved_memory_block,
            disabled_sections=request.disabled_sections,
            turn_injection_context=turn_injection_context,
        )
        self._last_debug_breakdown = assembled.debug_breakdown
        self._last_assembled_contexts = {
            "turn_injection_context": dict(assembled.turn_injection_context),
        }
        return ContextRenderResult(
            system_prompt=assembled.system_prompt,
            turn_injection_context=dict(assembled.turn_injection_context),
            messages=list(assembled.messages),
            debug_breakdown=list(assembled.debug_breakdown),
        )

    def _build_system_prompt_result(
        self,
        skill_names: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        message_timestamp: "datetime | None" = None,
        retrieved_memory_block: str = "",
        disabled_sections: set[str] | None = None,
    ) -> SystemPromptBuildResult:
        ctx = TurnContext(
            workspace=self.workspace,
            memory=self.memory,
            skills=self.skills,
            skill_names=skill_names or [],
            channel=channel,
            chat_id=chat_id,
            message_timestamp=message_timestamp,
            retrieved_memory_block=retrieved_memory_block,
        )
        built = self._system_prompt_builder.build(
            ctx,
            disabled_sections=disabled_sections,
        )
        self._last_debug_breakdown = built.debug_breakdown
        if built.debug_breakdown:
            logger.info(
                "prompt breakdown: %s",
                ", ".join(
                    f"{item.name}[chars={item.chars},tokens~={item.est_tokens},static={int(item.is_static)},cache={int(item.cache_hit)}]"
                    for item in built.debug_breakdown
                ),
            )
        return built
