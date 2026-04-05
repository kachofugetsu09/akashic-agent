import base64
import logging
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from agent.core.types import ContextRenderResult, ContextRequest
from agent.core.prompt_block import (
    ActiveSkillsPromptBlock,
    IdentityPromptBlock,
    LongTermMemoryPromptBlock,
    MemesPromptBlock,
    MemoryBlockPromptBlock,
    SelfModelPromptBlock,
    SkillsCatalogPromptBlock,
    SystemPromptBuildResult,
    SystemPromptBuilder,
    TurnContext,
)
from agent.memes.catalog import MemeCatalog
from agent.prompting import (
    PromptAssembler,
    PromptSectionMeta,
    build_runtime_guard_message,
    build_system_context_message,
)
from agent.skills import SkillsLoader
from prompts.agent import (
    build_agent_environment_prompt,
    build_agent_request_time_prompt,
    build_agent_static_identity_prompt,
    build_current_session_prompt,
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
        system_context: dict[str, str] | None,
        runtime_guard_context: dict[str, str] | None,
        channel: str | None,
        media: list[str] | None,
    ) -> list[dict[str, Any]]:
        prompt = system_prompt
        if channel:
            policy = self._policies.get(channel)
            if policy is not None:
                prompt = policy.augment_system_prompt(prompt)

        # 顺序是有意设计的：system prompt -> side context -> runtime guard -> history -> 当前用户消息。
        messages: list[dict[str, Any]] = [{"role": "system", "content": prompt}]
        for text in (system_context or {}).values():
            if text.strip():
                messages.append(build_system_context_message(text))
        for text in (runtime_guard_context or {}).values():
            if text.strip():
                messages.append(build_runtime_guard_message(text))
        messages.extend(history)
        messages.append(
            {
                "role": "user",
                "content": self._build_user_content(current_message, media),
            }
        )
        return messages

    def _build_user_content(
        self, text: str, media: list[str] | None
    ) -> str | list[dict[str, Any]]:
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


class ContextBuilder:
    def __init__(self, workspace: Path, memory: "ProfileReader"):
        self.workspace = workspace
        self.skills = SkillsLoader(workspace)
        self.memory = memory
        self._system_prompt_builder = SystemPromptBuilder(
            [
                IdentityPromptBlock(render_fn=build_agent_static_identity_prompt),
                MemoryBlockPromptBlock(),
                LongTermMemoryPromptBlock(),
                SelfModelPromptBlock(),
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
            "system_context": {},
            "runtime_guard_context": {},
        }

    @property
    def last_debug_breakdown(self) -> list[PromptSectionMeta]:
        return list(self._last_debug_breakdown)

    @property
    def last_assembled_contexts(self) -> dict[str, dict[str, str]]:
        return {
            "system_context": dict(self._last_assembled_contexts["system_context"]),
            "runtime_guard_context": dict(
                self._last_assembled_contexts["runtime_guard_context"]
            ),
        }

    def build_system_context(
        self,
        *,
        channel: str | None = None,
        chat_id: str | None = None,
        message_timestamp: "datetime | None" = None,
    ) -> dict[str, str]:
        # 这里只放“本轮系统事实”，避免把 request_time / session 之类的易变信息塞回主 prompt。
        context = {
            "request_time": build_agent_request_time_prompt(
                message_timestamp=message_timestamp
            ),
            "environment": build_agent_environment_prompt(),
        }
        if channel and chat_id:
            context["current_session"] = build_current_session_prompt(
                channel=channel,
                chat_id=chat_id,
            ).strip()
        return context

    def build_runtime_guard_context(
        self,
        *,
        preflight_prompt: str | None = None,
    ) -> dict[str, str]:
        # runtime guard 和 system_context 同样用 system role，
        # 但语义上是“本轮约束”，由调用方按 turn 动态注入。
        if not preflight_prompt:
            return {}
        return {"preflight": preflight_prompt}

    def render(self, request: ContextRequest) -> ContextRenderResult:
        runtime_guard_context = self.build_runtime_guard_context(
            preflight_prompt=request.preflight_prompt
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
            runtime_guard_context=runtime_guard_context,
        )
        self._last_debug_breakdown = assembled.debug_breakdown
        self._last_assembled_contexts = {
            "system_context": dict(assembled.system_context),
            "runtime_guard_context": dict(assembled.runtime_guard_context),
        }
        return ContextRenderResult(
            system_prompt=assembled.system_prompt,
            system_context=dict(assembled.system_context),
            runtime_guard_context=dict(assembled.runtime_guard_context),
            messages=list(assembled.messages),
            debug_breakdown=list(assembled.debug_breakdown),
        )

    def _build_system_prompt_result(
        self,
        skill_names: list[str] | None = None,
        message_timestamp: "datetime | None" = None,
        retrieved_memory_block: str = "",
        disabled_sections: set[str] | None = None,
    ) -> SystemPromptBuildResult:
        ctx = TurnContext(
            workspace=self.workspace,
            memory=self.memory,
            skills=self.skills,
            skill_names=skill_names or [],
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

    def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        message_timestamp: "datetime | None" = None,
        retrieved_memory_block: str = "",
        disabled_sections: set[str] | None = None,
    ) -> str:
        return self.render(
            ContextRequest(
                history=[],
                current_message="",
                skill_names=skill_names,
                message_timestamp=message_timestamp,
                retrieved_memory_block=retrieved_memory_block,
                disabled_sections=disabled_sections,
            )
        ).system_prompt

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        media: list[str] | None = None,
        skill_names: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        message_timestamp: "datetime | None" = None,
        retrieved_memory_block: str = "",
        disabled_sections: set[str] | None = None,
        runtime_guard_context: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        preflight_prompt = None
        if runtime_guard_context:
            preflight_prompt = runtime_guard_context.get("preflight")
        return self.render(
            ContextRequest(
                history=history,
                current_message=current_message,
                media=media,
                skill_names=skill_names,
                channel=channel,
                chat_id=chat_id,
                message_timestamp=message_timestamp,
                retrieved_memory_block=retrieved_memory_block,
                disabled_sections=disabled_sections,
                preflight_prompt=preflight_prompt,
            )
        ).messages

    def _build_user_content(
        self, text: str, media: list[str] | None
    ) -> str | list[dict[str, Any]]:
        return self._envelope_builder._build_user_content(text, media)

    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> list[dict[str, Any]]:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": result,
            }
        )
        return messages

    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
    ) -> list[dict[str, Any]]:
        msg: dict[str, Any] = {"role": "assistant"}
        if content:
            msg["content"] = content
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if reasoning_content:
            msg["reasoning_content"] = reasoning_content
        messages.append(msg)
        return messages
