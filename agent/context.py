import base64
import json
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
    MemoryBlockPromptBlock,
    RecentContextPromptBlock,
    SelfModelPromptBlock,
    SessionContextPromptBlock,
    SkillsCatalogPromptBlock,
    SystemPromptBuildResult,
    SystemPromptBuilder,
    TurnContext,
)
from agent.prompting import (
    PromptAssembler,
    PromptSectionMeta,
    PromptSectionRender,
    build_context_frame_message,
)
from agent.skills import SkillsLoader
from prompts.agent import (
    build_agent_static_identity_prompt,
    build_current_message_time_envelope,
    build_skills_catalog_prompt,
    build_telegram_rendering_prompt,
)

if TYPE_CHECKING:
    from core.memory.markdown import MemoryProfileApi

logger = logging.getLogger("agent.context")


class ChannelPolicy(Protocol):
    channel: str

    def augment_system_prompt(self, prompt: str) -> str: ...


class TelegramChannelPolicy:
    channel = "telegram"

    def augment_system_prompt(self, prompt: str) -> str:
        return prompt + build_telegram_rendering_prompt()


class MessageEnvelopeBuilder:
    def __init__(
        self,
        policies: dict[str, ChannelPolicy] | None = None,
        *,
        multimodal: bool = True,
        vl_available: bool = False,
    ):
        self._policies = policies or {}
        self._multimodal = multimodal
        self._vl_available = vl_available

    def set_media_capabilities(
        self,
        *,
        multimodal: bool,
        vl_available: bool,
    ) -> None:
        self._multimodal = multimodal
        self._vl_available = vl_available

    def build(
        self,
        *,
        history: list[dict[str, Any]],
        current_message: str,
        system_prompt: str,
        context_frame: str,
        channel: str | None,
        message_timestamp: datetime | None,
        media: list[str] | None,
    ) -> list[dict[str, Any]]:
        prompt = system_prompt
        if channel:
            policy = self._policies.get(channel)
            if policy is not None:
                prompt = policy.augment_system_prompt(prompt)

        # 顺序是有意设计的：stable system -> history -> context frame -> 当前用户消息。
        messages: list[dict[str, Any]] = [{"role": "system", "content": prompt}]
        messages.extend(history)
        if context_frame.strip():
            messages.append(build_context_frame_message(context_frame))
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
        if not self._multimodal:
            return self._build_text_with_media_refs(text, media)

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

    def _build_text_with_media_refs(self, text: str, media: list[str]) -> str:
        refs: list[str] = []
        local_image_paths: list[str] = []
        for item in media:
            value = str(item)
            if value.startswith(("http://", "https://")):
                refs.append(f"- 图片URL: {value}")
                continue

            p = Path(value)
            mime, _ = mimetypes.guess_type(p)
            if not p.is_file() or (mime and not mime.startswith("image/")):
                continue
            refs.append(f"- 图片路径: {value}")
            local_image_paths.append(value)

        if not refs:
            return text

        lines = [text, "", "[附加媒体]", *refs]
        if self._vl_available and local_image_paths:
            lines.append(
                "当前主模型不能直接接收图片内容；需要识别图片时，调用 read_image_vision 工具。"
            )
            for path in local_image_paths:
                quoted_path = json.dumps(path, ensure_ascii=False)
                lines.append(
                    f'- read_image_vision(path={quoted_path}, prompt="描述这张图片的内容")'
                )
        elif self._vl_available:
            lines.append(
                "当前主模型不能直接接收图片内容；远程图片需先取得本地路径后再读图。"
            )
        else:
            lines.append("当前主模型不能直接接收图片内容，且未配置 VL 视觉模型。")
        return "\n".join(lines)

    def _stamp_current_message(
        self,
        text: str,
        *,
        message_timestamp: datetime | None = None,
    ) -> str:
        stripped = text.lstrip()
        if not stripped:
            return build_current_message_time_envelope(
                message_timestamp=message_timestamp
            )
        if stripped.startswith("[当前消息时间:"):
            return text
        stamp = build_current_message_time_envelope(message_timestamp=message_timestamp)
        return f"{stamp}\n{text}"


class ContextBuilder:
    def __init__(
        self,
        workspace: Path,
        memory: "MemoryProfileApi",
        *,
        multimodal: bool = True,
        vl_available: bool = False,
    ):
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
                SkillsCatalogPromptBlock(render_fn=build_skills_catalog_prompt),
            ]
        )
        self._envelope_builder = MessageEnvelopeBuilder(
            policies={TelegramChannelPolicy.channel: TelegramChannelPolicy()},
            multimodal=multimodal,
            vl_available=vl_available,
        )
        self._assembler = PromptAssembler(self)
        self._last_debug_breakdown: list[PromptSectionMeta] = []
        self._last_assembled_contexts: dict[str, dict[str, str]] = {
            "turn_injection_context": {},
        }

    def set_media_capabilities(
        self,
        *,
        multimodal: bool,
        vl_available: bool,
    ) -> None:
        self._envelope_builder.set_media_capabilities(
            multimodal=multimodal,
            vl_available=vl_available,
        )

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

    def render(
        self,
        request: ContextRequest,
        *,
        system_sections_top: list[PromptSectionRender] | None = None,
        system_sections_bottom: list[PromptSectionRender] | None = None,
    ) -> ContextRenderResult:
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
            system_sections_top=system_sections_top,
            system_sections_bottom=system_sections_bottom,
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
            retrieved_memory_block=retrieved_memory_block,
        )
        built = self._system_prompt_builder.build(
            ctx,
            disabled_sections=disabled_sections,
        )
        self._last_debug_breakdown = built.debug_breakdown
        return built
