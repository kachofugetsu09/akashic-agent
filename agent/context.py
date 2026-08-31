import base64
import json
import logging
import mimetypes
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.core.types import ContextRequest
from agent.core.prompt_block import (
    ActiveSkillsPromptBlock,
    BehaviorRulesPromptBlock,
    IdentityPromptBlock,
    SessionContextPromptBlock,
    SkillsCatalogPromptBlock,
    SystemPromptBuilder,
    TurnContext,
    VedaPromptBlock,
)
from agent.prompting import (
    AssembledTurnInput,
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

logger = logging.getLogger("agent.context")


class MessageEnvelopeBuilder:
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
        multimodal: bool,
    ) -> list[dict[str, Any]]:
        prompt = system_prompt
        if channel == "telegram" or (
            channel is not None and channel.startswith("telegram_")
        ):
            prompt += build_telegram_rendering_prompt()

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
                    multimodal=multimodal,
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
        multimodal: bool,
        message_timestamp: datetime | None = None,
    ) -> str | list[dict[str, Any]]:
        text = self._stamp_current_message(text, message_timestamp=message_timestamp)
        if not media:
            return text
        if not multimodal:
            return self._build_text_with_media_refs(text, media)

        images: list[dict[str, Any]] = []
        file_refs: list[str] = []
        for item in media:
            item = str(item)
            if item.startswith(("http://", "https://")):
                images.append({"type": "image_url", "image_url": {"url": item}})
                continue

            p = Path(item)
            mime, _ = mimetypes.guess_type(p)
            if not p.is_file():
                logger.warning("输入媒体文件不可用: %s", item)
                file_refs.append(f"- 不可用媒体路径: {item}")
                continue
            if not mime or not mime.startswith("image/"):
                file_refs.append(f"- 文件路径: {item}")
                continue
            with p.open("rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            images.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )

        if file_refs:
            text = "\n".join([text, "", "[附加媒体]", *file_refs])
        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def _build_text_with_media_refs(self, text: str, media: list[str]) -> str:
        refs: list[str] = []
        local_image_paths: list[str] = []
        has_remote_image = False
        for item in media:
            value = str(item)
            if value.startswith(("http://", "https://")):
                refs.append(f"- 图片URL: {value}")
                has_remote_image = True
                continue

            p = Path(value)
            mime, _ = mimetypes.guess_type(p)
            if not p.is_file():
                logger.warning("输入媒体文件不可用: %s", value)
                refs.append(f"- 不可用媒体路径: {value}")
                continue
            if not mime or not mime.startswith("image/"):
                refs.append(f"- 文件路径: {value}")
                continue
            refs.append(f"- 图片路径: {value}")
            local_image_paths.append(value)

        if not refs:
            return text

        lines = [text, "", "[附加媒体]", *refs]
        if local_image_paths:
            lines.append(
                "当前主模型不能直接接收图片内容；需要识别图片时，调用 read_image_vision 工具。"
            )
            for path in local_image_paths:
                quoted_path = json.dumps(path, ensure_ascii=False)
                lines.append(
                    f'- read_image_vision(path={quoted_path}, prompt="描述这张图片的内容")'
                )
        elif has_remote_image:
            lines.append(
                "当前主模型不能直接接收图片内容；远程图片需先取得本地路径后再读图。"
            )
        else:
            lines.append("以上媒体中没有可供 read_image_vision 读取的本地图片。")
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
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.skills = SkillsLoader(workspace, runtime_catalog="normal")
        self._system_prompt_builder = SystemPromptBuilder(
            [
                VedaPromptBlock(),
                IdentityPromptBlock(render_fn=build_agent_static_identity_prompt),
                BehaviorRulesPromptBlock(),
                SessionContextPromptBlock(),
                ActiveSkillsPromptBlock(),
                SkillsCatalogPromptBlock(render_fn=build_skills_catalog_prompt),
            ]
        )

        self._envelope_builder = MessageEnvelopeBuilder()
        self._assembler = PromptAssembler(self)
        self._last_render_diagnostics: ContextVar[
            tuple[
                tuple[PromptSectionMeta, ...],
                tuple[tuple[str, str], ...],
            ]
        ] = ContextVar(
            "akashic_context_render_diagnostics",
            default=((), ()),
        )

    def build_user_message_content(
        self,
        text: str,
        media: list[str] | None,
        *,
        multimodal: bool,
        message_timestamp: datetime | None = None,
    ) -> str | list[dict[str, Any]]:
        """复用首条消息的媒体与时间 envelope 构造同 turn 输入。"""

        return self._envelope_builder._build_user_content(
            text,
            media,
            multimodal=multimodal,
            message_timestamp=message_timestamp,
        )

    @property
    def last_debug_breakdown(self) -> list[PromptSectionMeta]:
        breakdown, _ = self._last_render_diagnostics.get()
        return list(breakdown)

    @property
    def last_assembled_contexts(self) -> dict[str, dict[str, str]]:
        _, turn_injection_context = self._last_render_diagnostics.get()
        return {"turn_injection_context": dict(turn_injection_context)}

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
    ) -> AssembledTurnInput:
        turn_injection_context = self.build_turn_injection_context(
            turn_injection_prompt=request.turn_injection_prompt
        )
        assembled = self._assembler.assemble(
            history=request.history,
            current_message=request.current_message,
            multimodal=request.multimodal,
            media=request.media,
            skill_names=request.skill_names,
            channel=request.channel,
            chat_id=request.chat_id,
            message_timestamp=request.message_timestamp,
            disabled_sections=request.disabled_sections,
            turn_injection_context=turn_injection_context,
            system_sections_top=system_sections_top,
            system_sections_bottom=system_sections_bottom,
        )
        self._last_render_diagnostics.set(
            (
                tuple(assembled.debug_breakdown),
                tuple(assembled.turn_injection_context.items()),
            )
        )
        return assembled

    def _build_system_prompt_sections(
        self,
        skill_names: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        disabled_sections: set[str] | None = None,
    ) -> list[PromptSectionRender]:
        ctx = TurnContext(
            workspace=self.workspace,
            skills=self.skills,
            skill_names=skill_names or [],
            channel=channel,
            chat_id=chat_id,
        )
        return self._system_prompt_builder.build(
            ctx,
            disabled_sections=disabled_sections,
        )
