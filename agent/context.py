import base64
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prompts.agent import (
    build_agent_identity_prompt,
    build_current_session_prompt,
    build_skills_catalog_prompt,
    build_sop_index_prompt,
    build_telegram_rendering_prompt,
)
from agent.skills import SkillsLoader

if TYPE_CHECKING:
    from core.memory.port import MemoryPort


class ContextBuilder:
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md"]

    def __init__(self, workspace: Path, memory: "MemoryPort"):
        self.workspace = workspace
        self.skills = SkillsLoader(workspace)
        self.memory = memory

    def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        message_timestamp: "datetime | None" = None,
        retrieved_memory_block: str = "",
    ) -> str:
        parts = []
        # 核心identity
        parts.append(self._get_identity(message_timestamp))

        # memory v2 检索命中块（紧接 identity 后，高优先级）
        if retrieved_memory_block:
            parts.append(retrieved_memory_block)

        memory = self.memory.get_memory_context()
        if memory:
            parts.append(memory)

        # Akashic 自我认知（人格/关系理解）
        self_content = self.memory.read_self()
        if self_content:
            parts.append(f"## Akashic 自我认知\n\n{self_content}")

        # SOP 索引（用于维护：新增/修改 SOP 时必须对照此索引操作）
        # 注意：SOP 执行内容已由 memory2 向量检索注入，无需在此 read_file 读取 SOP 全文
        sop_readme = self.workspace / "sop" / "README.md"
        if sop_readme.exists():
            try:
                sop_index = sop_readme.read_text(encoding="utf-8").strip()
                parts.append(build_sop_index_prompt(sop_index))
            except Exception:
                pass

        # 技能渐进式加载：
        # 第一步：always 技能 + 本轮显式请求的技能 → 直接内嵌完整正文
        always_skills = self.skills.get_always_skills()
        skills_to_load: list[str] = []
        seen: set[str] = set()
        for name in [*always_skills, *(skill_names or [])]:
            if name in seen:
                continue
            seen.add(name)
            skills_to_load.append(name)
        if skills_to_load:
            always_content = self.skills.load_skills_for_context(skills_to_load)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        # 第二步：其余技能注入摘要（名称/描述/路径/可用状态），
        # 模型识别到任务匹配时，通过 read_file 读取对应 SKILL.md 获取完整指令
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(build_skills_catalog_prompt(skills_summary))

        return "\n\n---\n\n".join(parts)

    def _get_identity(self, message_timestamp: "datetime | None" = None) -> str:
        return build_agent_identity_prompt(
            workspace=self.workspace,
            message_timestamp=message_timestamp,
        )

    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []

        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")

        return "\n\n".join(parts) if parts else ""

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
    ) -> list[dict[str, Any]]:
        """
        Build the complete message list for an LLM call.

        Args:
            history: Previous conversation messages.
            current_message: The new user message.
            media: Optional list of local file paths for images/media.
            channel: Current channel (telegram, feishu, etc.).
            chat_id: Current chat/user ID.
            retrieved_memory_block: memory v2 检索命中块。

        Returns:
            List of messages including system prompt.
        """
        messages = []
        system_prompt = self.build_system_prompt(
            skill_names=skill_names,
            message_timestamp=message_timestamp,
            retrieved_memory_block=retrieved_memory_block,
        )
        if channel and chat_id:
            system_prompt += build_current_session_prompt(
                channel=channel,
                chat_id=chat_id,
            )
        if channel == "telegram":
            system_prompt += build_telegram_rendering_prompt()
        messages.append({"role": "system", "content": system_prompt})

        # History
        messages.extend(history)

        # Current message (with optional image attachments)
        user_content = self._build_user_content(current_message, media)
        messages.append({"role": "user", "content": user_content})

        return messages

    def _build_user_content(
        self, text: str, media: list[str] | None
    ) -> str | list[dict[str, Any]]:
        """Build user message content with optional images (local paths or HTTP URLs)."""
        if not media:
            return text

        images = []
        for item in media:
            item = str(item)
            if item.startswith(("http://", "https://")):
                images.append({"type": "image_url", "image_url": {"url": item}})
            else:
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

    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> list[dict[str, Any]]:
        """
        Add a tool result to the message list.

        Args:
            messages: Current message list.
            tool_call_id: ID of the tool call.
            tool_name: Name of the tool.
            result: Tool execution result.

        Returns:
            Updated message list.
        """
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
        """
        Add an assistant message to the message list.

        Args:
            messages: Current message list.
            content: Message content.
            tool_calls: Optional tool calls.
            reasoning_content: Thinking output (Kimi, DeepSeek-R1, etc.).

        Returns:
            Updated message list.
        """
        msg: dict[str, Any] = {"role": "assistant"}

        # Omit empty content — some backends reject empty text blocks
        if content:
            msg["content"] = content

        if tool_calls:
            msg["tool_calls"] = tool_calls

        # Include reasoning content when provided (required by some thinking models)
        if reasoning_content:
            msg["reasoning_content"] = reasoning_content

        messages.append(msg)
        return messages
