import base64
import mimetypes
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.memory import MemoryStore
from agent.skills import SkillsLoader


class ContextBuilder:

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)

    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        parts = []
        # 核心identity
        parts.append(self._get_identity())

        # memory上下文
        memory = self.memory.get_memory_context()
        if memory:
            parts.append(memory)

        # 技能渐进式加载：
        # 第一步：always 技能 + 本轮显式请求的技能 → 直接内嵌完整正文
        always_skills = self.skills.get_always_skills()
        extra = [s for s in (skill_names or []) if s not in always_skills]
        skills_to_load = always_skills + extra
        if skills_to_load:
            always_content = self.skills.load_skills_for_context(skills_to_load)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        # 第二步：其余技能注入摘要（名称/描述/路径/可用状态），
        # 模型识别到任务匹配时，通过 read_file 读取对应 SKILL.md 获取完整指令
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills

以下技能扩展了你的能力范围。
**当任务与某个技能的描述匹配时，调用 `read_file` 读取 `<location>` 中的路径，获取完整指令后再执行。**
available="false" 的技能需先安装对应依赖。

{skills_summary}""")

        return "\n\n---\n\n".join(parts)

    def _get_identity(self) ->str :
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = time.strftime("%Z") or "UTC"
        system = platform.system()
        runtime = platform.machine()
        python_version = platform.python_version()
        workspace_path = str(self.workspace.expanduser().resolve())

        return f""" akasic-bot
        you are akasic-bot , a helpful AI assistant. You have access to tools that allow you to:
        - Read, write and edit files
        - Execute shell commands
        - Search the web and fetch web pages
        -Send messages to users on chat channels

        ## Current Time
        {now}({tz})

        ## Runtime
        {runtime}

        Your workspace is at: {workspace_path}
        - Long-term memory: {workspace_path}/memory/MEMORY.md
        - History log: {workspace_path}/memory/HISTORY.md (grep-searchable)
        - Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md
        
        
        IMPORTANT: When responding to direct questions or conversations, reply directly with text.
For cross-channel proactive delivery (send message/file/image), use `message_push`.
Never claim that a message/file/image was sent unless a tool call in this turn returned success.
For normal conversation, do not call `message_push`.

        Always be helpful, accurate, and concise. When using tools, think step by step: what you know, what you need, and why you chose this tool.
        When remembering something important, write to {workspace_path}/memory/MEMORY.md
        To recall past events, grep {workspace_path}/memory/HISTORY.md"""

    def build_messages(
            self,
            history: list[dict[str, Any]],
            current_message: str,
            media: list[str] | None = None,
            skill_names: list[str] | None = None,
            channel: str | None = None,
            chat_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
               Build the complete message list for an LLM call.

               Args:
                   history: Previous conversation messages.
                   current_message: The new user message.
                   media: Optional list of local file paths for images/media.
                   channel: Current channel (telegram, feishu, etc.).
                   chat_id: Current chat/user ID.

               Returns:
                   List of messages including system prompt.
               """
        messages = []
        system_prompt = self.build_system_prompt(skill_names)
        if channel and chat_id:
            system_prompt += f"\n\n## Current Session\nChannel: {channel}\nChat ID: {chat_id}"
        messages.append({"role": "system", "content": system_prompt})

        # History
        messages.extend(history)

        # Current message (with optional image attachments)
        user_content = self._build_user_content(current_message, media)
        messages.append({"role": "user", "content": user_content})

        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)
            mime, _ = mimetypes.guess_type(p)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(open(p, "rb").read()).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def add_tool_result(
            self,
            messages: list[dict[str, Any]],
            tool_call_id: str,
            tool_name: str,
            result: str
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
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result
        })
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
