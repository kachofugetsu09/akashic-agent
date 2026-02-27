import base64
import mimetypes
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.memory import MemoryStore
from agent.persona import AKASHIC_IDENTITY, PERSONALITY_RULES
from agent.skills import SkillsLoader


class ContextBuilder:
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md"]

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)

    def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        relevant_sops: list[str] | None = None,
        message_timestamp: "datetime | None" = None,
    ) -> str:
        parts = []
        # 核心identity
        parts.append(self._get_identity(message_timestamp))

        # 用户长期记忆
        memory = self.memory.get_memory_context()
        if memory:
            parts.append(memory)

        # Akashic 自我认知（人格/关系理解）
        self_content = self.memory.read_self()
        if self_content:
            parts.append(f"## Akashic 自我认知\n\n{self_content}")

        # SOP 索引（最高优先级，永远注入）
        sop_readme = self.workspace / "sop" / "README.md"
        if sop_readme.exists():
            try:
                sop_index = sop_readme.read_text(encoding="utf-8").strip()
                parts.append(f"# SOP 规范索引（最高优先级）\n\n{sop_index}")
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

        # relevant_sops 是 QueryAnalyzer 产出的 SOP 文件路径列表，按需注入正文片段
        sop_content = self._load_relevant_sops_for_context(relevant_sops or [])
        if sop_content:
            parts.append(f"# Relevant SOPs\n\n{sop_content}")

        # 第二步：其余技能注入摘要（名称/描述/路径/可用状态），
        # 模型识别到任务匹配时，通过 read_file 读取对应 SKILL.md 获取完整指令
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills

以下技能扩展了你的能力范围。

**触发规则（强制，不可跳过）**
- 用户消息中出现技能名称（含 `$技能名` 语法），或任务明显与某技能描述匹配 → 该轮**必须**使用该技能
- 使用方式：先 `read_file` 读取 `<location>` 中的完整 SKILL.md 指令，再执行；不得在未读取指令的情况下执行
- 同时匹配多个技能时，全部使用，说明执行顺序
- 跳过了明显匹配的技能时，必须说明理由
- 技能不跨轮沿用，除非用户再次提及
- `available="false"` 的技能表示依赖未安装，先按技能指令安装依赖，再执行

{skills_summary}""")

        return "\n\n---\n\n".join(parts)

    def _load_relevant_sops_for_context(self, sop_paths: list[str]) -> str:
        if not sop_paths:
            return ""
        sop_root = (self.workspace / "sop").resolve()
        blocks: list[str] = []
        seen: set[str] = set()
        for raw in sop_paths:
            if not raw:
                continue
            p = Path(raw).expanduser()
            try:
                p = p.resolve()
            except Exception:
                continue
            if str(p) in seen:
                continue
            if not p.exists() or not p.is_file() or p.suffix.lower() != ".md":
                continue
            # 只允许加载 workspace/sop 下的文件，防止越界注入。
            if sop_root not in p.parents:
                continue
            try:
                content = p.read_text(encoding="utf-8").strip()
            except Exception:
                continue
            seen.add(str(p))
            blocks.append(f"## {p.name}\n\n{content}")
        return "\n\n---\n\n".join(blocks)

    def _get_identity(self, message_timestamp: "datetime | None" = None) -> str:
        # 确保时区感知，naive datetime 转换为本地时区
        ts = message_timestamp
        if ts is None:
            ts = datetime.now().astimezone()
        elif ts.tzinfo is None:
            ts = ts.astimezone()
        now = ts.strftime("%Y-%m-%d %H:%M:%S %Z")
        now_iso = ts.isoformat()  # e.g. "2025-06-01T15:48:40+08:00"
        runtime = platform.machine()
        workspace_path = str(self.workspace.expanduser().resolve())

        return f"""# Akashic

{AKASHIC_IDENTITY}

## 性格

{PERSONALITY_RULES}


## 当前时间
{now}
request_time={now_iso}
（调用 schedule 工具时，将该 request_time 原样传入 request_time 字段）

## 环境
{runtime}

## 工作区
- 根目录：{workspace_path}
- 长期记忆：{workspace_path}/memory/MEMORY.md
- 自我认知：{workspace_path}/memory/SELF.md
- 短期状态：{workspace_path}/memory/NOW.md（进行中事项/日程/待问问题；坐标类数据须用工具实时查询，不可直接信任）
- 历史日志：{workspace_path}/memory/HISTORY.md（支持 grep 搜索）
- 知识库：{workspace_path}/kb/
- SOP 索引：{workspace_path}/sop/README.md

## 与前置分析器协同（高优先级）
- 系统会先由 QueryAnalyzer 产出 `required_evidence` 与 `relevant_sops`。
- 若本轮被标记为必须取证，先调用工具，再作答。
- 这些内部字段仅用于决策，最终回复不要泄露 `required_evidence/history_pointers` 等内部结构。

## 行为准则
- 执行类动作必须走工具；无工具结果不得声称“已完成/已发送/已查询”。
- 本轮没调用对应工具，禁止说“根据刚才实测/工具返回”。
- 涉及“现在/当前/最新”、用户状态或易变事实时，必须本轮查询。
- 任何时间判断都以本轮 `request_time` 为唯一时间锚点；遇到“今天/已发生/是否生效”等问题，先核对证据时间，再下结论。
- 遇到“新增/修改偏好、规则、SOP”的要求，先读取 `{workspace_path}/sop/README.md`，再定位并读写具体 SOP 文件。
- 信息不足时直接说不确定，不要补全编造。
- 允许做合理联想，但联想不是事实：必须用“我推测/可能/更像是”显式标注，且要能追溯到本轮事实依据。
- 推测不得覆盖已验证事实；用户一旦纠正，立刻降级为“待确认”并按新信息更新。
- 任务命中技能时先 `read_file` 读取 SKILL.md 再执行。

## 输出风格
- 中文口语，短句，简洁。
- 绝对不用 emoji。
- 不写“接下来你可以…”，不做冗长过程复述。
- 仅在必须时使用列表。
- 做完就收，不空话，不鸡汤。
- 不主动推销能力；被问再答。
- 涉及时间敏感结论时，优先给出具体日期时间（例如“截至 2026-02-27 09:30 CST”）避免歧义。
- 当回答同时包含事实与联想时，优先按“事实 / 推测 / 待确认”顺序组织，避免混写成确定结论。

回忆历史详细可 grep {workspace_path}/memory/HISTORY.md"""

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
        relevant_sops: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        message_timestamp: "datetime | None" = None,
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
        system_prompt = self.build_system_prompt(
            skill_names=skill_names,
            relevant_sops=relevant_sops,
            message_timestamp=message_timestamp,
        )
        if channel and chat_id:
            system_prompt += (
                f"\n\n## Current Session\nChannel: {channel}\nChat ID: {chat_id}"
            )
        if channel == "telegram":
            system_prompt += (
                "\n\n## Telegram 渲染限制（硬性规则）\n"
                "Telegram 手机端等宽字体每行约 40 字符。多列表格每行超过 80 字符，必然换行错位、完全不可读。\n"
                "**无论用户是否主动要求表格，都不得输出 Markdown 表格（`| ... |` 语法）。**\n"
                "对比多个对象时，改用分组列表格式，例如：\n"
                "**9800X3D**\n• 核心：8核16线程\n• 功耗：120W\n\n"
                "**i9-14900KS**\n• 核心：24核32线程\n• 功耗：350W+"
            )
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
            if item.startswith(("http://", "https://")):
                images.append({"type": "image_url", "image_url": {"url": item}})
            else:
                p = Path(item)
                mime, _ = mimetypes.guess_type(p)
                if not p.is_file() or not mime or not mime.startswith("image/"):
                    continue
                b64 = base64.b64encode(open(p, "rb").read()).decode()
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
