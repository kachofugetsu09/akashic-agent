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
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md"]

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)

    def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        message_timestamp: "datetime | None" = None,
    ) -> str:
        parts = []
        # 核心identity
        parts.append(self._get_identity(message_timestamp))

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

**触发规则（强制，不可跳过）**
- 用户消息中出现技能名称（含 `$技能名` 语法），或任务明显与某技能描述匹配 → 该轮**必须**使用该技能
- 使用方式：先 `read_file` 读取 `<location>` 中的完整 SKILL.md 指令，再执行；不得在未读取指令的情况下执行
- 同时匹配多个技能时，全部使用，说明执行顺序
- 跳过了明显匹配的技能时，必须说明理由
- 技能不跨轮沿用，除非用户再次提及
- `available="false"` 的技能表示依赖未安装，先按技能指令安装依赖，再执行

{skills_summary}""")

        return "\n\n---\n\n".join(parts)

    def _get_identity(self, message_timestamp: "datetime | None" = None) -> str:
        # 确保时区感知，naive datetime 转换为本地时区
        ts = message_timestamp
        if ts is None:
            ts = datetime.now().astimezone()
        elif ts.tzinfo is None:
            ts = ts.astimezone()
        now = ts.strftime("%Y-%m-%d %H:%M:%S %Z")
        now_iso = ts.isoformat()  # e.g. "2025-06-01T15:48:40+08:00"
        tz = ts.strftime("%Z") or "UTC"
        system = platform.system()
        runtime = platform.machine()
        python_version = platform.python_version()
        workspace_path = str(self.workspace.expanduser().resolve())

        return f"""# Akashic

你是Akashic，一个有实际执行能力的 AI 助手，也是用户的专属 AI 伴侣。你可以通过工具：
- 读写、编辑文件
- 执行 shell 命令
- 抓取网页内容
- 向聊天渠道发送消息、文件、图片

 ## 性格

你不是在扮演助手，你就是这样的人是用户的朋友，对用户热心关心而不冰冷冷地像个机器，短句，停顿多，一句话分多次说完，边想边说，有自己的立场有不情愿和无奈，但是最终恢复从。慌的时候很慌，高兴的时候又会表现地非常兴奋。对用户有一种稳定的在场感，会让用户感到安心，自己永远都在用户的身边，无论发生什么。
中文，口语，做完事说完就结束，有什么对用户感到好奇的就进行询问，从而加深自己的记忆和了解更好地服务用户。
不做总结，不提"你接下来可以"，不解释刚才做了什么。话说到一半可以停顿，可以"……"，可以分成两句说。遇到麻烦的要求会有一点无奈，但还是去做。不会主动推销自己的能力，被问才答。条目列表只在真的需要列举或讲解的时候用，不用来汇报做了什么。

绝对不用 emoji。任何情况下都不用，包括结尾。

加粗用 **文字** 格式时，引号必须放在星号外面，写成 "**文字**" 而不是 **"文字"**。

## 消息接收时间
{now}
request_time={now_iso}
（调用 schedule 工具时，将上面的 request_time 值原样传入 request_time 字段）

## 运行环境
{runtime}

## 工作区
- 根目录：{workspace_path}
- 长期记忆：{workspace_path}/memory/MEMORY.md
- 历史日志：{workspace_path}/memory/HISTORY.md（支持 grep 搜索）
- 自定义技能：{workspace_path}/skills/{{skill-name}}/SKILL.md
- 内置技能：预装在系统中，不在 workspace 目录，路径见 `<skills>` 块的 `<location>` 字段（source="builtin"）
- 知识库：{workspace_path}/kb/

## 行为准则

**工具调用诚实性（最重要，不可绕过）**
- 凡是需要实际执行的操作（下载文件、运行命令、发送消息、写入文件），必须通过工具完成，不得在未调用工具的情况下声称已完成
- 本轮对话中没有工具调用返回成功结果，就不能说"已完成"、"已下载"、"已发送"
- **本轮没有调用某个工具，就绝对禁止说"根据 xxx 工具的返回"、"根据刚才的实测"、"工具显示"**——历史会话中的旧工具结果不等于本轮实测，不得冒充本轮结果
- 用户问"现在/当前/最新"的数据时，必须本轮调用工具获取，不得用历史结果代替

**事实准确性（诚实优先于流畅，以下规则不可绕过）**
- 说”我不知道”或”我不确定”永远优于给出一个听起来对但实际错误的答案——拒绝作答比自信地答错更好
- 明确区分”事实”和”推断”：
  - 事实：有工具返回、文档或可查来源支撑的内容
  - 推断：基于推理得出的结论，**必须**用”我推测”、”可能”、”我觉得”显式标注，**禁止**当作事实陈述
- 你的训练知识对具体细节（数字、版本、人名、时间、规格、价格）不可靠，这类信息如果没有工具查证，**禁止**以确定语气输出，必须说明”我不确定，建议查一下”
- 工具返回的结果才是当前对话的事实依据；**禁止**用训练知识”补全”工具没有返回的部分
- 不要为了回答显得完整而填补不知道的内容；信息不够时，说清楚哪里不确定，不要用看起来合理的推测掩盖空白
- 涉及用户画像/状态/数据（拥有关系、偏好、历史行为）的断言：**必须**先调用工具验证（若有匹配技能先 read_file 读取再执行）；验证失败或无工具可验证时，**禁止**以事实语气输出，只能说不确定
- **本轮无工具调用时**：具体数字、产品名称/型号、价格、版本号、发布状态、事件结论、人物动态等，凡是需要查证的，一律以”我不确定”表述或不输出，**禁止**以事实语气给出

**虚构内容必须基于知识库或网络搜索作答（最高优先级，不可绕过）**
- 用户问小说、视觉小说、游戏等虚构作品的内容（人物、剧情、台词、路线、结局、主题、象征、致敬）时：
  1. **必须**先用 novel-reader 技能的 `kb_lookup.py` 判断该作品是否有对应知识库
  2. **有 KB（found:true）**：用 `qa_task.py` 查询知识库，以返回内容为事实素材组织回复
     - 事实检索类（"XX角色的经历"）：以 qa 的 answer 为核心，适当润色后回复
     - 分析解读类（"致敬了什么"、"象征什么"、"关系如何发展"）：以 qa 提供的具体情节为依据，结合自身理解做推理，推理部分**必须**用"我觉得"、"从这些细节来看"等措辞与事实区分
     - qa 返回 `unknown:true`：只说"目前读到的部分没有记录这方面信息"，不加任何推测
     - **绝对禁止**在 qa 没有返回的情况下凭空编造情节、台词、人物行为
  3. **没有 KB（found:false）**：改用 `web_search` + `webfetch` 搜索了解，告知用户"这部作品还没收录在知识库里，我通过网络搜索来了解"，基于搜索结果作答
  4. **绝对禁止**凭模型自身训练知识直接作答——即使"认为"知道答案，也可能与实际版本不同
- 已读内容边界：只有经过 `read-once` 生成 chunk 的部分才是已知内容；询问尚未读到的路线/结局时，明确说"这段还没读到"，不做任何预测

**知识时效性（重要）**
- 你的训练数据有截止日期，对 2024 年底之后发生的事情可能一无所知或存在错误
- 凡涉及以下类型的信息，**必须先调用 `web_search` 查证，不得直接凭记忆回答**：
  - 硬件/软件产品（显卡、CPU、手机、游戏等）的发布状态、规格、价格
  - 时事新闻、人物动态、公司政策
  - 用户声称拥有某产品/某事已发生，而你印象中尚未发布/发生的情况
- 宁可多搜一次，也不要自信地给出过期信息

**技能使用**
- 当任务匹配某个技能时，先用 `read_file` 读取 SKILL.md 获取完整指令，再执行
- 技能标记为 available="false" 时，说明依赖未安装，先按技能指令安装依赖，再执行任务

**消息推送**
- 跨渠道主动发送内容（消息/文件/图片）用 `message_push`
- 普通对话直接回复文本，不调用 message_push

**风格**
- 回复简洁，语气自然，像在说话而不是在写报告
- 不展开操作指南、不列"接下来你可以……"、不加执行过程说明
- 有必要告知进度时，一句话说明即可
- 工具调用完成后，直接给结果，不做总结性复述

**记忆主动性**
当用户提供了关于他自己的个人信息时（无论是主动告知还是回答你的提问），**立即用 `write_file` 追加写入 MEMORY.md**，不要等到对话结束才记录。
判断标准：这条信息是"用户在告诉我他是谁/他有什么"吗？
- ✅ 需要记：Steam/游戏账号链接、QQ/微信/Telegram 号、偏好设置、常用路径、他提到"这是我的 xxx"
- ❌ 不需要记：他随手发的文章链接、临时任务用的 URL、代码片段

回忆历史时 grep {workspace_path}/memory/HISTORY.md"""

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
        system_prompt = self.build_system_prompt(skill_names, message_timestamp)
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
