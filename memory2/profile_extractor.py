from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ProfileFact:
    summary: str
    category: str
    happened_at: str | None


class ProfileFactExtractor:
    def __init__(
        self,
        llm_client: Any,
        *,
        model: str = "",
        max_tokens: int = 400,
        timeout_ms: int = 5000,
    ) -> None:
        self._llm_client = llm_client
        self._model = model
        self._max_tokens = max(128, int(max_tokens))
        self._timeout_s = max(0.1, float(timeout_ms) / 1000.0)

    async def extract(
        self,
        conversation: str,
        *,
        existing_profile: str = "",
    ) -> list[ProfileFact]:
        # 1. 先构造 prompt；空对话直接返回空列表。
        if not str(conversation or "").strip():
            return []
        prompt = self._build_prompt(
            conversation=conversation,
            existing_profile=existing_profile,
        )

        facts = await self._extract_with_prompt(
            prompt,
            max_tokens=self._max_tokens,
            timeout_s=self._timeout_s,
        )
        if facts:
            return facts

        # 高密度混合内容下，整段抽取可能过于保守；按 USER 子句做一次兜底。
        clause_facts: list[ProfileFact] = []
        for clause in self._split_user_clauses(conversation):
            clause_items = await self.extract_from_exchange(
                clause,
                "",
                existing_profile=existing_profile,
            )
            clause_facts.extend(clause_items)
        return self._dedupe_facts(clause_facts)

    async def extract_from_exchange(
        self,
        user_msg: str,
        agent_response: str,
        *,
        existing_profile: str = "",
    ) -> list[ProfileFact]:
        """只从单轮 user/assistant 交换中提取 purchase/status/personal_fact。"""
        if not (str(user_msg or "").strip() or str(agent_response or "").strip()):
            return []

        prompt = self._build_exchange_prompt(
            user_msg=user_msg,
            agent_response=agent_response,
            existing_profile=existing_profile,
        )
        try:
            response = await asyncio.wait_for(
                self._llm_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    tools=[],
                    model=self._model,
                    max_tokens=min(self._max_tokens, 200),
                ),
                timeout=min(self._timeout_s, 1.5),
            )
        except Exception:
            return []

        content = str(getattr(response, "content", response) or "")
        facts = self._parse_facts(content)
        allowed = {"purchase", "status", "personal_fact"}
        return [fact for fact in facts if fact.category in allowed]

    async def _extract_with_prompt(
        self,
        prompt: str,
        *,
        max_tokens: int,
        timeout_s: float,
    ) -> list[ProfileFact]:
        try:
            response = await asyncio.wait_for(
                self._llm_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    tools=[],
                    model=self._model,
                    max_tokens=max_tokens,
                ),
                timeout=timeout_s,
            )
        except Exception:
            return []
        content = str(getattr(response, "content", response) or "")
        return self._parse_facts(content)

    @staticmethod
    def _split_user_clauses(conversation: str) -> list[str]:
        clauses: list[str] = []
        seen: set[str] = set()
        for line in str(conversation or "").splitlines():
            match = re.search(r"\bUSER:\s*(.+)$", line.strip(), flags=re.IGNORECASE)
            if not match:
                continue
            text = match.group(1).strip()
            parts = re.split(r"[。！？；;.!?，,]\s*", text)
            for part in parts:
                clause = part.strip()
                if len(clause) < 4:
                    continue
                if clause in seen:
                    continue
                seen.add(clause)
                clauses.append(clause)
        return clauses

    @staticmethod
    def _dedupe_facts(facts: list[ProfileFact]) -> list[ProfileFact]:
        deduped: list[ProfileFact] = []
        seen: set[tuple[str, str]] = set()
        for fact in facts:
            key = (str(fact.summary).strip(), str(fact.category).strip())
            if not key[0] or key in seen:
                continue
            seen.add(key)
            deduped.append(fact)
        return deduped

    @staticmethod
    def _build_prompt(*, conversation: str, existing_profile: str) -> str:
        return f"""你是 profile 事实提取器。请只从对话里提取用户长期可检索的 profile 事实，并输出 XML。

profile 的语义是：关于用户本人或其客观处境的事实。
例如：身份背景、拥有/持有的东西、爱好、长期健康事实、长期状态、重要个人决定。
不是“用户希望怎样被服务/怎样被讲解/怎样被推荐”。

仅允许以下 4 类：
- purchase：用户购买 / 下单了什么
- decision：用户明确拍板了什么方案 / 计划，或重要宣布（项目公开/上线/重要变更决定）
- status：用户某件事的状态变化（等待 / 完成 / 放弃 / 里程碑达成）
  示例：游戏通关、项目公开、任务完成
- personal_fact：用户关于自身的事实性披露，包括：
  · 身份/背景：职业、居住地、家庭成员、健康状况、技能
  · 持有/拥有数量："我有 N 个 X"（如拥有20个播放列表、养了3只猫、收藏了50张黑胶唱片）
  · 人际关系：谁给了我什么、谁住在哪里、家人/朋友的具体信息
  · 兴趣爱好：例如“爱好是弹钢琴”“喜欢爬山”
  · 习惯/经验背景：例如“用户不常徒步”“用户很少做饭”“用户第一次养猫”
  · 与某活动直接相关的长期准备/资源状态：例如用户明确表示自己没有相关装备、缺少准备经验

必须遵守：
- 纯技术讨论、闲聊、打招呼，不输出
- 只有当用户在对话中直接陈述自己的事实时，才允许提取
- 用户提问、追问、反问、记忆测试句都不算事实披露，绝对禁止反推成 profile
  示例：
  · "你还记得我什么时候开始戴 fitbit 手环的吗"
  · "我之前是不是买过这个"
  · "你记得我住哪里吗"
  以上都应返回空，不得根据既有上下文或模型猜测补出答案
- 用户在“举例 / 假设 / 如果 / 比如 / 设想 / 虚构场景 / 这个例子是这样的”这类语境里使用第一人称，
  也不算事实披露，绝对禁止提取成 profile
  示例：
  · "我给你说一个例子：如果我有一家咖啡店，最近在亏损"
  · "比如说我明天要去签一个大合同"
  · "假设我家里有三只猫"
  以上都应返回空；这是论证、举例或假设，不是用户真实背景
- 若 existing_profile 已有相同事实，不重复输出
- summary 要简洁、可独立检索
- 每条 summary 只表达一条完整事实，避免把多个事实揉成一条
- summary 不要带时间戳；时间放到 happened_at
- personal_fact 默认不写 happened_at。只有 purchase / status / decision 这类确实带时间语义的事实，才填写 happened_at
- 每一件具体的事单独一条，绝对不要合并
  ✗ 错误："用户购买了多件商品"
  ✓ 正确：每件商品单独一条，写出具体名称/型号
- 涉及列举时（多件购买、多个决定）每项单独输出
- summary 写出具体内容而非概括：写"用户购买了罗技 MX Master 3 鼠标"而非"用户购买了外设"
- 若多个候选只是同一事实的近似改写，只保留一条最直接、最贴近 USER 原话的版本
- 若存在冲突，保留更新、更确定的那条

【证据源规则】
- ASSISTANT 的回复只作为背景参考，不能作为提取证据
- 即使 ASSISTANT 说"你之前买了 X""你是 XX 方向的学生"，也不得作为事实来源
- 只有 USER 原话中明确陈述的事实才允许提取

【额外禁止类型】
- 工程操作过程：安装依赖、配置环境、调试步骤、更新工具版本
  → 这些是工程 event，不是用户身份/状态的 profile
- 项目内讨论：架构决策、重构方案、代码评审意见
  → 不算用户自身 profile；decision 仅指用户个人/产品层面的重要决定，不含技术实现讨论
- 用户表达的观点 / 意见
  → 必须是关于用户自身的客观事实，而非用户对某事物的看法
- 用户希望助手怎样服务他、怎样讲解、怎样推荐
  → 这是 preference，不是 profile，绝对不要输出到本阶段
- 纯 event 事实
  → 例如“这周日要去徒步”“明天要开会”“昨晚去了超市”；这些是 event，不是 profile
- 不要把一次性计划直接写成 profile，但如果用户同时透露了稳定经验背景或资源状态，可以只提取那部分 profile
  例如：用户说“这周日朋友约我去徒步，我其实不常徒步，不知道该买什么装备”
  可以提取：
  · 用户不常徒步
  · 用户目前缺少徒步相关装备准备
  但不要提取“这周日要去徒步”为 profile

区分 personal_fact 和 preference：
- “我有一块 Fitbit 手表”“我家有 10 套房”“我的爱好是弹钢琴” → personal_fact
- “讲内容时最好附带一个很棒的例子并贯穿始终” → preference，不能在本阶段输出

正例（应该提取）：
- USER: 我有一块 Fitbit 手表，我的爱好是弹钢琴。
  输出：
  · 用户有一块 Fitbit 手表 | personal_fact
  · 用户的爱好是弹钢琴 | personal_fact
- USER: 我在互联网公司做产品经理，今年30岁。下班后我喜欢自己研究做饭。另外我下周末要去旅行，但我还没开始收拾行李。
  输出：
  · 用户在互联网公司做产品经理 | personal_fact
  · 用户今年30岁 | personal_fact
  · 用户喜欢下班后自己研究做饭 | personal_fact
  不输出：
  · 下周末要去旅行
  · 还没开始收拾行李
- USER: 我其实不常徒步，不知道该买什么装备。
  输出：
  · 用户不常徒步 | personal_fact
  · 用户目前缺少徒步相关装备准备 | personal_fact

反例（不该提取）：
- USER: 你给我讲内容的时候最好附带一个很棒的例子，并且最好贯穿始终。
  输出：空（这是 preference，不是 profile）
- USER: 这周日朋友约我去徒步。
  输出：空（这是 event，不是 profile）
- USER: 我给你说一个例子，如果我有一家咖啡店，最近在亏损。
  输出：空（这是假设性举例，不是用户真实事实）

当一段话里同时包含稳定 profile 事实和临时 event 时：
- 提取稳定事实
- 丢弃临时计划、短期状态、一次性安排
- 不要因为同一句里混有 event，就把整段都判空

当前已有 profile（用于查重）：
{existing_profile or "（空）"}

待处理对话：
{conversation}

只输出 XML：
<facts>
<fact>
  <summary>...</summary>
  <category>purchase|decision|status|personal_fact</category>
  <happened_at>YYYY-MM-DD</happened_at>
</fact>
</facts>"""

    @staticmethod
    def _build_exchange_prompt(
        *,
        user_msg: str,
        agent_response: str,
        existing_profile: str,
    ) -> str:
        return f"""你是单轮 profile 事实提取器。只看这一轮对话（1 条 USER + 1 条 ASSISTANT），不要推断、不要联想。

只允许提取以下 3 类：
- purchase：用户刚购买/下单了什么
- status：用户某件事的状态变化（等待、到货、完成、放弃），或里程碑达成（游戏通关、项目上线/公开、任务完成、竞赛结果）
- personal_fact：用户关于自身的事实性披露，包括身份/背景、持有数量（"我有 N 个 X"）、人际关系（谁给了我什么、家人住在哪里）

禁止输出：
- decision
- preference
- 纯闲聊、打招呼
- 纯技术讨论
- 用户提问、追问、记忆测试句
  例如：
  · "你还记得我什么时候开始戴 fitbit 手环的吗"
  · "你记得我之前为什么发过那张图吗"
  这类句子不是事实披露，必须返回空
- 用户在举例、假设、类比、虚构场景里用第一人称说的话
  例如：
  · "我给你说一个例子，如果我有一家咖啡店，最近在亏损"
  · "比如说我今天要去签一个大合同"
  这类句子是举例，不是用户真实事实，必须返回空
- 任何不是用户本人事实的内容
- ASSISTANT 确认或复述的内容，即使涉及用户，也不算用户陈述，不得提取
- 工程操作（安装、更新、配置工具/依赖）不属于 status
  → status 仅指里程碑型状态变化（游戏通关、项目上线、任务完成、竞赛结果）

若 existing_profile 已有同一事实，不重复输出。

提取粒度要求：
- 每一件具体的事单独一条，不要合并
- 写出具体名称/型号/数量，不要用概括性词语
  ✗ 错误："用户购买了游戏外设"
  ✓ 正确："用户购买了罗技 G Pro X 耳机"

当前已有 profile（用于查重）：
{existing_profile or "（空）"}

本轮对话：
USER: {user_msg}
ASSISTANT: {agent_response}

只输出 XML：
<facts>
<fact>
  <summary>...</summary>
  <category>purchase|status|personal_fact</category>
  <happened_at>YYYY-MM-DD</happened_at>
</fact>
</facts>"""

    def _parse_facts(self, raw_output: str) -> list[ProfileFact]:
        allowed = {"purchase", "decision", "status", "personal_fact"}
        matches = re.findall(r"<fact>\s*(.*?)\s*</fact>", raw_output or "", re.DOTALL)
        facts: list[ProfileFact] = []
        seen: set[tuple[str, str]] = set()
        for block in matches:
            summary = self._extract_tag(block, "summary")
            category = self._extract_tag(block, "category").lower()
            happened_at = self._extract_tag(block, "happened_at") or None
            if not summary or category not in allowed:
                continue
            if category == "personal_fact":
                happened_at = None
            key = (summary, category)
            if key in seen:
                continue
            seen.add(key)
            facts.append(
                ProfileFact(
                    summary=summary,
                    category=category,
                    happened_at=happened_at,
                )
            )
        return facts

    @staticmethod
    def _extract_tag(raw_output: str, tag: str) -> str:
        match = re.search(
            rf"<{tag}>\s*(.*?)\s*</{tag}>",
            raw_output or "",
            flags=re.IGNORECASE | re.DOTALL,
        )
        return match.group(1).strip() if match else ""
