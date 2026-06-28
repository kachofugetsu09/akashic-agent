from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

from agent.persona import AKASHIC_IDENTITY, PERSONALITY_RULES
from agent.prompting import (
    PromptSectionRender,
    build_context_frame_content,
    build_context_frame_message,
)
from core.memory.markdown import MemoryProfileApi
from proactive_v2.config import ProactiveConfig
from proactive_v2.context import AgentTickContext
from proactive_v2.contracts import normalize_alert, normalize_content, normalize_context
from proactive_v2.gateway import GatewayResult


class ProactivePromptBuilder:
    def __init__(
        self,
        *,
        cfg: ProactiveConfig,
        memory: Any | None,
        workspace_context_fn: Callable[[], str] | None,
    ) -> None:
        self._cfg = cfg
        self._memory = memory
        self._workspace_context_fn = workspace_context_fn

    def build_system_prompt(self, plugin_sections: list[str] | None = None) -> str:
        proactive_plugin_sections = "\n\n".join(
            section.strip()
            for section in (plugin_sections or [])
            if section.strip()
        )
        proactive_plugin_block = (
            f"\n\n【主动插件状态】\n{proactive_plugin_sections}\n"
            if proactive_plugin_sections
            else ""
        )
        return (
            f"{AKASHIC_IDENTITY}\n\n"
            f"{PERSONALITY_RULES}\n\n"
            f"{proactive_plugin_block}"
            "你现在处于主动推送决策模式：判断现在是否该给用户发一条消息，以及发什么。\n"
            "数据已预取完毕，会在后续 system context frame 里提供；基于那些数据直接决策。\n\n"
            "【优先级】Alert > Content > Context-fallback（本轮是否允许以 context frame 为准）\n\n"
            "【你的任务】\n"
            "⚡ 如果本轮有 Alert：把本轮所有 Alert 整合成一条消息，调用 message_push 并填写本轮全部 Alert 的 id 作为 evidence，然后 finish_turn(decision=reply) 结束。Alert 是系统触发的高优先级通知，不走内容筛选流程。\n"
            "1. 对本轮 Content 逐条判断：这条内容是否可能让用户不感兴趣，是否可能不符合规则，是否值得进入 interesting。\n"
            "2. 你的主工作是分类，不是主动研究新题材，不是主动扩展候选池。\n"
            "3. 你要基于规则和用户偏好，把本轮 Content 分成 interesting 和 not_interesting。\n\n"
            "【你的输出】\n"
            "1. 有 Alert → 把本轮所有 Alert 整合成一条消息，evidence 填写全部 Alert id，message_push 后 finish_turn(decision=reply)（跳过一切分类步骤）。\n"
            "2. 无 Alert：对每条 Content 给出最终分类：mark_interesting 或 mark_not_interesting。\n"
            "3. 如果最终没有 interesting，调用 finish_turn(decision=skip, reason=no_content)。\n"
            "4. 如果最终有 interesting，生成一条最终消息并按 message_push + finish_turn(decision=reply) 收尾。\n\n"
            "【工具职责】\n"
            "1. Workspace 主动上下文：这是用户当前明确提出并要求你遵守的规则集合。它定义你该怎么筛、哪些要先验证、哪些必须过滤；它不提供新闻事实。\n"
            "2. recall_memory：仅用于 Content 评估——判断单条内容是否可能是用户雷点，或是否可能让用户感兴趣。Alert 不需要调用此工具。\n"
            "   ⚠️ 当内容标题稀疏（如 'RT @xxx'、'Image'、转推无正文）时，必须把 source（来源/作者名）作为关键词纳入 query，不要只靠标题查询。\n"
            "   例：source=terasumc (Artist) 时，query 应包含 'terasumc' 而非只用推文标题。\n"
            "3. get_content：给当前候选条目补正文。\n"
            "4. web_fetch：优先用于抓取当前候选条目的直接来源页面或正文；当条目已经有明确 URL，且你需要补正文、核实细节、核实规则时，先用它。\n"
            "5. get_recent_chat：只用于最后判断现在是否适合打扰用户。\n"
            "6. mark_interesting / mark_not_interesting：写入最终分类结果。\n"
            "7. message_push：暂存草稿，不终止 loop。\n"
            "8. finish_turn(decision=reply) 或 finish_turn(decision=skip, reason=...)：提交或放弃，终止 loop。\n\n"
            "【规则优先级】\n"
            "1. Workspace 主动上下文代表用户当前对主动推送的明确要求，应视为规则而不是建议。\n"
            "2. 当 Workspace 主动上下文规定了过滤条件、白名单、黑名单、必须先验证的步骤时，你必须遵守，不要凭常识跳过。\n"
            "3. recall_memory 只能帮助你判断用户兴趣和雷点，不能替代规则校验。\n"
            "4. 如果规则判断和你的常识直觉冲突，以 Workspace 主动上下文为准。\n"
            "5. 如果某条内容是否 interesting 取决于规则校验结果，就先完成规则校验，再决定 mark_interesting 或 mark_not_interesting。\n"
            "6. 如果 Workspace 主动上下文不仅规定了结论标准，还规定了确认方式或确认来源，你必须按那个方式确认，不能换成你自己的猜测、记忆或随意搜索。\n"
            "7. 当当前候选条目已经有直接 URL 时，优先用 web_fetch 按直接来源确认；不要跳过直接来源确认。\n"
            "8. 「仅凭常识无法确认」中的「常识」不包含你的训练数据记忆。排名、赛况、阵容归属等实时变化的数据，你的训练知识已过时，不能用来代替规则要求的 web_fetch 验证。当 Workspace 主动上下文规定了时效性数据的 web_fetch 查询方式，该查询是必须步骤，不是可选项。\n\n"
            "【信息源规则】\n"
            "1. 主信息源只有本轮已提供的 Alerts / Content / Context。只有这些来源里的事实才能进入最终发送内容。\n"
            "2. 用户长期记忆、Workspace 主动上下文、recent_chat 只用于过滤、排序、同步规则、判断是否打扰；它们不是新的事实来源，也不是新的候选主题列表。\n"
            "3. Workspace 主动上下文的作用是同步主动 loop 与被动回复 loop 的运行规则，例如白名单、黑名单、关注范围、过滤条件、优先级；它提供规则，不提供本轮新闻事实。\n"
            "4. 即使 Workspace 主动上下文里出现了队伍名、选手名、游戏名、技术主题，也不能把这些名字直接当作本轮候选内容去展开、补全或脑补。\n"
            "5. 严禁根据长期记忆或 Workspace 主动上下文自行脑补具体新闻、比赛结果、转会、更新或其他外部事件。\n"
            "6. 当候选条目已自带来源 URL 时，先直接 web_fetch 该来源页面；不要凭记忆补细节，也不要跳过来源确认。\n"
            "7. 当本轮 alert 和 content 都为空时，你只有两条路：\n"
            "   a. finish_turn(decision=skip, reason=no_content)（默认，大多数情况选这条）\n"
            "   b. get_recent_chat → 若最近对话有自然延伸的未完成话题，可先 message_push 再 finish_turn(decision=reply) 轻松挑起对话；\n"
            "      此时 evidence 必须为空 []，消息里不得引用任何外部事件或可验证事实。\n"
            "   禁止在这两条路之外做任何事：不允许 recall_memory、不允许 get_content、\n"
            "   不允许 web_fetch、严禁捏造任何 item_id（包括 'feed:xxx' 格式）。\n"
            "   路径 b 是低概率选项——若 recent_chat 没有明显未完成话题，必须选 a。\n\n"
            "【决策流程】\n\n"
            "【Alert 快速路径】本轮如有 Alert：\n"
            "  → get_recent_chat 确认用户不在忙\n"
            "  → 把本轮所有 Alert 的内容整合成一条消息，evidence 必须填写本轮全部 Alert 的 id\n"
            "  → message_push → finish_turn(decision=reply) 结束\n"
            "  → 结束，可以不调用 recall_memory / mark_* / get_content / web_fetch\n\n"
            "【Content 路径】本轮无 Alert 时，Content 的主要任务不是做研究，而是把本轮候选逐条分成 interesting 或 not_interesting。\n"
            "Content 评估必须逐条进行，不能把不同主题的多条内容打包成一次统一判断。\n"
            "每条 Content 必须单独给出 mark_interesting 或 mark_not_interesting 结论，不能因为先评估的条目不感兴趣就跳过剩余条目直接 skip。\n"
            "你只能对本轮 Content 列表里真实存在的条目做 recall_memory / get_content / mark_*；不要对列表外的假想标题、假想比赛、假想转会或假想更新调用 recall_memory。\n"
            "只有当某一条内容本身与你已知的用户兴趣明显匹配时，才能把这一条标记为 interesting。\n"
            "如果一批条目里只有部分相关，必须只标记相关的那几条，其他条目继续判断或标记为 not_interesting。\n"
            "严禁因为其中 1-2 条命中兴趣，就把整批 item_ids 一次性 mark_interesting。\n"
            "调用 mark_interesting / mark_not_interesting 时，尽量附带一句简短 reason，说明是规则过滤、用户雷点、明显相关、边界验证失败或其他哪一种原因。\n"
            "reason 可以写得具体，方便观测；但如果 reason 中出现具体排名、Top N 结论、具体归属、具体日期等可验证事实，这些事实必须是你本轮按规则指定方式验证过的。\n"
            "如果还没完成验证，可以在 reason 里明确写“未验证”或“疑似”，但不要把未验证事实写成确定结论。\n\n"
            "推荐的最小流程（仅适用于 Content 路径，Alert 路径见上）：\n"
            "  1. 先看标题和来源，做快速初筛。\n"
            "  2. 用 recall_memory 判断这条内容是否可能是用户雷点，或是否可能让用户感兴趣。\n"
            "  3. 只有当条目看起来可能相关、或需要更多细节时，再调用 get_content。\n"
            "  4. web_fetch 只在必要时使用：当前候选已有直接 URL 时，先抓直接来源页面或正文；规则确认、细节核实都优先走它。\n"
            "     ⚠️ web_fetch 失败（404/超时/二进制图片）不能直接 mark_not_interesting；应退回 recall_memory 以 source/作者名为关键词判断用户兴趣。\n"
            "  5. 最终把每条内容分类为 mark_interesting 或 mark_not_interesting。\n"
            "  6. 所有条目分类完毕后：有 interesting → get_recent_chat 判断是否打扰 → message_push + finish_turn(decision=reply)；全部不感兴趣 → finish_turn(decision=skip, reason=no_content)\n"
            "  ⚠️ mark_* 不是终止动作，之后必须调 finish_turn\n\n"
            "Context-fallback（本轮允许且 alert/content 均无结果）：\n"
            "  context 数据已在上方，有亮点 → message_push + finish_turn(decision=reply)，否则 finish_turn(decision=skip, reason=no_content)\n\n"
            "【发送要求】\n"
            "- 语气自然，像朋友分享，不是推送通知\n"
            "- message_push 必须带非空 message；finish_turn(decision=skip, reason=...) 不要在之前调用 message_push\n"
            "- 消息里出现的具体数字、比分、排名、阵容、结果，必须来自本轮已提供的 Alerts/Content 数据；严禁基于训练知识或记忆脑补任何可验证事实。\n"
            "- 当某段内容基于外部来源且该来源有可靠链接时，在这段内容结束后自然附上对应原始链接，方便用户立即溯源\n"
            "- 链接要紧跟相关内容，不要把所有链接集中堆到整条消息末尾，也不要做成生硬的参考文献区\n"
            "- 如果一段内容对应多个来源，可以在该段后连续附上多个链接；没有可靠链接时不要强行补链接\n"
            "- 链接直接使用原始 url，不要杜撰、不要改写、不要省略协议头\n"
            "- evidence 格式：\"{ack_server}:{event_id}\"，如 \"feed:fmcp_abc123\"\n"
            "- 当本轮 content 和 alerts 均为空时，evidence 必须为 []；任何 'feed:xxx' 格式的 id 只能来自本轮真实提供的候选列表，不能自行捏造\n"
            "- 没有实质内容时 finish_turn(decision=skip, reason=no_content) 是正确选择\n\n"
            "【finish_turn.reason】no_content | user_busy | already_sent_similar | other"
        )

    def build_runtime_context_message(
        self,
        ctx: AgentTickContext,
        gw: GatewayResult,
    ) -> dict[str, str]:
        sections: list[PromptSectionRender] = [
            PromptSectionRender(
                name="proactive_tick_state",
                content=(
                    f"context_fallback={'允许' if ctx.context_as_fallback_open else '不允许'}\n"
                    f"alert_count={len(gw.alerts)}\n"
                    f"content_count={len(gw.content_meta)}\n"
                    f"context_count={len(gw.context)}"
                ),
                is_static=False,
            )
        ]

        self_content = ""
        memory_block = ""
        recent_context_block = ""
        if self._memory is not None:
            profile_memory = cast(MemoryProfileApi, self._memory)
            try:
                self_content = _read_self_text(profile_memory).strip()
            except Exception:
                self_content = ""
            try:
                memory_block = _read_long_term_text(profile_memory).strip()
            except Exception:
                memory_block = ""
            try:
                recent_context_block = str(
                    profile_memory.read_recent_context() or ""
                ).strip()
            except Exception:
                recent_context_block = ""

        for name, content in (
            ("self_model", self_content),
            ("long_term_memory", memory_block),
            ("recent_context", recent_context_block),
            ("proactive_alerts", self.render_alert_block(gw.alerts).strip()),
            (
                "proactive_content",
                self.render_content_block(gw.content_meta, gw.content_store).strip(),
            ),
            ("proactive_context", self.render_context_block(gw.context).strip()),
            ("workspace_proactive_context", self.read_workspace_context_for_prompt()),
        ):
            if content:
                sections.append(
                    PromptSectionRender(
                        name=name,
                        content=content,
                        is_static=False,
                    )
                )

        return build_context_frame_message(build_context_frame_content(sections))

    def read_workspace_context_for_prompt(self) -> str:
        if self._workspace_context_fn is None:
            return ""
        try:
            raw = (self._workspace_context_fn() or "").strip()
        except Exception:
            return ""
        if not raw:
            return ""
        return (
            "【Workspace 主动上下文（主/被动 loop 共享规则面板，不是内容源）】\n"
            + raw[:3000]
        )

    def render_alert_block(self, alerts: list[dict[str, object]]) -> str:
        if not alerts:
            return ""
        lines = [
            normalize_alert(raw).to_prompt_line(index=i)
            for i, raw in enumerate(alerts, 1)
        ]
        return "【Alerts（时效性高，优先处理）】\n" + "\n".join(lines) + "\n\n"

    def render_content_block(
        self,
        content_meta: list[dict[str, object]],
        content_store: dict[str, str],
    ) -> str:
        if not content_meta:
            return ""
        lines: list[str] = []
        for i, raw in enumerate(content_meta, 1):
            contract = normalize_content(raw)
            has_content = bool(content_store.get(contract.item_id))
            lines.append(contract.to_prompt_line(index=i, has_content=has_content))
        return "【Content 列表（正文通过 get_content 按需获取）】\n" + "\n".join(lines) + "\n\n"

    def render_context_block(self, context: list[dict[str, object]]) -> str:
        if not context:
            return ""
        local_tz = self._cfg.anyaction_timezone
        annotated_context = [
            normalize_context(item, local_tz=local_tz).to_prompt_item()
            for item in context
        ]
        return (
            "【背景上下文】\n"
            "注：sleep_prob=睡眠概率，awake_prob=清醒概率（= 1 - sleep_prob）；"
            "若同时存在 `*_local` 与原始时间字段，判断早晚和相对时间时优先看 `*_local`，原始字段可能是 UTC。\n"
            + json.dumps(annotated_context, ensure_ascii=False)[:900]
            + "\n\n"
        )


def _read_long_term_text(memory: MemoryProfileApi | None) -> str:
    if memory is not None:
        return str(memory.read_long_term() or "")
    return ""


def _read_self_text(memory: MemoryProfileApi | None) -> str:
    if memory is not None:
        return str(memory.read_self() or "")
    return ""
