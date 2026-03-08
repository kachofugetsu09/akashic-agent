import asyncio
import logging

import json_repair

logger = logging.getLogger("agent.loop")


class AgentLoopConsolidationMixin:
    def _on_post_mem_task_done(self, task: asyncio.Task, session_key: str) -> None:
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            logger.info("post_response_memorize task cancelled: %s", session_key)
            return
        except Exception as e:
            self._post_mem_failures += 1
            logger.warning(
                "post_response_memorize task inspection failed session=%s failures=%d err=%s",
                session_key,
                self._post_mem_failures,
                e,
            )
            return

        if exc is not None:
            self._post_mem_failures += 1
            logger.warning(
                "post_response_memorize task failed session=%s failures=%d err=%s",
                session_key,
                self._post_mem_failures,
                exc,
            )

    async def _consolidate_memory_bg(self, session, key: str) -> None:
        try:
            await self._consolidate_memory(session)
            await self.session_manager.save_async(session)
        finally:
            self._consolidating.discard(key)

    async def _consolidate_memory(self, session, archive_all: bool = False) -> None:
        memory = self._memory_port
        consolidate_up_to = len(session.messages)
        if archive_all:
            old_messages = list(session.messages)
            keep_count = 0
            logger.info(
                f"Memory consolidation (archive_all): {len(session.messages)} total messages archived"
            )
        else:
            keep_count = self.memory_window // 2
            if len(session.messages) <= keep_count:
                logger.debug(
                    f"Session {session.key}: No consolidation needed (messages={len(session.messages)}, keep={keep_count})"
                )
                return
            messages_to_process = len(session.messages) - session.last_consolidated
            if messages_to_process <= 0:
                logger.debug(
                    f"Session {session.key}: No new messages to consolidate (last_consolidated={session.last_consolidated}, total={len(session.messages)})"
                )
                return
            consolidate_up_to = len(session.messages) - keep_count
            old_messages = session.messages[
                session.last_consolidated : consolidate_up_to
            ]
            if not old_messages:
                return
            logger.info(
                f"Memory consolidation started: {len(session.messages)} total, {len(old_messages)} new to consolidate, {keep_count} keep"
            )

        source_ref = (
            f"{session.key}@{session.last_consolidated}-{consolidate_up_to}"
            if not archive_all
            else f"{session.key}@archive_all"
        )

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            role = m["role"].upper()
            if m["role"] == "tool":
                continue
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {role}: {m['content']}")
        conversation = "\n".join(lines)
        current_memory = await asyncio.to_thread(memory.read_long_term)

        prompt = f"""你是记忆提取代理（Memory Extraction Agent）。从对话中精确提取结构化信息，返回 JSON。

## 字段说明

### 1. "history_entry" → HISTORY.md
2-5 句事件摘要，以 [YYYY-MM-DD HH:MM] 开头，保留足够细节便于未来 grep 检索。

### 2. "user_facts" → MEMORY.md 候选缓冲
只写用户的**稳定持久事实**，格式为带分类标注的 bullet 列表（对照下方用户档案查重，只写新增内容）。

✓ 写：用户身份/设备/账号/凭证/用户自身技能与经历/稳定偏好/立场/游戏数据
✓ 写：用户对知识库内容（小说/游戏/作品）的情感倾向——但用指针格式，不复制内容本体
✗ 不写：工具或系统的限制、bug、API 状态（不是用户事实）
✗ 不写：一次性操作记录（"用户执行了X"、"已完成Y"）
✗ 不写：对话过程描述
✗ 不写：短期/临时状态（由 agent 通过 update_now 工具实时维护）
✗ 不写：面向 agent 的执行流程、工具顺序、SOP 条款（此类应放在 SOP 或行为记忆，不放用户画像）
✗ 不写：任何"标准流程/默认流程/操作规范/步骤 1-2-3/先做 A 再做 B"式条目，即使它在很多对话中都适用
✗ 不写：任何要求 agent 在回复前/查询前/分析前 调用某工具、读取某文件、执行某 skill/MCP 的规则
✗ 不写：某次任务的操作性细节（配置参数、具体数值、当次选择的方案）——这些不是用户画像
✗ 不写：知识库内容本体（世界观细节、剧情、角色设定）——内容留在 KB 文件，这里只存用户反应

判断标准："如果同一个用户开始一段新对话，这条信息还有意义吗？"若否，不写。
再加一条强约束：只要一条内容的主语是 "agent 应该怎么做"、"系统应该先做什么"、"某类任务的标准做法是什么"，就算它长期有效，也仍然不是 user_facts，必须返回空，不得写入。
宁可漏掉边界模糊的条目，也不要把 SOP/流程/规约写进 user_facts。

**知识库内容的指针化格式**（用户对 KB 内容有明确情感/立场时使用）：
`用户 [情感/立场描述] [内容简称] -> [KB文件路径]`
格式示例：
- [2236偏好] 用户最喜爱的设定：正三角形（具体举例非抽象隐喻）-> ~/.akasic/workspace/kb/2236/summaries/chunks/hime_ending-chunk-0003.source.md
- [2236偏好] 用户对伸司的评价从负面反转为认可其悲剧性 -> ~/.akasic/workspace/kb/2236/summaries/chunks/common-chunk-0039.source.md

若无新事实，返回 ""。
普通格式示例：
- [账号与凭证] UCD Moodle Token: xxxxx (User ID: 22578)
- [偏好] 用户偏好按课程单独分配日历颜色

反例示例（这些都必须返回 ""，不能放进 user_facts）：
- "回复前必须先 read_file 看 SOP"
- "复杂任务要按标准流程先检索再总结"
- "分析前必须调用某个 tool / skill / MCP"

### 3. "corrections" → 覆盖 MEMORY.md 错误记录
仅当对话中有明确信息推翻档案现有记录时才写，必须同时写旧值和新值。若无需纠正，返回 ""。
✗ 不写：工具加载失败/报错（属于运行时故障，不代表用户现实状态发生改变）
✗ 不写：单次工具调用返回异常（需用户本人明确确认才算纠正）
格式示例：
- [纠正] 主力机显示器：档案记录 Dell U2723D → 用户确认实际为 LG 27GP950

### 4. "self_insights" → SELF.md 候选洞察
agent 从本次对话中发现的用户**行为模式新规律**，格式为带 [SELF] 标注的 bullet 列表。
必须是跨对话可泛化的规律，不是描述这次发生了什么。若无新洞察，返回 []。
格式示例：
- [SELF] 用户在涉及日程/课表时要求精确时间，拒绝模糊描述
- [SELF] 用户对工具调用中间步骤不感兴趣，只关心最终结论

---

## 当前用户档案（用于 user_facts 查重）
{current_memory or "（空）"}

## 待处理对话
{conversation}

只返回合法 JSON，不要 markdown 代码块。"""

        try:
            response = await self.provider.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "你是记忆提取代理，只返回合法 JSON。",
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=[],
                model=self.model,
                max_tokens=1024,
            )
            text = (response.content or "").strip()

            if not text:
                logger.warning(
                    "Memory consolidation: LLM returned empty response, skipping"
                )
                return
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json_repair.loads(text)
            if not isinstance(result, dict):
                logger.warning(
                    f"Memory consolidation: unexpected response type, skipping. Response: {text[:200]}"
                )
                return

            if "history_entry" in result:
                await asyncio.to_thread(
                    memory.append_history_once,
                    result["history_entry"],
                    source_ref=source_ref,
                    kind="history_entry",
                )

            user_facts = result.get("user_facts", "")
            if user_facts and isinstance(user_facts, str) and user_facts.strip():
                appended = await asyncio.to_thread(
                    memory.append_pending_once,
                    user_facts,
                    source_ref=source_ref,
                    kind="user_facts",
                )
                if appended:
                    logger.info(
                        f"Memory consolidation: appended {len(user_facts.splitlines())} user_facts to PENDING"
                    )
            corrections = result.get("corrections", "")
            if corrections and isinstance(corrections, str) and corrections.strip():
                appended = await asyncio.to_thread(
                    memory.append_pending_once,
                    corrections,
                    source_ref=source_ref,
                    kind="corrections",
                )
                if appended:
                    logger.info(
                        f"Memory consolidation: appended {len(corrections.splitlines())} corrections to PENDING"
                    )

            self_insights = result.get("self_insights", [])
            if self_insights and isinstance(self_insights, list):
                insight_lines = [
                    s if s.strip().startswith("[SELF]") else f"[SELF] {s.strip()}"
                    for s in self_insights
                    if isinstance(s, str) and s.strip()
                ]
                if insight_lines:
                    appended = await asyncio.to_thread(
                        memory.append_pending_once,
                        "\n".join(insight_lines),
                        source_ref=source_ref,
                        kind="self_insights",
                    )
                    if appended:
                        logger.info(
                            f"Memory consolidation: appended {len(insight_lines)} self_insights to PENDING"
                        )

            history_entry = result.get("history_entry", "")
            asyncio.create_task(
                self._memory_port.save_from_consolidation(
                    history_entry=history_entry,
                    behavior_updates=[],
                    source_ref=source_ref,
                    scope_channel=getattr(session, "_channel", ""),
                    scope_chat_id=getattr(session, "_chat_id", ""),
                )
            )

            if archive_all:
                session.last_consolidated = 0
            else:
                session.last_consolidated = consolidate_up_to
            logger.info(
                f"Memory consolidation done: {len(session.messages)} messages, last_consolidated={session.last_consolidated}"
            )
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
