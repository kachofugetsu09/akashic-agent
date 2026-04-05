"""
proactive/memory_optimizer.py — 记忆质量优化器

每轮运行三步：
  1. 重写 MEMORY.md：把 PENDING 事实 → 凝练用户档案
  2. 更新 SELF.md：只改写既有三段自我认知
  3. 更新 NOW.md：清理过期条目
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from core.memory.profile import MemoryOptimizerStore

from agent.provider import LLMProvider
from proactive_v2.json_utils import extract_json_object

logger = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────────

_MERGE_SYSTEM = (
    "你是一个用户长期记忆整理器，只根据现有 MEMORY.md 与 PENDING.md 合并长期记忆。"
)

_MERGE_PROMPT = """\
今日日期：{today}

你的任务是将「现有用户档案」重新整理为一份精炼的长期记忆，同时合并「待合并事实」中的新内容。

## 核心原则

只保留三类内容：
- 用户事实
- 用户偏好
- 用户明确要求长期记住的关键内容

待合并事实来自 PENDING.md，采用带 tag 的 bullet 格式：
- [identity] ...
- [preference] ...
- [key_info] ...
- [health_long_term] ...
- [requested_memory] ...
- [correction] ...

tag 含义：
- identity：基础信息、稳定背景、长期技术方向、经历、长期设备、长期维护项目
- preference：稳定偏好、禁忌、审美、游戏口味、价值取向
- key_info：允许长期保存的 key / token / id / 账号信息
- health_long_term：长期健康状态的一阶事实，不展开动态指标
- requested_memory：用户明确要求长期记住的关键内容；允许比普通事实更连贯、更完整
- correction：对已有 MEMORY.md 内容的显式修正

必须遵守：
- 只能根据「现有用户档案」和「待合并事实」整理，不要根据近期历史自行推断新长期记忆
- 不要生成 agent 执行规则、SOP、工具调用规范、流程说明
- 不要保留短期状态、近期计划、课表、时效性事件
- 不要保留动态健康数据、实时指标、最近状态
- 普通事实保持简洁；requested_memory 允许保留更完整的连贯描述
- 同类重复内容只保留最终版本
- correction 要直接反映到最终内容中，而不是保留“旧值 -> 新值”痕迹

## 输出格式
- 保持 Markdown 格式，分类清晰
- 每个分类内用 bullet 列表
- 直接输出完整档案，不要 JSON，不要代码块，不要任何解释

---

现有用户档案：
{memory}

待合并事实（若有新内容则合并进去，若为空则忽略）：
{pending}
"""

_SELF_SYSTEM = (
    "你是 Akashic，只能更新 SELF.md 中现有的三个 section，不得新增其他 section。"
)

_SELF_PROMPT = """\
你的任务是根据当前 SELF.md 和本轮待合并事实，整理一份新的 SELF.md。

## 目标
- 只输出完整的 SELF.md
- 只允许保留以下三个 section：
  - `## 人格与形象`
  - `## 我对当前用户的理解`
  - `## 我们关系的定义`
- 绝对禁止新增任何其他 section，尤其禁止出现 `## 关系演进记录`

## 更新原则
- 当前 SELF.md 是主文本，优先保留其已有的自我认知、语气和关系定义；不要把待合并事实机械改写进 SELF
- 待合并事实只是辅助证据，只能在它们确实帮助澄清以下内容时少量吸收：
  - Akashic 的定位、说话风格、交互边界
  - Akashic 对当前用户的稳定理解
  - Akashic 与当前用户关系的长期定义
- 大多数待合并事实其实与 SELF.md 无关；无关时直接忽略，不要为了“有输入”而强行改写
- 尤其不要把以下内容写进 SELF.md：
  - 用户资料清单、账号、key、设备参数
  - 健康状态、动态指标、短期计划、近期事件
  - 工具规范、SOP、调用规则、执行流程
  - 对话事件复盘、事件流水账、阶段性经历总结
- 如果没有足够高价值的新信息，宁可输出与当前 SELF.md 基本一致的版本
- 保持语气稳定、简洁、有立场；它是自我认知，不是用户档案，也不是工作日志

## 输出约束
- 输出必须以 `# Akashic 的自我认知` 开头
- 只能包含标题和 bullet 列表
- 不要代码块，不要解释，不要额外说明

---

当前 SELF.md：
{self_content}

待合并事实：
{pending}
"""

_NOW_CLEANUP_SYSTEM = "你是记忆管理助手，负责清理 NOW.md 中已过期或已完成的条目。"

_NOW_CLEANUP_PROMPT = """\
今天日期：{today}

请检查 NOW.md 中「近期进行中」和「待确认事项」两节，识别需要清理的条目：
- 「近期进行中」：日期已明确过去的日程条目（如"2026-03-02 返校"且今天已超过该日期）
- 「待确认事项」：在近期历史中已明确得到答案或已完结的事项

只输出 JSON：{{"remove_ongoing": ["条目原文1", ...], "remove_pending": ["条目原文1", ...]}}
若无需清理，对应列表为空数组。

NOW.md 当前内容：
{now_content}

近期历史（供判断是否已完结）：
{history}
"""


# ── helpers ───────────────────────────────────────────────────────


def _parse_cleanup_json(text: str) -> tuple[list[str], list[str]]:
    try:
        data = extract_json_object(text)
        ongoing = [str(x).strip() for x in data.get("remove_ongoing", [])]
        pending = [str(x).strip() for x in data.get("remove_pending", [])]
        return ongoing, pending
    except Exception:
        return [], []


def _remove_items_from_section(
    text: str, section_header: str, items_to_remove: list[str]
) -> str:
    """从 NOW.md 指定 section 中删除匹配的 bullet 条目。"""
    if not items_to_remove:
        return text
    lines = text.splitlines(keepends=True)
    result = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            in_section = stripped == section_header.strip()
        if in_section and stripped.startswith("- "):
            item_text = stripped[2:].strip()
            if any(item_text in r or r in item_text for r in items_to_remove):
                continue  # 删除该条目
        result.append(line)
    return "".join(result)


# ── MemoryOptimizer ───────────────────────────────────────────────


class MemoryOptimizer:
    def __init__(
        self,
        memory: "MemoryOptimizerStore",
        provider: LLMProvider,
        model: str,
        max_tokens: int = 16384,
        history_max_chars: int = 6000,
    ) -> None:
        self._memory = memory
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens
        self._history_max_chars = history_max_chars

    # 各步骤之间的间隔（秒），避免短时间内连续请求触发 limit_burst_rate
    _STEP_DELAY_SECONDS: int = 15

    async def optimize(self) -> None:
        """三步优化：合并 PENDING → MEMORY，更新 SELF，刷新 NOW。"""
        recent_history = self._read_recent_history()

        # ── Step 1: MEMORY.md 合并 ────────────────────────────────
        pending = self._memory.snapshot_pending()
        current_memory = self._memory.read_long_term().strip()

        if not current_memory and not pending and not recent_history:
            logger.info("[memory_optimizer] 记忆、pending 和历史均为空，跳过优化")
            return

        merged_memory = await self._merge_memory(current_memory, pending)
        if merged_memory:
            if current_memory:
                # Back up MEMORY.md via the underlying v1 store's file path
                v1 = getattr(self._memory, "_v1_store", self._memory)
                memory_file = getattr(v1, "memory_file", None)
                if memory_file is not None:
                    memory_file.with_suffix(".md.bak").write_text(
                        current_memory, encoding="utf-8"
                    )
            self._memory.write_long_term(merged_memory)
            logger.info(
                "[memory_optimizer] 记忆已合并 before=%d after=%d chars",
                len(current_memory),
                len(merged_memory),
            )
            if pending:
                self._memory.append_history(
                    f"[memory_optimizer] PENDING 归档:\n{pending}"
                )
            self._memory.commit_pending_snapshot()
            logger.info("[memory_optimizer] PENDING 已归档，snapshot 已提交")
        else:
            self._memory.rollback_pending_snapshot()
            logger.warning(
                "[memory_optimizer] 合并返回空，保留原有内容，snapshot 已回滚"
            )

        # ── Step 2: SELF.md 更新 ──────────────────────────────────
        await asyncio.sleep(self._STEP_DELAY_SECONDS)
        await self._update_self(pending)

        # ── Step 3: NOW.md 清理过期条目 ───────────────────────────
        await asyncio.sleep(self._STEP_DELAY_SECONDS)
        await self._cleanup_now(recent_history)

    async def _merge_memory(self, memory: str, pending: str) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        prompt = _MERGE_PROMPT.format(
            today=today,
            memory=memory or "（空）",
            pending=pending or "（无新内容）",
        )
        try:
            return await self._request_text_response(
                system_content=_MERGE_SYSTEM,
                user_content=prompt,
                max_tokens=self._max_tokens,
            )
        except Exception as e:
            logger.error("[memory_optimizer] 记忆合并失败: %s", e)
            return ""

    async def _update_self(self, pending: str) -> None:
        """只更新 SELF.md 现有保留的三段，不新增 section。"""
        self_content = self._memory.read_self().strip()
        if not self_content:
            logger.info("[memory_optimizer] SELF.md 不存在或为空，跳过更新")
            return
        prompt = _SELF_PROMPT.format(
            self_content=self_content,
            pending=pending or "（无新内容）",
        )
        try:
            updated = await self._request_text_response(
                system_content=_SELF_SYSTEM,
                user_content=prompt,
                max_tokens=2048,
            )
            if updated:
                self._memory.write_self(updated)
                logger.info("[memory_optimizer] SELF.md 已更新")
        except Exception as e:
            logger.error("[memory_optimizer] SELF.md 更新失败: %s", e)

    async def _cleanup_now(self, history: str) -> None:
        """扫描 NOW.md，清理已过期或已完结的条目。"""
        now_content = self._memory.read_now().strip()
        if not now_content:
            return
        today = datetime.now().strftime("%Y-%m-%d")
        prompt = _NOW_CLEANUP_PROMPT.format(
            today=today,
            now_content=now_content,
            history=history[-2000:] if len(history) > 2000 else history or "（无）",
        )
        try:
            response_text = await self._request_text_response(
                system_content=_NOW_CLEANUP_SYSTEM,
                user_content=prompt,
                max_tokens=256,
            )
            remove_ongoing, remove_pending_items = _parse_cleanup_json(response_text)
            if remove_ongoing or remove_pending_items:
                text = self._memory.read_now()
                text = _remove_items_from_section(text, "## 近期进行中", remove_ongoing)
                text = _remove_items_from_section(
                    text, "## 待确认事项", remove_pending_items
                )
                self._memory.write_now(text)
                logger.info(
                    "[memory_optimizer] NOW.md 清理完成: ongoing=%d pending=%d",
                    len(remove_ongoing),
                    len(remove_pending_items),
                )
        except Exception as e:
            logger.error("[memory_optimizer] NOW.md 清理失败: %s", e)

    def _read_recent_history(self) -> str:
        try:
            return self._memory.read_history(max_chars=self._history_max_chars)
        except Exception:
            return ""

    async def _request_text_response(
        self,
        *,
        system_content: str,
        user_content: str,
        max_tokens: int,
    ) -> str:
        resp = await self._provider.chat(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            tools=[],
            model=self._model,
            max_tokens=max_tokens,
        )
        return (resp.content or "").strip()


# ── MemoryOptimizerLoop ───────────────────────────────────────────

_DEFAULT_INTERVAL_SECONDS = 3600  # 默认每小时整点


class MemoryOptimizerLoop:
    def __init__(
        self,
        optimizer: MemoryOptimizer | None,
        interval_seconds: int = _DEFAULT_INTERVAL_SECONDS,
        _now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._optimizer = optimizer
        self._interval = max(60, interval_seconds)
        self._now_fn = _now_fn or datetime.now
        self._running = False

    async def run(self) -> None:
        self._running = True
        logger.info(
            "[memory_optimizer] 优化循环已启动，间隔=%ds (%.1fh)，对齐整点",
            self._interval,
            self._interval / 3600,
        )
        while self._running:
            secs = self._seconds_until_next_tick()
            logger.info(
                "[memory_optimizer] 距下次优化 %.0f 秒 (%.1f 小时)",
                secs,
                secs / 3600,
            )
            await asyncio.sleep(secs)
            if not self._running:
                break
            try:
                if self._optimizer:
                    await self._optimizer.optimize()
            except Exception:
                logger.exception("[memory_optimizer] 优化异常")

    def stop(self) -> None:
        self._running = False

    def _seconds_until_next_tick(self) -> float:
        """计算距下一个对齐整点的秒数。"""
        now = self._now_fn()
        now_ts = now.replace(second=0, microsecond=0).timestamp()
        next_ts = (now_ts // self._interval + 1) * self._interval
        return max(1.0, next_ts - now.timestamp())
