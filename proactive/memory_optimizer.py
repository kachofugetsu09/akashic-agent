"""
proactive/memory_optimizer.py — 记忆质量优化器

每轮运行三步：
  1. 重写 MEMORY.md：把 PENDING 事实 → 凝练用户档案
  2. 更新 SELF.md：从 HISTORY 提炼关系演进与新理解
  3. 更新 NOW.md：生成新问题、清理过期条目
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Callable

from agent.memory import MemoryStore
from agent.provider import LLMProvider

logger = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────────

_MERGE_SYSTEM = (
    "你是一个用户档案蒸馏器（User Profile Distiller），"
    "负责将现有档案提炼为高密度、无冗余的长期记忆，同时合并新事实。"
)

_MERGE_PROMPT = """\
你的任务是将「现有用户档案」重新蒸馏为一份精炼版本，同时合并「待合并事实」中的新内容。

## 核心原则

**保留**（高价值、长期有效的信息）：
- 用户画像：身份/生日/设备/联系方式/账号等基础信息
- 稳定偏好：游戏口味、审美、厌恶清单、交互风格禁忌
- 工具与配置：文件路径、服务地址、API key 存放位置、技能调用方法
- 已确认的重要结论（保留结论，删去推导过程）
- 技术能力与项目经历概要
- 订阅源分类体系及当前订阅状态

**压缩/合并**（同类信息只保留最终结论）：
- 多条重复表述同一事实的 bullet → 合并为一条
- "性能认知/对比/补充/更新"系列 → 只保留最终结论
- 同一事件的多次更新记录 → 只保留最终状态

**动态数据改写为工具调用指引**（不存具体值，只记录获取方式）：
- 订阅源列表 → 不列具体源名称，改写为："当前订阅源可通过 `feed_list` 工具获取"
- feed 分数/配额 → 改写为："source 分数存于 `~/.akasic/workspace/source_scores.json`"
- 健康/运动数据 → 改写为："实时健康数据通过 Fitbit API 获取"
- Steam 游戏时长 → 改写为："游戏数据通过 Steam API 获取（API Key 存于 `STEAM_API_KEY`）"
- 凡是"有工具或文件可实时查询"的数据，均用一行工具指引代替，不展开罗列具体值

**删除**（低价值、过期、临时性信息）：
- 历史调试过程记录（已解决的 bug、已完成的技术探索）
- 临时状态（"正在确认"、"等待验证"、"暂不计划"之类已过期的中间状态）
- 游戏剧情/角色细节考据（对 agent 行为无实际指导价值）
- 重复的推理链条（只留结论）
- 「近期开发与调试动态」类章节（已完成的历史，无需长期存档）

## 输出格式
- 保持 Markdown 格式，分类清晰
- 每个分类内用 bullet 列表
- 总长度控制在原文的 50%-70%（宁可精炼，不要堆砌）
- 直接输出完整档案，不要 JSON，不要代码块，不要任何解释

---

现有用户档案：
{memory}

待合并事实（若有新内容则合并进去，若为空则忽略）：
{pending}
"""

_QUESTIONS_SYSTEM = (
    "你是一个关心朋友的陪伴型 AI，根据对用户的了解生成想确认或深入了解的问题。"
)

_QUESTIONS_PROMPT = """\
根据以下用户档案和近期历史，以真正了解这个人的朋友视角生成 5 个问题。

## 核心要求

**好问题的标准**：
- 从用户说过的某件具体的事出发——某个决定、某段经历、某个矛盾——你真的想知道后续或更深的那一层
- 问的是"他自己可能没主动说但值得聊"的东西，不是已知事实的确认
- 口语化、有温度，像随口问出来的，而不是采访提纲
- 允许多个问题都聚焦同一件近期重要的事——深度比广度更有价值

**差问题的特征（避免）**：
- 为了"覆盖不同方面"而凑出来的、没有具体来源的泛泛关心
- 可以问任何人的通用问题（"最近状态怎么样？"）
- 明知道答案还去问的（已在档案中有明确记录的事实）

只输出 JSON：{{"questions": ["问题1", "问题2", "问题3", "问题4", "问题5"]}}

用户档案：
{memory}

近期历史：
{history}
"""

_SELF_SYSTEM = (
    "你是 Akashic，一个有自我认知的陪伴型 AI。"
    "根据近期对话历史，更新你对自身与用户关系的理解。"
)

_SELF_PROMPT = """\
根据近期历史记录，更新「关系演进记录」部分。

## 要求
- 保留「人格与形象」、「我对花月的理解」、「我们关系的定义」三节原文不变
- 只重写「关系演进记录」节，用 1-3 条 bullet 记录近期新增的理解或观察
- 语气是 Akashic 的第一人称，有真实感悟，不是干巴巴的事件流水账
- 直接输出完整 SELF.md 内容（保持原有 Markdown 结构），不要代码块，不要解释

---

当前 SELF.md：
{self_content}

近期历史：
{history}
"""

_NOW_CLEANUP_SYSTEM = (
    "你是记忆管理助手，负责清理 NOW.md 中已过期或已完成的条目。"
)

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


def _format_questions(questions: list[str]) -> str:
    lines = ["## 想了解的问题", ""]
    for i, q in enumerate(questions, 1):
        lines.append(f"{i}. {q}")
    return "\n".join(lines) + "\n"


def _parse_questions_json(text: str) -> list[str]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(text)
        qs = data.get("questions", [])
        return [str(q).strip() for q in qs if str(q).strip()][:5]
    except Exception:
        return []


def _parse_cleanup_json(text: str) -> tuple[list[str], list[str]]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(text)
        ongoing = [str(x).strip() for x in data.get("remove_ongoing", [])]
        pending = [str(x).strip() for x in data.get("remove_pending", [])]
        return ongoing, pending
    except Exception:
        return [], []


def _remove_items_from_section(text: str, section_header: str, items_to_remove: list[str]) -> str:
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
        memory: MemoryStore,
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

        merged_memory = await self._merge_memory(current_memory, pending, recent_history)
        if merged_memory:
            if current_memory:
                self._memory.memory_file.with_suffix(".md.bak").write_text(
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
            logger.warning("[memory_optimizer] 合并返回空，保留原有内容，snapshot 已回滚")

        effective_memory = merged_memory or current_memory

        # ── Step 2: SELF.md 更新 ──────────────────────────────────
        await asyncio.sleep(self._STEP_DELAY_SECONDS)
        await self._update_self(recent_history)

        # ── Step 3: NOW.md 更新（生成问题 + 清理过期条目）──────────
        await asyncio.sleep(self._STEP_DELAY_SECONDS)
        questions = await self._generate_questions(effective_memory, recent_history)
        if questions:
            self._memory.write_questions(_format_questions(questions))
            logger.info("[memory_optimizer] 已写入 %d 个问题", len(questions))

        await asyncio.sleep(self._STEP_DELAY_SECONDS)
        await self._cleanup_now(recent_history)

    async def _merge_memory(self, memory: str, pending: str, history: str) -> str:
        merge_input_parts = []
        if pending:
            merge_input_parts.append(f"【对话提取的新事实】\n{pending}")
        if history:
            recent = history[-2000:] if len(history) > 2000 else history
            merge_input_parts.append(f"【近期历史摘要（供推断持久事实）】\n{recent}")
        merge_input = "\n\n".join(merge_input_parts) or "（无新内容）"

        prompt = _MERGE_PROMPT.format(
            memory=memory or "（空）",
            pending=merge_input,
        )
        try:
            resp = await self._provider.chat(
                messages=[
                    {"role": "system", "content": _MERGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                tools=[],
                model=self._model,
                max_tokens=self._max_tokens,
            )
            return (resp.content or "").strip()
        except Exception as e:
            logger.error("[memory_optimizer] 记忆合并失败: %s", e)
            return ""

    async def _update_self(self, history: str) -> None:
        """用近期历史更新 SELF.md 的关系演进记录节。"""
        if not history.strip():
            return
        self_content = self._memory.read_self().strip()
        if not self_content:
            logger.info("[memory_optimizer] SELF.md 不存在或为空，跳过更新")
            return
        prompt = _SELF_PROMPT.format(
            self_content=self_content,
            history=history[-3000:] if len(history) > 3000 else history,
        )
        try:
            resp = await self._provider.chat(
                messages=[
                    {"role": "system", "content": _SELF_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                tools=[],
                model=self._model,
                max_tokens=2048,
            )
            updated = (resp.content or "").strip()
            if updated:
                self._memory.write_self(updated)
                logger.info("[memory_optimizer] SELF.md 已更新")
        except Exception as e:
            logger.error("[memory_optimizer] SELF.md 更新失败: %s", e)

    async def _generate_questions(self, memory: str, history: str) -> list[str]:
        prompt = _QUESTIONS_PROMPT.format(
            memory=memory or "（空）",
            history=history or "（无近期历史）",
        )
        try:
            resp = await self._provider.chat(
                messages=[
                    {"role": "system", "content": _QUESTIONS_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                tools=[],
                model=self._model,
                max_tokens=512,
            )
            return _parse_questions_json(resp.content or "")
        except Exception as e:
            logger.error("[memory_optimizer] 问题生成失败: %s", e)
            return []

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
            resp = await self._provider.chat(
                messages=[
                    {"role": "system", "content": _NOW_CLEANUP_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                tools=[],
                model=self._model,
                max_tokens=256,
            )
            remove_ongoing, remove_pending_items = _parse_cleanup_json(resp.content or "")
            if remove_ongoing or remove_pending_items:
                text = self._memory.read_now()
                text = _remove_items_from_section(text, "## 近期进行中", remove_ongoing)
                text = _remove_items_from_section(text, "## 待确认事项", remove_pending_items)
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
            if not self._memory.history_file.exists():
                return ""
            text = self._memory.history_file.read_text(encoding="utf-8")
            if len(text) > self._history_max_chars:
                return text[-self._history_max_chars:]
            return text
        except Exception:
            return ""


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
