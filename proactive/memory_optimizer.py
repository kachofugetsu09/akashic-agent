"""
proactive/memory_optimizer.py — 每日记忆质量优化器

每天 00:00 运行两步：
  1. 重写 MEMORY.md：把事件日志 → 凝练用户档案（提炼、推断、删除冗余）
  2. 生成 QUESTIONS.md：5 个朋友视角的待确认问题，覆盖式写入
"""

from __future__ import annotations

import asyncio
import json
import logging
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
根据以下用户档案和近期历史，以关心朋友的视角生成 5 个问题。

要求：
- 问模糊的、可能有变化的、或想深入了解的方面（不要问已知事实）
- 简短口语化，像朋友之间的关心或好奇
- 覆盖不同方面（职业/健康/兴趣/心理/日常，不要集中在同一类）
- 只输出 JSON：{{"questions": ["问题1", "问题2", "问题3", "问题4", "问题5"]}}

用户档案：
{memory}

近期历史：
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

    async def optimize(self) -> None:
        """合并 PENDING 事实到 MEMORY + 生成问题列表。"""
        # Phase-1：原子快照 PENDING.md。
        # rename 之后 append_pending 写入新文件，本轮处理内容与后续增量完全隔离。
        pending = self._memory.snapshot_pending()
        current_memory = self._memory.read_long_term().strip()
        recent_history = self._read_recent_history()

        if not current_memory and not pending and not recent_history:
            logger.info("[memory_optimizer] 记忆、pending 和历史均为空，跳过优化")
            # 无快照可提交，直接返回
            return

        # Step 1: 合并 PENDING 到 MEMORY（只增不删）
        merged_memory = await self._merge_memory(
            current_memory, pending, recent_history
        )
        if merged_memory:
            # 合并前备份
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
            # 归档 PENDING 到 HISTORY
            if pending:
                self._memory.append_history(
                    f"[memory_optimizer] PENDING 归档:\n{pending}"
                )
            # Phase-2 成功：删除快照
            self._memory.commit_pending_snapshot()
            logger.info("[memory_optimizer] PENDING 已归档，snapshot 已提交")
        else:
            # Phase-2 失败：快照回滚合并回 PENDING.md，下轮重试
            self._memory.rollback_pending_snapshot()
            logger.warning("[memory_optimizer] 合并返回空，保留原有内容，snapshot 已回滚")

        # Step 2: 生成问题列表
        questions = await self._generate_questions(
            merged_memory or current_memory, recent_history
        )
        if questions:
            self._memory.write_questions(_format_questions(questions))
            logger.info("[memory_optimizer] 已写入 %d 个问题", len(questions))

    async def _merge_memory(self, memory: str, pending: str, history: str) -> str:
        """将 pending 事实合并进 memory，只增不删。"""
        # 构建待合并内容：PENDING + 从 history 推断的持久事实
        merge_input_parts = []
        if pending:
            merge_input_parts.append(f"【对话提取的新事实】\n{pending}")
        if history:
            # 只取最近部分 history 避免 prompt 过长
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

    def _read_recent_history(self) -> str:
        try:
            if not self._memory.history_file.exists():
                return ""
            text = self._memory.history_file.read_text(encoding="utf-8")
            if len(text) > self._history_max_chars:
                return text[-self._history_max_chars :]
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

    def _seconds_until_midnight(self) -> float:
        """计算距下一个午夜 00:00 的秒数（结果始终 > 0）。"""
        now = self._now_fn()
        tomorrow_midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return max(1.0, (tomorrow_midnight - now).total_seconds())

    def _seconds_until_next_tick(self) -> float:
        """计算距下一个对齐整点的秒数。
        例如间隔 3600s：现在 14:23 → 睡到 15:00，共 37 分钟。
        例如间隔 7200s：现在 14:23 → 睡到 16:00，共 1h37m。
        """
        now = self._now_fn()
        # 当前时间距 epoch 的秒数（取整到分钟，忽略秒级抖动）
        now_ts = now.replace(second=0, microsecond=0).timestamp()
        # 下一个对齐 tick 的时间戳
        next_ts = (now_ts // self._interval + 1) * self._interval
        return max(1.0, next_ts - now.timestamp())
