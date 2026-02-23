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

_REWRITE_SYSTEM = (
    "你是一个用户档案维护者（User Profile Curator），"
    "帮助 AI 助手维护关于用户的凝练知识库。"
)

_REWRITE_PROMPT = """\
将以下长期记忆和近期历史整理为一份用户档案。

## 必须完整保留（绝对不能删除或缩写）
- 姓名、生日、学校、专业、身份
- 联系方式（QQ、Telegram、邮箱等）
- 硬件设备型号（CPU、GPU、手机型号等具体型号）
- 文件路径、目录结构（代码目录、工作区路径等）
- 账号 ID、API Key 存储位置、Steam ID 等标识符
- 实习/工作经历的具体项目名称、技术细节
- 游戏具体数据（游戏时长、成就进度等）

## 可以整理的内容
- 合并重复/相似表述（保留信息量更大的那条）
- 从事件推断持久特征（"玩 Nioh/Sekiro/CS2" → "PC 平台玩家，不关注主机内容"）
- 将零散条目归入合适分类

## 必须删除
- 已完成的一次性操作任务（"帮我设置了 X"、"已执行 Y"）
- 纯事件日志（"On YYYY-MM-DD 用户做了 Z"）
- 明确过时的待定项（"将要做 X" 类，且已有结论的）

格式：固定分类 Markdown，每条事实一行 bullet，只写有内容的分类。
分类顺序：
## 用户画像
## 硬件与环境
## 兴趣与偏好
## 健康与习惯
## 关系定位
## 当前状态（动态，标注截至 YYYY-MM）

直接输出 Markdown，不要 JSON，不要代码块。

---

当前记忆：
{memory}

近期历史摘要：
{history}
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
        max_tokens: int = 2048,
        history_max_chars: int = 6000,
    ) -> None:
        self._memory = memory
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens
        self._history_max_chars = history_max_chars

    async def optimize(self) -> None:
        """重写记忆 + 生成问题列表，写回文件。"""
        current_memory = self._memory.read_long_term().strip()
        recent_history = self._read_recent_history()

        if not current_memory and not recent_history:
            logger.info("[memory_optimizer] 记忆和历史均为空，跳过优化")
            return

        # Step 1: Rewrite memory
        new_memory = await self._rewrite_memory(current_memory, recent_history)
        if new_memory:
            self._memory.write_long_term(new_memory)
            logger.info("[memory_optimizer] 记忆已重写 chars=%d", len(new_memory))
        else:
            logger.warning("[memory_optimizer] 记忆重写返回空，保留原有内容")

        # Step 2: Generate questions
        questions = await self._generate_questions(
            new_memory or current_memory, recent_history
        )
        if questions:
            self._memory.write_questions(_format_questions(questions))
            logger.info("[memory_optimizer] 已写入 %d 个问题", len(questions))

    async def _rewrite_memory(self, memory: str, history: str) -> str:
        prompt = _REWRITE_PROMPT.format(
            memory=memory or "（空）",
            history=history or "（无近期历史）",
        )
        try:
            resp = await self._provider.chat(
                messages=[
                    {"role": "system", "content": _REWRITE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                tools=[],
                model=self._model,
                max_tokens=self._max_tokens,
            )
            return (resp.content or "").strip()
        except Exception as e:
            logger.error("[memory_optimizer] 记忆重写失败: %s", e)
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
                return text[-self._history_max_chars:]
            return text
        except Exception:
            return ""


# ── MemoryOptimizerLoop ───────────────────────────────────────────

class MemoryOptimizerLoop:
    def __init__(
        self,
        optimizer: MemoryOptimizer | None,
        _now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._optimizer = optimizer
        self._now_fn = _now_fn or datetime.now
        self._running = False

    async def run(self) -> None:
        self._running = True
        logger.info("[memory_optimizer] 优化循环已启动，每日 00:00 执行")
        while self._running:
            secs = self._seconds_until_midnight()
            logger.info(
                "[memory_optimizer] 距下次优化 %.0f 秒 (%.1f 小时)",
                secs, secs / 3600,
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
        now = self._now_fn()
        tomorrow = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return max(1.0, (tomorrow - now).total_seconds())
