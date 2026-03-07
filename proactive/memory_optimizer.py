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
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from core.memory.port import MemoryPort

from agent.provider import LLMProvider

logger = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────────

_MERGE_SYSTEM = (
    "你是一个用户档案蒸馏器（User Profile Distiller），"
    "负责将现有档案提炼为高密度、无冗余的长期记忆，同时合并新事实。"
)

_MERGE_PROMPT = """\
今日日期：{today}

你的任务是将「现有用户档案」重新蒸馏为一份精炼版本，同时合并「待合并事实」中的新内容。

## 核心原则

**保留**（高价值、长期稳定的信息）：
- 用户画像：身份/生日/设备/联系方式/账号等基础信息
- 稳定偏好：游戏口味、审美、厌恶清单、交互风格禁忌
- 工具与配置：文件路径、服务地址、API key 存放位置、技能调用方法
- 已确认的**跨时间稳定规律**（保留结论，删去推导过程）
- 技术能力与项目经历概要
- 订阅源分类体系及当前订阅状态

**压缩/合并**（同类信息只保留最终结论）：
- 多条重复表述同一事实的 bullet → 合并为一条
- "性能认知/对比/补充/更新"系列 → 只保留最终结论
- 同一事件的多次更新记录 → 只保留最终状态

**动态数据改写为工具调用指引**（不存具体值，只记录获取方式）：
 - 订阅源列表 → 不列具体源名称，改写为："当前订阅源可通过 `feed_manage(action=list)` 工具获取"
- feed 分数/配额 → 改写为："source 分数存于 `~/.akasic/workspace/source_scores.json`"
- 健康/运动数据（含心率、血氧、睡眠、步数等任何生理指标的具体数值或基线推断）→ 不存数值，改写为："实时数据通过 `fitbit_health_snapshot` 工具查询"
- Steam 游戏时长 → 改写为："游戏数据通过 Steam API 获取（API Key 存于 `STEAM_API_KEY`）"
- 凡是"有工具或文件可实时查询"的数据，均用一行工具指引代替，不展开罗列具体值

**删除**（低价值、过期、临时性信息）：
- 历史调试过程记录（已解决的 bug、已完成的技术探索）
- 临时状态（"正在确认"、"等待验证"、"暂不计划"之类已过期的中间状态）
- 游戏剧情/角色细节考据（对 agent 行为无实际指导价值）
- 重复的推理链条（只留结论）
- 「近期开发与调试动态」类章节（已完成的历史，无需长期存档）
- **时效性事件与状态快照**（以下类型必须删除，不得保留）：
  - 游戏/产品发布日期、赛季信息（如"X月X日定档"、"S16赛季"）——今日日期为 {today}，已过期的必须删除
  - 已过去的日程节点（如"X月X日开学"、"X月X日返校"——日期早于今日的全部删除）
  - 系统/服务/管道连接状态（如"当前管道未接通"、"Token 已失效"）——此类状态随时过期，不应固化进长期记忆
- **任务级操作规范**（针对单一具体任务学到的操作步骤，不应写进通用档案）：
  - 专为某类 RAG 分析任务定制的检索策略、叙事分析方法论
  - 针对某次具体工程任务的操作流程
  - 此类内容若有保留价值，应存入对应 SOP 文件，而非用户档案
- **agent 执行规约类内容**（不应进入用户画像）：
  - "必须先调用某工具"、"回复前必须读取某文件"、"步骤 1/2/3" 等交互铁律
  - 与工具链实现强绑定的流程规范（如 mcp_list、updatenow、task_note 等具体调用约束）
  - 这类内容应存放在 SOP/行为记忆层，不写入 MEMORY.md

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
        memory: "MemoryPort",
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

        merged_memory = await self._merge_memory(
            current_memory, pending, recent_history
        )
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

        effective_memory = merged_memory or current_memory

        # ── Step 2: SELF.md 更新 ──────────────────────────────────
        await asyncio.sleep(self._STEP_DELAY_SECONDS)
        await self._update_self(recent_history)

        # ── Step 3: NOW.md 清理过期条目 ───────────────────────────
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

        today = datetime.now().strftime("%Y-%m-%d")
        prompt = _MERGE_PROMPT.format(
            today=today,
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
            remove_ongoing, remove_pending_items = _parse_cleanup_json(
                resp.content or ""
            )
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
