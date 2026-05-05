"""
proactive/memory_optimizer.py — 记忆质量优化器

每轮运行两步：
  1. 重写 MEMORY.md：把 PENDING 事实 → 凝练用户档案
  2. 更新 SELF.md：只改写既有三段自我认知
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from core.memory.engine import MemoryEngine

from agent.memory import DEFAULT_SELF_MD
from agent.provider import LLMProvider

logger = logging.getLogger(__name__)


class MemoryOptimizerBusy(RuntimeError):
    pass


# ── Prompts ──────────────────────────────────────────────────────

_MERGE_SYSTEM = (
    "你是一个用户长期记忆整理器。"
    "你的工作不是概括对话，而是从记忆中剔除噪音，"
    "只保留对未来每次对话都产生底色影响的长期记忆。"
)

_MERGE_PROMPT = """\
今日日期：{today}

你的任务是将「现有用户档案」重新整理为一份精炼的长期记忆，同时合并「待合并事实」中的新内容。
但更重要的是：**你必须剔除那些不应该存在于用户档案中的内容。**

## 核心判断标准：缺席成本测试

对每一条内容，问自己：
> 在 6 个月后的一次全新对话中，如果这条信息没有被注入，agent 是否会在某个回复中出现方向性失误？

是 → 保留。否 → 删除。

## 三种应保留的内容

- 「用户事实」：关于用户稳定身份的信息——他是谁、他有什么、他身上不可改变或长期稳定的事实
- 「用户偏好」：用户在对话中持续的审美取向、交互禁忌和根本价值判断——不是具体的爱好列表，而是定义了他是什么样的人的方向性偏好
- 「用户明确要求长期记住的关键内容」：用户亲口说"记住""写进长期记忆"的内容，保持原文连贯性，不删减

待合并事实来自 PENDING.md，采用带 tag 的 bullet 格式：
- [identity] ...
- [preference] ...
- [key_info] ...
- [health_long_term] ...
- [requested_memory] ...
- [correction] ...

tag 含义（与 consolidation 阶段一致）：
- identity：基础信息、稳定背景、长期技术方向、经历、长期设备、长期维护项目
- preference：稳定偏好、禁忌、审美、游戏口味、价值取向
- key_info：允许长期保存的 key / token / id / 账号信息
- health_long_term：长期健康状态的一阶事实，不展开动态指标
- requested_memory：用户明确要求长期记住的关键内容；允许比普通事实更连贯、更完整
- correction：对已有 MEMORY.md 内容的显式修正

## 什么必须剔除

### 网络运维细节
内网 IP、路由模式（如"CGNAT""桥接模式""NAT"）、运营商名称、MAC 地址等网络层配置。
→ 这些是瞬时运维信息，不是用户画像。项目路径、配置文件名、环境变量名等与用户开发环境直接相关的信息可以保留。

### 时效性数字和瞬时情绪
具体数字的动态指标（如 Star 数、增长率）、版本变更叙事（"V4 发布后切换"）、瞬时情绪（"失落""失望"）。
→ 保留背后的价值观（如"高度认可某模型"），删除数字和事件过程。

### 临时状态描述
描述当前正在进行、随时会结束的状态（如"最近加班频繁""这周在赶项目""目前在等 offer"）。
→ 与规律性习惯区分：每周/每天持续的行为模式（如"每周去健身房""喜欢手冲咖啡"）可以保留；带"最近""这周""目前"等时间限定词的瞬时状态必须删除。

### Agent 执行规则伪装成用户偏好
以"偏好"开头但实际描述 agent 应如何执行（如检索维度划分策略、元数据标注规范等）。
→ 这些是 procedure，不是用户身份，删除。


## 示例

<example id="drop_network_ops_not_project">
输入：
- 开发环境：项目位于 /home/user/project，配置文件 config.toml，需要设置 DB_PASSWORD
- 家庭网络：电信宽带，桥接模式，内网 IP 192.168.1.x

→ 保留开发环境信息（项目路径和配置是用户的开发画像），删除家庭网络（内网 IP 和路由模式是瞬时运维细节，对理解用户无贡献）。
</example>

<example id="drop_timed_metrics">
输入：
- GitHub 仓库截至 3 月已有 200 Star；用户对增长放缓感到焦虑

→ 全部删除。Star 数会过期，情绪是瞬时的——三个月后这条就是谎言。
</example>

<example id="drop_transient_keep_habit">
输入：
- 用户每周去健身房，主要做力量训练
- 用户最近加班频繁，每天靠咖啡撑着

→ 保留"每周去健身房"（规律性习惯，是用户的长期生活方式）。删除"最近加班频繁，靠咖啡撑着"（带"最近"时间限定词的瞬时状态，随时会变）。
</example>

<example id="drop_meta_rule">
输入：
- 信息检索偏好：偏好按多维度分类搜索，能用标签组合过滤，拒绝单一关键词匹配

→ 全部删除。语义上描述的是 agent 应怎么执行检索任务（procedure），不是用户自身的身份或偏好。即使以"偏好"开头，只要主题是系统运作方式，就不属于用户画像。
</example>


## 整理原则
- 合并同类：把多条表达同一方向偏好的内容合并为一条方向性陈述
- 上收方向：把多个具体知识点上收为一个审美方向
- 同类重复只保留最终版本
- correction 要直接反映到最终内容中，不要保留"旧值 → 新值"痕迹
- 不要生成 agent 执行规则、SOP、工具调用规范
- 不要保留短期状态、时效性事件
- 普通事实保持简洁；requested_memory 允许保留更完整的连贯描述

## 输出格式
- 标题 `# 用户长期记忆`
- 三个大分类：`## 用户事实`、`## 用户偏好`、`## 用户明确要求长期记住的关键内容`
- 每个分类内用 bullet 列表，每条 1-2 行
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

# ── MemoryOptimizer ───────────────────────────────────────────────


class MemoryOptimizer:
    def __init__(
        self,
        memory: "MemoryEngine",
        provider: LLMProvider,
        model: str,
        max_tokens: int = 16384,
    ) -> None:
        self._memory = memory
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens
        self._lock = asyncio.Lock()

    # 各步骤之间的间隔（秒），避免短时间内连续请求触发 limit_burst_rate
    _STEP_DELAY_SECONDS: int = 15

    @property
    def is_running(self) -> bool:
        return self._lock.locked()

    async def optimize(self) -> None:
        """两步优化：合并 PENDING → MEMORY，更新 SELF。"""
        if self._lock.locked():
            raise MemoryOptimizerBusy("memory optimizer 正在运行")
        async with self._lock:
            await self._optimize()

    async def _optimize(self) -> None:
        # ── Step 1: MEMORY.md 合并 ────────────────────────────────
        pending = self._memory.snapshot_pending()
        current_memory = self._memory.read_long_term().strip()

        if not current_memory and not pending:
            logger.info("[memory_optimizer] 记忆和 pending 均为空，跳过优化")
            return

        merged_memory = await self._merge_memory(current_memory, pending)
        if merged_memory:
            if current_memory:
                self._memory.backup_long_term()
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
        self_content = self._memory.read_self().strip() or DEFAULT_SELF_MD.strip()
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

_DEFAULT_INTERVAL_SECONDS = 10800  # 默认每 3 小时整点


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
