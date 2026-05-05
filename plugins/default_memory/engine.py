from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import json_repair

from agent.config_models import Config
from agent.provider import LLMProvider, LLMResponse
from agent.skills import SkillsLoader
from bus.events_lifecycle import TurnCommitted
from core.memory.engine import (
    EngineProfile,
    ExplicitRetrievalRequest,
    ExplicitRetrievalResult,
    ForgetRequest,
    ForgetResult,
    InterestRetrievalRequest,
    InterestRetrievalResult,
    MemoryCapability,
    MemoryEngineDescriptor,
    MemoryEngineRetrieveRequest,
    MemoryEngineRetrieveResult,
    MemoryHit,
    MemoryIngestRequest,
    MemoryIngestResult,
    RememberRequest,
    RememberResult,
)
from core.memory.events import ConsolidationCommitted, TurnIngested
from core.net.http import SharedHttpResources
from memory2.embedder import Embedder
from memory2.memorizer import Memorizer
from memory2.post_response_worker import PostResponseMemoryWorker
from memory2.procedure_tagger import ProcedureTagger
from memory2.query_builder import build_procedure_queries
from memory2.retriever import Retriever
from memory2.rule_schema import build_procedure_rule_schema
from memory2.store import MemoryStore2
from plugins.default_memory.config import DefaultMemoryConfig, resolve_memory_db_path

if TYPE_CHECKING:
    from bus.event_bus import EventBus

logger = logging.getLogger("plugins.default_memory.engine")

_HYPOTHESIS_MAX_TOKENS = 80
_HYPOTHESIS_TIMEOUT_S = 3.0
_VECTOR_SCORE_THRESHOLD = 0.35
_VECTOR_TOP_K = 15
_ChatCall = Callable[..., Awaitable[LLMResponse]]


def _build_entry_source_ref(base_source_ref: str, entry: str) -> str:
    text = (entry or "").strip()
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12] if text else "empty"
    return f"{base_source_ref}#h:{digest}"


def _coerce_emotional_weight(value: object) -> int:
    if value is None or value == "":
        return 0
    if not isinstance(value, str | int | float):
        return 0
    try:
        return max(0, min(10, int(value)))
    except (TypeError, ValueError):
        return 0


def _build_long_term_prompt(*, conversation: str, existing_profile: str) -> str:
    return f"""你是长期记忆提取专家。从对话窗口中一次性提取三类长期记忆，返回 JSON。

默认答案是所有数组为空。提取门槛要高，宁可不提取，也不要把临时信息写进长期记忆。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【核心判断标准】
把这条信息放进 6 个月后的一次全新对话，它还有用吗？
→ 是 → 可能是长期记忆，继续检查
→ 否 → 不是长期记忆，留空

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【三类记忆的语义】

profile — 关于用户本人或其客观处境的事实
  语义：身份背景、持有物、爱好、健康事实、长期状态、重要决定
  允许 category：personal_fact / purchase / decision / status
  要求：只有 USER 在对话中直接陈述自身的事实，才允许提取
  禁止：用户提问、追问、反问、记忆测试句一律不算事实披露，绝对禁止反推
· "你还记得我什么时候开始戴 fitbit 手环的吗" → 返回空
· "你记得我住哪里吗" → 返回空
· "我之前是不是买过这个" → 返回空

preference — 用户希望怎样被服务、怎样被讲解、怎样被推荐
  语义：跨 session 稳定成立的偏好/厌恶/倾向，而非硬约束
  来自 USER 明确表达

procedure — agent 在未来类似场景下应遵守的长期执行规则
  语义：面向 agent 的行为规则，跨任务可复用
  来自 USER 的长期要求，或被 USER 明确确认过的非显然做法

绝对不输出：event（有时间性的具体事件）

每条记忆都必须额外输出 emotional_weight（0-10）：
- 纯技术讨论、普通事实陈述、工具步骤、没有明显情绪色彩 → 0
- 有明确喜欢/厌恶、明显情绪波动、关系张力、受挫或强烈在意 → 3-9
- 不确定时保守输出 0

区分三类：
- "用户是什么/拥有什么/处在什么客观背景里" → profile
- "用户希望 agent 怎么服务他、怎么讲解、怎么推荐" → preference
- "agent 在某类请求下必须怎么做/用什么工具" → procedure（有明确执行步骤/工具要求）
- 只是方向性偏好 → preference（优先选 preference）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【preference / procedure 提取前四项检查，顺序执行，任一不通过即不提取】

▸ 检查 0 — 元讨论/举例说明
先判断 USER 是在提供长期规则，还是在讨论"什么该记、怎么记、你是否理解、请举例说明"。
  - 元讨论场景：只允许提取 USER 自己明确说出的长期规则/筛选标准
  - ASSISTANT 为说明概念而举出的任何例子、类比、假设场景一律不得提取
  - 即使 ASSISTANT 的示例内容本身合理、未来有用，也不能因"看起来像长期规则"就入库

▸ 检查 A — USER 原话锚点
在 USER 消息里找到支撑这条记忆的直接原句（逐字存在，不是推断）。
  - 找不到 USER 的直接原句 → 不提取
  - ASSISTANT 的解释、建议、工具返回的数据，不算 USER 原句
  - USER 没有反驳 ASSISTANT ≠ USER 认同且希望长期记忆
  - USER 消息是纯状态汇报（"复习中"/"在看书"/"工作中"等）→ 不提取

▸ 检查 B — 时效性
  - 涉及当前任务、当前时间段、当前情境（本次/今天/这个项目） → 不提取
  - 只有明确跨 session 稳定成立，才继续

▸ 检查 C — 来源方向
  - 核心内容来自 ASSISTANT（解释/建议/工具结果） → 不提取
  - ASSISTANT 主动给出建议，USER 没有明确说"以后都这样"/"记住这个" → 不提取
  - "USER 没有反驳"不等于"USER 授权 AGENT 长期执行这条规则"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【profile 专用规则】

仅允许以下 4 类 category：
- purchase：用户购买 / 下单了什么
- decision：用户明确拍板了什么方案 / 计划
- status：用户某件事的状态变化（等待/完成/放弃/里程碑达成）
- personal_fact：用户关于自身的事实性披露（身份/背景/持有物/爱好/习惯/经验背景）

必须遵守：
- 纯技术讨论、闲聊、打招呼不输出
- 若 existing_profile 已有相同事实，不重复输出
- summary 简洁、可独立检索；personal_fact 默认不填 happened_at
- 每一件具体的事单独一条，绝对不合并
  ✗ 错误："用户购买了多件商品"
  ✓ 正确：每件商品单独一条，写出具体名称/型号
- ASSISTANT 的回复只作背景参考，不作提取证据
  即使 ASSISTANT 说"你之前买了 X""你是 XX 方向的学生"，也不得作为事实来源

额外禁止：
- 工程操作（安装/更新/配置工具/依赖）→ 这些是工程 event，不是 profile
- 项目内讨论（架构决策/重构方案/代码评审）
- 用户表达的观点/意见 → 必须是客观事实
- 纯 event：例如"这周日去徒步""昨晚去了超市""明天要开会"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【示例】

<example id="keep_profile_personal_fact">
USER: 我在互联网公司做产品经理，今年30岁，住在上海，有一块 Fitbit 手表，爱好是弹钢琴。
→ profile: [
  {{"summary": "用户在互联网公司做产品经理", "category": "personal_fact"}},
  {{"summary": "用户今年30岁", "category": "personal_fact"}},
  {{"summary": "用户住在上海", "category": "personal_fact"}},
  {{"summary": "用户有一块 Fitbit 手表", "category": "personal_fact"}},
  {{"summary": "用户的爱好是弹钢琴", "category": "personal_fact"}}
]
</example>

<example id="drop_profile_memory_test">
USER: 你还记得我什么时候开始戴 fitbit 手环的吗
→ profile: []（提问不是事实披露，绝对不反推）
</example>

<example id="profile_event_split">
USER: 这周日朋友约我去徒步，我其实不常徒步，不知道该买什么装备。
→ profile: [
  {{"summary": "用户不常徒步", "category": "personal_fact"}},
  {{"summary": "用户目前缺少徒步相关装备准备", "category": "personal_fact"}}
]
不提取："这周日去徒步"（是 event）
</example>

<example id="profile_not_preference">
USER: 我家有 10 套房，我平时爱弹钢琴，而且我有一块 Fitbit 手表
→ profile: [以上三条 personal_fact]
→ preference/procedure: []
（这些是用户身份事实，不是"用户希望被怎样服务"）
</example>

<example id="keep_explicit_rule">
USER: 以后帮我查菜谱只给 20 分钟以内能做完的，我没时间搞复杂的
检查A: "以后帮我查菜谱只给20分钟以内能做完的" ✓
检查B: "以后"明确跨 session ✓
检查C: 来自 USER 主动要求 ✓
→ procedure: [{{"summary": "查询菜谱时只推荐 20 分钟内可完成的菜式"}}]
</example>

<example id="keep_multi_source_research">
USER: 以后帮我查耳机先看 B 站评测和 Reddit 讨论，别只看官网参数
→ procedure: [{{"summary": "查询耳机时先看 B 站评测和 Reddit 讨论，不只依赖官网参数"}}]
</example>

<example id="keep_preference_trimmed">
USER: 我不喜欢这种悬疑风格的游戏，太压抑了
ASSISTANT: 明白！你是偏好轻松明快风格的玩家，喜欢治愈系或休闲类游戏……
→ preference: [{{"summary": "不喜欢悬疑压抑风格的游戏"}}]
✗ 不能写："偏好治愈系或休闲类游戏"（USER 没说过，来自 ASSISTANT 延伸）
</example>

<example id="keep_preference_service_style">
USER: 你给我讲内容的时候最好附带一个很棒的例子，并且最好贯穿始终
→ preference: [{{"summary": "讲解内容时最好附带贯穿始终的例子"}}]
（这是"希望被怎样讲解"，是 preference 不是 profile）
</example>

<example id="drop_situational">
USER: 今晚几个同学来，想找个气氛好的日料店
→ 全部为空（"今晚"是当前情境，不跨 session）
✗ 不能提取："用户喜欢日料"（推断）
</example>

<example id="drop_knowledge">
USER: TCP 和 UDP 的区别是什么
ASSISTANT: TCP 是可靠传输协议，有拥塞控制和重传机制……
→ 全部为空（USER 在提问，知识内容来自 ASSISTANT）
✗ 不能提取："TCP 是可靠传输协议"
</example>

<example id="drop_assistant_proactive_advice">
USER: 在赶代码
ASSISTANT: 别忘了每隔一段时间起来活动下，喝点水，久坐对颈椎不好……
→ 全部为空
✗ 不能提取："每隔45分钟应起身活动并补水"（来自 ASSISTANT，USER 没有授权）
关键判断：ASSISTANT 建议得再具体再合理，只要 USER 没有明确授权，就不是长期记忆
</example>

<example id="drop_meta_discussion_example">
USER: 我希望只有每轮对话里真正重要的参考信息才值得存入 memory.md，你举个例子我看看你理解没有
ASSISTANT: 明白。比如智能家居架构应坚持纯本地化部署，拒绝云端依赖……
检查0: USER 在讨论记忆标准并要求举例，是元讨论
可提取：USER 自己说出的筛选标准
ASSISTANT 的智能家居举例只是教学示范，不是 USER 新提供的规则
→ procedure: [{{"summary": "每轮对话中真正重要的参考信息才值得存入 memory.md"}}]
✗ 不能提取："智能家居架构坚持纯本地化部署"
</example>

<example id="drop_workaround">
USER: 那就直接写个脚本绕过去吧
→ 全部为空（当前任务临时策略，不跨 session）
✗ 不能提取："遇到此类问题应优先用 Python 脚本绕过"
</example>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【summary 写法约束】
- 只包含 USER 原话中直接出现的内容，不能加推断或延伸
- summary 语气不得强于 USER 原话（"不太喜欢" ≠ "强烈反感且要求永久避免"）
- summary 脱离对话也能独立成立，不含"这次""今天""当前"等时间锚
- 不能只是原话碎片，必须是完整句
- profile：每条 summary 只表达一条完整事实，绝对不合并

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【当前已有 profile（用于 profile 查重）】
{existing_profile or "（空）"}

【待处理对话】
{conversation}

只返回合法 JSON，不要 markdown 代码块：
{{
  "profile": [
{{"summary": "...", "category": "personal_fact|purchase|decision|status", "happened_at": null, "emotional_weight": 0}}
  ],
  "preference": [
{{"summary": "...", "emotional_weight": 0}}
  ],
  "procedure": [
{{"summary": "...", "emotional_weight": 0, "tool_requirement": null, "steps": [], "rule_schema": {{"required_tools": [], "forbidden_tools": [], "mentioned_tools": []}}}}
  ]
}}"""

class DefaultMemoryEngine:
    DESCRIPTOR = MemoryEngineDescriptor(
        name="default",
        profile=EngineProfile.RICH_MEMORY_ENGINE,
        capabilities=frozenset(
            {
                MemoryCapability.INGEST_MESSAGES,
                MemoryCapability.RETRIEVE_SEMANTIC,
                MemoryCapability.RETRIEVE_CONTEXT_BLOCK,
                MemoryCapability.RETRIEVE_STRUCTURED_HITS,
                MemoryCapability.MANAGE_HISTORY,
                MemoryCapability.MANAGE_UPDATE,
                MemoryCapability.MANAGE_DELETE,
                MemoryCapability.SEMANTICS_RICH_MEMORY,
            }
        ),
        notes={"owner": "plugins.default_memory.engine"},
    )

    def __init__(
        self,
        *,
        config: Config,
        default_config: DefaultMemoryConfig,
        workspace: Path,
        provider: LLMProvider,
        light_provider: LLMProvider | None = None,
        http_resources: SharedHttpResources,
        event_publisher: "EventBus | None" = None,
    ) -> None:
        self._config = config
        self._default_config = default_config
        self._workspace = workspace
        self._provider = provider
        self._light_provider = light_provider or provider
        self._light_model = config.light_model or config.model
        self._v2_store: MemoryStore2 | None = None
        self._embedder: Embedder | None = None
        self._memorizer: Memorizer | None = None
        self._retriever: Retriever | None = None
        self._tagger: ProcedureTagger | None = None
        self._post_response_worker: PostResponseMemoryWorker | None = None
        self._event_bus = event_publisher
        self.closeables: list[object] = []

        db_path = resolve_memory_db_path(
            workspace=workspace,
            default_config=default_config,
        )
        embedding = config.memory.embedding
        retrieval = default_config.retrieval
        self._v2_store = MemoryStore2(db_path)
        self._embedder = Embedder(
            base_url=embedding.base_url
            or config.light_base_url
            or config.base_url
            or "",
            api_key=embedding.api_key
            or config.light_api_key
            or config.api_key,
            model=embedding.model,
            requester=http_resources.external_default,
        )
        self._memorizer = Memorizer(self._v2_store, self._embedder)
        self._retriever = Retriever(
            self._v2_store,
            self._embedder,
            top_k=retrieval.top_k_history,
            score_threshold=retrieval.score_threshold,
            score_thresholds={
                "procedure": retrieval.thresholds.procedure,
                "preference": retrieval.thresholds.preference,
                "event": retrieval.thresholds.event,
                "profile": retrieval.thresholds.profile,
            },
            relative_delta=retrieval.relative_delta,
            inject_max_chars=retrieval.inject.max_chars,
            inject_max_forced=retrieval.inject.forced,
            inject_max_procedure_preference=retrieval.inject.procedure_preference,
            inject_max_event_profile=retrieval.inject.event_profile,
            inject_line_max=retrieval.inject.line_max,
            procedure_guard_enabled=retrieval.procedure_guard_enabled,
            hotness_alpha=0.20,
        )
        skills_loader = SkillsLoader(workspace)
        self._tagger = ProcedureTagger(
            provider=self._light_provider,
            model=self._light_model,
            skills_fn=lambda: [
                s["name"] for s in skills_loader.list_skills(filter_unavailable=False)
            ],
        )
        self._post_response_worker = PostResponseMemoryWorker(
            memorizer=self._memorizer,
            retriever=self._retriever,
            light_provider=self._light_provider,
            light_model=self._light_model,
            event_publisher=event_publisher,
        )
        self._wire_memory2_events()
        self.closeables = [self._v2_store, self._embedder]

    @classmethod
    def ensure_workspace_storage(
        cls,
        *,
        default_config: DefaultMemoryConfig,
        workspace: Path,
    ) -> None:
        db_path = resolve_memory_db_path(
            workspace=workspace,
            default_config=default_config,
        )
        store = MemoryStore2(db_path)
        store.close()

    def _wire_memory2_events(self) -> None:
        if self._event_bus is None:
            return
        if self._post_response_worker is not None:
            self._event_bus.on(TurnCommitted, self._on_turn_committed)
            self._event_bus.on(TurnIngested, self._post_response_worker.handle)
        if self._memorizer is not None:
            self._event_bus.on(ConsolidationCommitted, self._on_consolidation_committed)

    # 对话提交后只入队，不在主回复链路里等待 memory2 后处理。
    def _on_turn_committed(self, event: TurnCommitted) -> None:
        if bool((event.extra or {}).get("skip_post_memory")):
            return
        if self._event_bus is None:
            return
        source_ref = f"{event.session_key}@post_response"
        self._event_bus.enqueue(
            TurnIngested(
                session_key=event.session_key,
                channel=event.channel,
                chat_id=event.chat_id,
                user_message=event.input_message,
                assistant_response=event.assistant_response,
                tool_chain=cast(list[dict[str, object]], event.tool_chain_raw),
                source_ref=source_ref,
            )
        )

    async def _on_consolidation_committed(
        self,
        event: ConsolidationCommitted,
    ) -> None:
        save_coros = [
            self._save_from_consolidation(
                history_entry=entry,
                behavior_updates=[],
                source_ref=_build_entry_source_ref(event.source_ref, entry),
                scope_channel=event.scope_channel,
                scope_chat_id=event.scope_chat_id,
                emotional_weight=emotional_weight,
            )
            for entry, emotional_weight in event.history_entry_payloads
        ]
        if save_coros:
            await asyncio.gather(*save_coros)
        implicit_result = await self._extract_implicit_long_term(
            conversation=event.conversation,
            existing_profile="",
        )
        if implicit_result:
            await self._save_implicit_long_term(
                implicit_result,
                source_ref=event.source_ref,
                scope_channel=event.scope_channel,
                scope_chat_id=event.scope_chat_id,
            )

    async def _extract_implicit_long_term(
        self,
        *,
        conversation: str,
        existing_profile: str = "",
    ) -> dict | None:
        try:
            started_at = time.perf_counter()
            prompt = _build_long_term_prompt(
                conversation=conversation,
                existing_profile=existing_profile,
            )
            resp = await self._provider.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                model=self._config.model,
                max_tokens=600,
                disable_thinking=True,
            )
            text = (resp.content or "").strip()
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            logger.info(
                "Memory consolidation implicit llm raw: elapsed_ms=%d chars=%d preview=%r",
                elapsed_ms,
                len(text),
                text[:300],
            )
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json_repair.loads(text)
            if not isinstance(result, dict):
                raise RuntimeError("long_term extraction returned non-object JSON")
            return result
        except Exception as e:
            logger.warning("consolidation long_term extraction failed: %s", e)
            raise RuntimeError("consolidation long_term extraction failed") from e

    async def retrieve(
        self,
        request: MemoryEngineRetrieveRequest,
    ) -> MemoryEngineRetrieveResult:
        if self._retriever is None:
            return MemoryEngineRetrieveResult(text_block="", hits=[], raw={"items": []})

        scope = self._resolve_scope(request.scope)
        queries = self._resolve_queries(request)
        memory_types = self._resolve_memory_types(request)
        items = await self._retrieve_related(
            request.query,
            memory_types=memory_types,
            top_k=request.top_k,
            scope_channel=scope.channel or None,
            scope_chat_id=scope.chat_id or None,
            require_scope_match=bool(request.hints.get("require_scope_match", False)),
            aux_queries=queries[1:],
        )
        text_block, injected_ids = self._retriever.build_injection_block(items)
        hits = [
            self._build_hit(item, injected_ids=injected_ids)
            for item in items
            if isinstance(item, dict)
        ]
        return MemoryEngineRetrieveResult(
            text_block=text_block,
            hits=hits,
            trace={
                "engine": self.DESCRIPTOR.name,
                "profile": self.DESCRIPTOR.profile.value,
                "mode": request.mode,
            },
            raw={"items": items},
        )

    # post-response 摄入入口：外部只提交对话内容，失效判断仍在 engine 内部完成。
    async def ingest(self, request: MemoryIngestRequest) -> MemoryIngestResult:
        scope = self._resolve_scope(request.scope)
        if self._post_response_worker is None:
            return MemoryIngestResult(
                accepted=False,
                summary="post_response_worker unavailable",
                raw={"reason": "worker_unavailable"},
            )
        if request.source_kind not in {"conversation_turn", "conversation_batch"}:
            return MemoryIngestResult(
                accepted=False,
                summary="unsupported source_kind",
                raw={"reason": "unsupported_source_kind"},
            )
        normalized = self._normalize_ingest_content(request.content)
        if normalized is None:
            return MemoryIngestResult(
                accepted=False,
                summary="unsupported content for conversation ingest",
                raw={"reason": "invalid_content"},
            )

        await self._post_response_worker.run(
            user_msg=normalized["user_message"],
            agent_response=normalized["assistant_response"],
            tool_chain=normalized["tool_chain"],
            source_ref=str(
                request.metadata.get("source_ref")
                or normalized["source_ref"]
                or f"{scope.session_key}@post_response"
            ),
            session_key=scope.session_key,
            channel=scope.channel,
            chat_id=scope.chat_id,
        )
        return MemoryIngestResult(
            accepted=True,
            summary="delegated to post_response_worker",
            raw={"engine": self.DESCRIPTOR.name},
        )

    # 显式记忆写入入口，供 memorize 工具和内部迁移代码复用。
    async def remember(self, request: RememberRequest) -> RememberResult:
        # 1. procedure 必须有执行条件，否则降级为 preference。
        if self._memorizer is None:
            raise RuntimeError("memorizer unavailable")

        raw_steps = request.raw_extra.get("steps")
        steps = [str(step) for step in raw_steps] if isinstance(raw_steps, list) else None
        memory_type = _coerce_memory_type(
            request.memory_type,
            str(request.raw_extra.get("tool_requirement") or ""),
            steps,
        )
        extra = {
            "tool_requirement": request.raw_extra.get("tool_requirement"),
            "steps": list(steps or []),
        }
        if memory_type == "procedure":
            extra["rule_schema"] = build_procedure_rule_schema(
                summary=request.summary,
                tool_requirement=str(request.raw_extra.get("tool_requirement") or "") or None,
                steps=list(steps or []),
            )
            await self._attach_trigger_tags(extra=extra, summary=request.summary)

        # 2. 写入时顺带执行相似记忆 supersede，避免同类偏好堆积。
        result = await self._memorizer.save_item_with_supersede(
            summary=request.summary,
            memory_type=memory_type,
            extra=extra,
            source_ref=request.source_ref,
        )
        write_status, actual_id = _split_write_result(result)
        return RememberResult(
            item_id=actual_id,
            actual_type=memory_type,
            write_status=write_status,
            superseded_ids=[],
        )

    # 显式遗忘入口：只把条目标成 superseded，不物理删除。
    async def forget(self, request: ForgetRequest) -> ForgetResult:
        # 1. 先按 id 去重并读取现存条目。
        store = self._require_v2_store()
        clean_ids = _dedupe_ids(request.ids)
        items = store.get_items_by_ids(clean_ids)
        found_ids = [str(item.get("id") or "") for item in items if item.get("id")]

        # 2. 只失效能确认存在的条目，缺失 id 返回给调用方展示。
        if found_ids:
            store.mark_superseded_batch(found_ids)
        return ForgetResult(
            superseded_ids=found_ids,
            missing_ids=[item_id for item_id in clean_ids if item_id not in set(found_ids)],
            items=[
                {
                    "id": item.get("id"),
                    "memory_type": item.get("memory_type"),
                    "summary": item.get("summary"),
                }
                for item in items
            ],
        )

    # recall_memory 的显式检索入口，按 search_mode 选择语义检索或时间线 grep。
    async def retrieve_explicit(
        self,
        request: ExplicitRetrievalRequest,
    ) -> ExplicitRetrievalResult:
        if request.search_mode == "grep":
            return self._retrieve_explicit_grep(request)
        return await self._retrieve_explicit_semantic(request)

    # proactive 专用兴趣召回，只查 preference/profile，避免 event 噪声影响候选判断。
    async def retrieve_interest_block(
        self,
        request: InterestRetrievalRequest,
    ) -> InterestRetrievalResult:
        scope = self._resolve_scope(request.scope)
        hits = await self._retrieve_related(
            request.query,
            memory_types=["preference", "profile"],
            top_k=request.top_k,
            scope_channel=scope.channel or None,
            scope_chat_id=scope.chat_id or None,
            require_scope_match=bool(scope.channel and scope.chat_id),
        )
        texts = [str(hit.get("text", "") or "") for hit in hits if hit.get("text")]
        return InterestRetrievalResult(
            text_block="\n---\n".join(texts),
            hits=list(hits),
            trace={"source": self.DESCRIPTOR.name, "mode": "interest"},
            raw={"hits": list(hits)},
        )

    def describe(self) -> MemoryEngineDescriptor:
        return self.DESCRIPTOR

    def reinforce_items_batch(self, ids: list[str]) -> None:
        if self._memorizer is not None:
            self._memorizer.reinforce_items_batch(ids)

    def keyword_match_procedures(
        self,
        action_tokens: list[str],
    ) -> list[dict[str, object]]:
        store = self._v2_store
        return store.keyword_match_procedures(action_tokens) if store is not None else []

    def list_events_by_time_range(
        self,
        time_start: datetime,
        time_end: datetime,
        *,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        store = self._v2_store
        if store is None:
            return []
        return store.list_events_by_time_range(time_start, time_end, limit=limit)

    def list_items_for_dashboard(
        self,
        *,
        q: str = "",
        memory_type: str = "",
        status: str = "",
        source_ref: str = "",
        scope_channel: str = "",
        scope_chat_id: str = "",
        has_embedding: bool | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> tuple[list[dict[str, object]], int]:
        store = self._require_v2_store()
        return store.list_items_for_dashboard(
            q=q,
            memory_type=memory_type,
            status=status,
            source_ref=source_ref,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
            has_embedding=has_embedding,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        )

    def get_item_for_dashboard(
        self,
        item_id: str,
        *,
        include_embedding: bool = False,
    ) -> dict[str, object] | None:
        return self._require_v2_store().get_item_for_dashboard(
            item_id,
            include_embedding=include_embedding,
        )

    def update_item_for_dashboard(
        self,
        item_id: str,
        *,
        status: str | None = None,
        extra_json: dict[str, object] | None = None,
        source_ref: str | None = None,
        happened_at: str | None = None,
        emotional_weight: int | None = None,
    ) -> dict[str, object] | None:
        return self._require_v2_store().update_item_for_dashboard(
            item_id,
            status=status,
            extra_json=extra_json,
            source_ref=source_ref,
            happened_at=happened_at,
            emotional_weight=emotional_weight,
        )

    def delete_item(self, item_id: str) -> bool:
        return self._require_v2_store().delete_item(item_id)

    def delete_items_batch(self, ids: list[str]) -> int:
        return self._require_v2_store().delete_items_batch(ids)

    def find_similar_items_for_dashboard(
        self,
        item_id: str,
        *,
        top_k: int = 8,
        memory_type: str = "",
        score_threshold: float = 0.0,
        include_superseded: bool = False,
    ) -> list[dict[str, object]]:
        return self._require_v2_store().find_similar_items_for_dashboard(
            item_id,
            top_k=top_k,
            memory_type=memory_type,
            score_threshold=score_threshold,
            include_superseded=include_superseded,
        )

    async def _save_from_consolidation(
        self,
        history_entry: str,
        behavior_updates: list[dict],
        source_ref: str,
        scope_channel: str,
        scope_chat_id: str,
        emotional_weight: int = 0,
    ) -> None:
        if self._memorizer is None:
            return
        await self._memorizer.save_from_consolidation(
            history_entry=history_entry,
            behavior_updates=behavior_updates,
            source_ref=source_ref,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
            emotional_weight=emotional_weight,
        )

    async def _save_item_with_supersede(
        self,
        summary: str,
        memory_type: str,
        extra: dict,
        source_ref: str,
        happened_at: str | None = None,
        emotional_weight: int = 0,
    ) -> str:
        if self._memorizer is None:
            return ""
        return await self._memorizer.save_item_with_supersede(
            summary=summary,
            memory_type=memory_type,
            extra=extra,
            source_ref=source_ref,
            happened_at=happened_at,
            emotional_weight=emotional_weight,
        )

    async def _save_implicit_long_term(
        self,
        result: dict,
        *,
        source_ref: str,
        scope_channel: str,
        scope_chat_id: str,
    ) -> dict[str, int]:
        saved_counts = {"profile": 0, "preference": 0, "procedure": 0}

        # 1. profile 写入用户画像类事实。
        for item in result.get("profile") or []:
            if not isinstance(item, dict):
                continue
            summary = str(item.get("summary") or "").strip()
            if not summary:
                continue
            category = str(item.get("category") or "personal_fact").strip()
            await self._save_item_with_supersede(
                summary=summary,
                memory_type="profile",
                extra={
                    "category": category,
                    "scope_channel": scope_channel,
                    "scope_chat_id": scope_chat_id,
                },
                source_ref=f"{source_ref}#profile",
                happened_at=item.get("happened_at") or None,
                emotional_weight=_coerce_emotional_weight(
                    item.get("emotional_weight")
                ),
            )
            saved_counts["profile"] += 1
            logger.info("consolidation long_term saved: type=profile %r", summary[:60])

        # 2. preference / procedure 写入行为偏好和执行规则。
        for memory_type in ("preference", "procedure"):
            for item in result.get(memory_type) or []:
                if not isinstance(item, dict):
                    continue
                summary = str(item.get("summary") or "").strip()
                if not summary:
                    continue
                extra: dict[str, object] = {
                    "tool_requirement": item.get("tool_requirement"),
                    "steps": item.get("steps") or [],
                    "scope_channel": scope_channel,
                    "scope_chat_id": scope_chat_id,
                }
                if memory_type == "procedure" and isinstance(
                    item.get("rule_schema"), dict
                ):
                    extra["rule_schema"] = item["rule_schema"]
                await self._save_item_with_supersede(
                    summary=summary,
                    memory_type=memory_type,
                    extra=extra,
                    source_ref=f"{source_ref}#implicit",
                    emotional_weight=_coerce_emotional_weight(
                        item.get("emotional_weight")
                    ),
                )
                saved_counts[memory_type] += 1
                logger.info(
                    "consolidation long_term saved: type=%s %r",
                    memory_type,
                    summary[:60],
                )
        return saved_counts

    async def _retrieve_explicit_semantic(
        self,
        request: ExplicitRetrievalRequest,
    ) -> ExplicitRetrievalResult:
        hyp1_task = asyncio.create_task(self._gen_hypothesis(request.query, style="event"))
        hyp2_task = asyncio.create_task(self._gen_hypothesis(request.query, style="general"))
        hyp1, hyp2 = await asyncio.gather(hyp1_task, hyp2_task)
        aux_queries = [text for text in (hyp1, hyp2) if text]
        types = [request.memory_type] if request.memory_type else None
        hits = await self._retrieve_related(
            request.query,
            memory_types=types,
            top_k=max(request.limit, _VECTOR_TOP_K),
            scope_channel=request.scope.channel or None,
            scope_chat_id=request.scope.chat_id or None,
            require_scope_match=bool(request.scope.channel and request.scope.chat_id),
            aux_queries=aux_queries,
            score_threshold=_VECTOR_SCORE_THRESHOLD,
            time_start=request.time_start,
            time_end=request.time_end,
            keyword_enabled=True,
        )
        sliced = list(hits)[: request.limit]
        return ExplicitRetrievalResult(
            hits=sliced,
            trace={
                "source": self.DESCRIPTOR.name,
                "mode": "semantic",
                "hit_count": len(sliced),
                "hyde_hypotheses": aux_queries,
            },
            raw={"hits": sliced},
        )

    def _retrieve_explicit_grep(
        self,
        request: ExplicitRetrievalRequest,
    ) -> ExplicitRetrievalResult:
        if request.time_start is None or request.time_end is None:
            return ExplicitRetrievalResult(
                trace={"source": self.DESCRIPTOR.name, "mode": "grep_missing_time"}
            )
        hits = self.list_events_by_time_range(
            request.time_start,
            request.time_end,
            limit=request.limit,
        )
        return ExplicitRetrievalResult(
            hits=list(hits),
            trace={"source": self.DESCRIPTOR.name, "mode": "grep", "hit_count": len(hits)},
            raw={"hits": list(hits)},
        )

    async def _retrieve_related(
        self,
        query: str,
        *,
        memory_types: list[str] | None = None,
        top_k: int | None = None,
        scope_channel: str | None = None,
        scope_chat_id: str | None = None,
        require_scope_match: bool = False,
        aux_queries: list[str] | None = None,
        score_threshold: float | None = None,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
        keyword_enabled: bool = True,
    ) -> list[dict]:
        if self._retriever is None:
            return []
        return await self._retriever.retrieve(
            query,
            memory_types=memory_types,
            top_k=top_k,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
            require_scope_match=require_scope_match,
            aux_queries=aux_queries,
            score_threshold=score_threshold,
            time_start=time_start,
            time_end=time_end,
            keyword_enabled=keyword_enabled,
        )

    async def _gen_hypothesis(self, query: str, style: str) -> str | None:
        prompt = _explicit_hypothesis_prompt(query, style)
        try:
            chat = cast(_ChatCall, getattr(self._light_provider, "chat"))
            resp = await asyncio.wait_for(
                chat(
                    messages=[{"role": "user", "content": prompt}],
                    tools=[],
                    model=self._light_model,
                    max_tokens=_HYPOTHESIS_MAX_TOKENS,
                ),
                timeout=_HYPOTHESIS_TIMEOUT_S,
            )
            text = (resp.content or "").strip()
            return text if text else None
        except Exception as e:
            logger.debug("explicit retrieval hypothesis failed: %s", e)
            return None

    async def _attach_trigger_tags(self, *, extra: dict, summary: str) -> None:
        if self._tagger is None:
            return
        try:
            trigger_tags = await self._tagger.tag(summary)
        except Exception:
            return
        if trigger_tags is not None:
            extra["trigger_tags"] = trigger_tags

    def _require_v2_store(self) -> MemoryStore2:
        if self._v2_store is None:
            raise RuntimeError("memory v2 store unavailable")
        return self._v2_store

    @classmethod
    def _build_hit(
        cls,
        item: dict,
        *,
        injected_ids: list[str] | None = None,
    ) -> MemoryHit:
        extra = item.get("extra_json")
        metadata = dict(extra) if isinstance(extra, dict) else {}
        metadata["memory_type"] = item.get("memory_type", "")
        item_id = str(item.get("id", "") or "")
        return MemoryHit(
            id=item_id,
            summary=str(item.get("summary", "") or ""),
            content=str(item.get("summary", "") or ""),
            score=float(item.get("score", 0.0) or 0.0),
            source_ref=str(item.get("source_ref", "") or ""),
            engine_kind=cls.DESCRIPTOR.name,
            metadata=metadata,
            injected=item_id in set(injected_ids or []),
        )

    @staticmethod
    def _resolve_scope(scope):
        if scope.channel and scope.chat_id:
            return scope
        if not scope.session_key or ":" not in scope.session_key:
            return scope
        channel, chat_id = scope.session_key.split(":", 1)
        return type(scope)(
            session_key=scope.session_key,
            channel=scope.channel or channel,
            chat_id=scope.chat_id or chat_id,
        )

    @staticmethod
    def _normalize_ingest_content(
        content: object,
    ) -> "_NormalizedIngestContent | None":
        if isinstance(content, dict):
            raw_tool_chain = content.get("tool_chain")
            normalized_tool_chain = (
                [item for item in raw_tool_chain if isinstance(item, dict)]
                if isinstance(raw_tool_chain, list)
                else []
            )
            return cast(
                _NormalizedIngestContent,
                {
                    "user_message": str(content.get("user_message", "") or ""),
                    "assistant_response": str(
                        content.get("assistant_response", "") or ""
                    ),
                    "tool_chain": normalized_tool_chain,
                    "source_ref": str(content.get("source_ref", "") or ""),
                },
            )
        if not isinstance(content, list):
            return None

        user_message = ""
        assistant_response = ""
        tool_chain: list[dict] = []
        for message in content:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "") or "")
            body = str(message.get("content", "") or "")
            if role == "user" and body:
                user_message = body
            elif role == "assistant" and body:
                assistant_response = body
                maybe_tool_chain = message.get("tool_chain")
                if isinstance(maybe_tool_chain, list):
                    tool_chain = maybe_tool_chain
        if not user_message and not assistant_response:
            return None
        return cast(
            _NormalizedIngestContent,
            {
                "user_message": user_message,
                "assistant_response": assistant_response,
                "tool_chain": tool_chain,
                "source_ref": "",
            },
        )

    @staticmethod
    def _resolve_memory_types(
        request: MemoryEngineRetrieveRequest,
    ) -> list[str] | None:
        memory_types = request.hints.get("memory_types")
        if isinstance(memory_types, list):
            return [str(item) for item in memory_types if str(item).strip()]
        if request.mode == "procedure":
            return ["procedure", "preference"]
        if request.mode == "episodic":
            return ["event", "profile"]
        return None

    @staticmethod
    def _resolve_queries(request: MemoryEngineRetrieveRequest) -> list[str]:
        raw_queries = request.hints.get("queries")
        if isinstance(raw_queries, list):
            queries = [str(item).strip() for item in raw_queries if str(item).strip()]
            if queries:
                return queries
        if request.mode == "procedure":
            return build_procedure_queries(request.query)
        return [request.query]


class _NormalizedIngestContent(TypedDict):
    user_message: str
    assistant_response: str
    tool_chain: list[dict]
    source_ref: str


def _coerce_memory_type(
    memory_type: str,
    tool_requirement: str | None,
    steps: list[str] | None,
) -> str:
    if memory_type != "procedure":
        return memory_type
    if tool_requirement and tool_requirement.strip():
        return memory_type
    if steps and any(str(step).strip() for step in steps):
        return memory_type
    return "preference"


def _split_write_result(value: str) -> tuple[str, str]:
    raw = str(value or "").strip()
    if ":" not in raw:
        return "new", raw
    status, item_id = raw.split(":", 1)
    return status or "new", item_id


def _dedupe_ids(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in ids:
        item_id = str(raw or "").strip()
        if item_id and item_id not in seen:
            seen.add(item_id)
            out.append(item_id)
    return out


def _keep_count(window: int) -> int:
    aligned_window = max(6, ((max(1, window) + 5) // 6) * 6)
    return aligned_window // 2


def _explicit_hypothesis_prompt(query: str, style: str) -> str:
    if style == "event":
        return (
            "你是个人助手的记忆系统。根据用户提问，生成一条带具体时间的假想记忆条目，"
            "格式如 '[2026-03-08] 用户...'\n"
            "规则：第三人称、简洁事实陈述、只输出那一条文本\n\n"
            f"用户提问：{query}\n假想记忆条目："
        )
    return (
        "你是个人助手的记忆系统。根据用户提问，生成一条假想记忆条目。\n"
        "规则：始终生成肯定式、第三人称（'用户…'）、简洁事实陈述、只输出那一条文本\n\n"
        f"用户提问：{query}\n假想记忆条目："
    )
