"""
proactive_runner.py — ProactiveLoop 场景测试的 Runner。

设计原则：
- 1:1 还原真实 ProactiveLoop（真实 LLM、真实组件链路）。
- StageOverrides 控制哪些阶段直接放行，让测试专注于 decide 层的 LLM 行为。
- bypass_gate  → TestAnyActionGate.should_act() 始终返回 True。
- bypass_score → ProactiveConfig.score_llm_threshold 设为 0，draw_score 必然通过。
- capture_send → TestActPort.send() 拦截消息而不真正推送。
- 测试场景通过 TestSensePort 注入 feed items、对话历史、睡眠信号等，
  无需依赖真实 Fitbit / MCP / SessionManager 数据。
"""
from __future__ import annotations

import copy
import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.config import load_config
from agent.config_models import Config
from agent.provider import LLMProvider
from core.net.http import (
    SharedHttpResources,
    clear_default_shared_http_resources,
    configure_default_shared_http_resources,
)
from feeds.base import FeedItem
from proactive.anyaction import AnyActionGate, QuotaStore
from proactive.components import (
    ProactiveFeatureScorer,
    ProactiveItemFilter,
    ProactiveMessageComposer,
    ProactiveMessageDeduper,
    ProactiveReflector,
    ProactiveSender,
    ReflectHooks,
)
from proactive.config import ProactiveConfig
from proactive.engine import (
    DecisionContext,
    FetchFilterResult,
    FetchSnapshot,
    ProactiveEngine,
)
from proactive.event import AlertEvent, GenericAlertEvent
from proactive.event import GenericContentEvent
from proactive.loop_helpers import (
    _Decision,
    _build_tfidf_vectors,
    _cosine_sparse,
    _decision_with_randomized_score,
    _format_items,
    _format_recent,
    _item_id,
    _parse_decision,
    _semantic_text,
    _source_key,
)
from proactive.ports import (
    DefaultDecidePort,
    DefaultMemoryRetrievalPort,
    ProactiveRetrievedMemory,
    ProactiveSourceRef,
    ProactiveSendMeta,
    RecentProactiveMessage,
)
from proactive.presence import PresenceStore
from proactive.state import ProactiveStateStore
from tests_scenarios.fixtures import ScenarioJudgeSpec
from tests_scenarios.judge_runner import ScenarioJudgeRunner, ScenarioJudgeVerdict
from tests_scenarios.proactive_fixtures import (
    ProactiveScenarioAssertions,
    ProactiveScenarioSpec,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Test doubles
# ──────────────────────────────────────────────────────────────────────────────


class TestSensePort:
    """
    SensePort 的测试实现。
    全部方法从 spec 读取，不依赖真实 SessionManager / Fitbit / MCP。
    """

    def __init__(self, spec: ProactiveScenarioSpec) -> None:
        self._spec = spec

    def compute_energy(self) -> float:
        return self._spec.energy

    def collect_recent(self) -> list[dict]:
        return list(self._spec.recent_messages)

    def collect_recent_proactive(self, n: int = 5) -> list[RecentProactiveMessage]:
        results: list[RecentProactiveMessage] = []
        for m in reversed(self._spec.recent_messages):
            if m.get("role") == "assistant" and m.get("proactive") and m.get("content"):
                ts = None
                raw_ts = str(m.get("timestamp", "") or "").strip()
                if raw_ts:
                    try:
                        ts = datetime.fromisoformat(raw_ts)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                    except Exception:
                        pass
                refs: list[ProactiveSourceRef] = []
                results.append(
                    RecentProactiveMessage(
                        content=str(m["content"]),
                        timestamp=ts,
                        state_summary_tag=str(m.get("state_summary_tag", "none") or "none"),
                        source_refs=refs,
                    )
                )
                if len(results) >= n:
                    break
        return list(reversed(results))

    def compute_interruptibility(
        self, *, now_hour: int, now_utc: datetime, recent_msg_count: int
    ) -> tuple[float, dict[str, float]]:
        # 在测试中直接返回中等可打扰度，不走复杂计算
        return 0.7, {"f_reply": 0.7, "f_activity": 0.7, "f_fatigue": 0.7, "random_delta": 0.0}

    async def fetch_items(self, limit_per_source: int) -> list[FeedItem]:
        return list(self._spec.feed_items)

    def filter_new_items(
        self, items: list[FeedItem]
    ) -> tuple[list[FeedItem], list[tuple[str, str]], list[tuple[str, str]]]:
        # 测试中所有 item 都是"新的"，跳过 seen/semantic 去重
        return list(items), [], []

    def read_memory_text(self) -> str:
        return self._spec.memory_text

    def has_global_memory(self) -> bool:
        return bool(self._spec.memory_text.strip())

    def last_user_at(self) -> datetime | None:
        if self._spec.minutes_since_last_user is None:
            return None
        return datetime.now(timezone.utc).replace(microsecond=0) - __import__(
            "datetime"
        ).timedelta(minutes=self._spec.minutes_since_last_user)

    def refresh_sleep_context(self) -> bool:
        return False

    def target_session_key(self) -> str:
        return "test:scenario"


class TestActPort:
    """拦截发送，记录 message 和 meta，不走真实 MessagePush。"""

    def __init__(self) -> None:
        self.sent_messages: list[dict[str, Any]] = []

    async def send(self, message: str, meta: Any | None = None) -> bool:
        self.sent_messages.append(
            {
                "message": message,
                "meta": meta,
                "sent_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        logger.info("[test_act] 拦截主动消息 len=%d", len(message))
        return True


class TestAnyActionGate:
    """bypass_gate=True 时始终允许通过；False 时走真实 AnyActionGate 逻辑。"""

    def __init__(self, *, bypass: bool, real_gate: AnyActionGate | None = None) -> None:
        self._bypass = bypass
        self._real = real_gate

    def should_act(
        self, *, now_utc: datetime, last_user_at: datetime | None
    ) -> tuple[bool, dict[str, Any]]:
        if self._bypass:
            return True, {"reason": "test_bypass", "used_today": 0, "remaining_today": 999}
        if self._real is not None:
            return self._real.should_act(now_utc=now_utc, last_user_at=last_user_at)
        return True, {"reason": "no_real_gate"}

    def record_action(self, *, now_utc: datetime) -> None:
        if self._real is not None:
            self._real.record_action(now_utc=now_utc)


class TestMemoryRetrievalPort:
    """在测试中跳过向量检索，直接返回空结果（避免依赖真实 memory DB）。"""

    async def retrieve_proactive_context(self, **kwargs: Any) -> ProactiveRetrievedMemory:
        return ProactiveRetrievedMemory.empty("test_bypass")


class TestProactiveEngine(ProactiveEngine):
    """
    继承自 ProactiveEngine，override _stage_fetch_filter 以注入测试数据。

    引擎的其余所有阶段（gate、sense、score、decide、act）完全复用真实逻辑。
    这是唯一需要 override 的地方：真实引擎的 fetch_filter 直接调 MCP 模块，
    无法通过 port 接口注入，只能在子类中替换。
    """

    def __init__(self, *, spec: ProactiveScenarioSpec, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._test_spec = spec

    def _feed_item_to_content_event(self, item: "FeedItem") -> GenericContentEvent:
        from proactive.item_id import compute_item_id

        event_id = compute_item_id(item)
        return GenericContentEvent(
            event_id=event_id,
            source_type=item.source_type,
            source_name=item.source_name,
            content=item.content,
            title=item.title,
            url=item.url,
            published_at=item.published_at,
        )

    def _load_alert_events(self) -> list[AlertEvent]:
        """把 spec.extra_signals["alert_events"] 转成 GenericAlertEvent 对象注入引擎。"""
        raw_alerts = self._test_spec.extra_signals.get("alert_events") or []
        results: list[AlertEvent] = []
        for a in raw_alerts:
            # spec 中 "message" 是可读描述，映射到 payload "content"（
            # GenericAlertEvent._extra_signal_fields 中 "message" = self.content）
            payload = dict(a)
            if "message" in payload and "content" not in payload:
                payload["content"] = payload["message"]
            elif "message" in payload:
                # 优先用更可读的 message 作为 content
                payload["content"] = payload["message"]
            # "ts" → "published_at"
            if "ts" in payload and "published_at" not in payload:
                payload["published_at"] = payload["ts"]
            try:
                results.append(GenericAlertEvent.from_mcp_payload(payload))
            except Exception:
                pass
        return results

    async def _stage_fetch_filter(self, ctx: DecisionContext) -> FetchFilterResult:
        """使用 spec 注入的 feed_items，不调用真实 MCP。"""
        fetch = ctx.ensure_fetch()
        feed_items = list(self._test_spec.feed_items)
        content_events = [self._feed_item_to_content_event(item) for item in feed_items]

        fetch.items = feed_items
        fetch.new_items = content_events
        fetch.new_entries = [
            (f"test:{event.event_id}", event.event_id)
            for event in content_events
        ]
        fetch.semantic_duplicate_entries = []

        # background_context 从 spec.extra_signals 读，不从 MCP 读
        bg = self._test_spec.extra_signals.get("background_context", {})
        fetch.background_context = bg.get("sources", []) if isinstance(bg, dict) else []

        fetch.has_memory = bool(self._test_spec.memory_text.strip())

        return FetchFilterResult(
            total_items=len(feed_items),
            discovered_count=len(feed_items),
            selected_count=len(feed_items),
            semantic_duplicate_count=0,
            pending_enabled=False,
            has_memory=fetch.has_memory,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 结果 & Runner
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ProactiveScenarioResult:
    spec_id: str
    artifact_dir: Path
    passed: bool
    tick_result: float | None       # engine.tick() 的返回值
    sent_messages: list[dict]       # TestActPort 捕获的消息
    decision_snapshot: dict         # engine ctx 的决策快照（从 stage trace 读）
    assertion_errors: list[str] = field(default_factory=list)
    judge_verdict: ScenarioJudgeVerdict | None = None
    error: str = ""

    def failure_message(self) -> str:
        parts = []
        if self.error:
            parts.append(self.error)
        parts.extend(self.assertion_errors)
        if self.judge_verdict and not self.judge_verdict.passed:
            parts.append(f"judge 未通过: {self.judge_verdict.reasons}")
        return "\n".join(parts)

    @property
    def final_message(self) -> str:
        if self.sent_messages:
            return self.sent_messages[-1].get("message", "")
        return ""


class ProactiveScenarioRunner:
    """
    1:1 还原 ProactiveLoop，允许通过 StageOverrides 跳过特定阶段。

    用法：
        runner = ProactiveScenarioRunner()
        result = await runner.run(spec)
        assert result.passed, result.failure_message()
    """

    def __init__(
        self,
        *,
        config_path: str | Path = "config.json",
        artifact_root: str | Path = ".pytest_artifacts/proactive-scenarios",
    ) -> None:
        self._config_path = Path(config_path)
        self._artifact_root = Path(artifact_root)

    async def run(self, spec: ProactiveScenarioSpec) -> ProactiveScenarioResult:
        artifact_dir = self._prepare_artifact_dir(spec.id)
        tick_result: float | None = None
        decision_snapshot: dict = {}
        assertion_errors: list[str] = []
        judge_verdict: ScenarioJudgeVerdict | None = None

        try:
            cfg = self._build_config(spec)
            http_resources = SharedHttpResources()
            configure_default_shared_http_resources(http_resources)

            provider = LLMProvider(
                api_key=cfg.api_key,
                base_url=cfg.base_url,
                system_prompt=cfg.system_prompt,
                extra_body=cfg.extra_body,
                request_timeout_s=60,
                max_retries=0,
            )

            sense_port = TestSensePort(spec)
            act_port = TestActPort()
            gate = TestAnyActionGate(bypass=spec.overrides.bypass_gate)

            # State store 使用临时路径（每次测试隔离）
            state_path = artifact_dir / "proactive_state.json"
            state = ProactiveStateStore(state_path)

            proactive_cfg = self._build_proactive_config(cfg, spec)

            reflector = self._build_reflector(
                provider, proactive_cfg, spec, fallback_model=cfg.model
            )
            decide_port = self._build_decide_port(provider, proactive_cfg, reflector)
            memory_retrieval = TestMemoryRetrievalPort()

            stage_trace_lines: list[dict] = []

            def _stage_trace_writer(payload: dict) -> None:
                stage_trace_lines.append(payload)
                p = artifact_dir / "stage_trace.jsonl"
                with p.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")

            engine = TestProactiveEngine(
                spec=spec,
                cfg=proactive_cfg,
                state=state,
                presence=None,
                rng=None,
                sense=sense_port,
                decide=decide_port,
                act=act_port,
                memory_retrieval=memory_retrieval,
                anyaction=gate,
                message_deduper=None,
                skill_action_runner=None,
                light_provider=provider,
                light_model=cfg.model,
                passive_busy_fn=None,
                stage_trace_writer=_stage_trace_writer,
                observe_writer=None,
            )

            tick_result = await engine.tick()
            decision_snapshot = self._extract_decision_snapshot(stage_trace_lines)

            assertion_errors = self._check_assertions(
                spec.assertions,
                sent_messages=act_port.sent_messages,
                decision_snapshot=decision_snapshot,
            )

            if spec.assertions.judge:
                judge_verdict = await self._run_judge(
                    provider=provider,
                    model=cfg.model,
                    spec=spec,
                    judge=spec.assertions.judge,
                    sent_messages=act_port.sent_messages,
                    decision_snapshot=decision_snapshot,
                )

            passed = not assertion_errors
            if judge_verdict is not None:
                passed = passed and judge_verdict.passed

            result = ProactiveScenarioResult(
                spec_id=spec.id,
                artifact_dir=artifact_dir,
                passed=passed,
                tick_result=tick_result,
                sent_messages=act_port.sent_messages,
                decision_snapshot=decision_snapshot,
                assertion_errors=assertion_errors,
                judge_verdict=judge_verdict,
            )

            clear_default_shared_http_resources(http_resources)
            await http_resources.aclose()

        except Exception as exc:
            logger.exception("[proactive_runner] 场景执行异常 spec_id=%s", spec.id)
            result = ProactiveScenarioResult(
                spec_id=spec.id,
                artifact_dir=artifact_dir,
                passed=False,
                tick_result=tick_result,
                sent_messages=[],
                decision_snapshot=decision_snapshot,
                assertion_errors=assertion_errors,
                error=str(exc),
            )

        self._write_artifacts(artifact_dir, spec, result)
        return result

    # ── 构造器 ──────────────────────────────────────────────────────────────

    def _build_config(self, spec: ProactiveScenarioSpec) -> Config:
        cfg = load_config(self._config_path)
        return cfg

    def _build_proactive_config(
        self, cfg: Config, spec: ProactiveScenarioSpec
    ) -> ProactiveConfig:
        p_cfg = copy.deepcopy(cfg.proactive)
        if spec.overrides.bypass_score:
            # 把 LLM 判断门槛降到 0，draw_score 不管多低都会进入 decide
            p_cfg.score_llm_threshold = 0.0
        # 关闭 pending queue，避免测试引入跨场景状态
        p_cfg.pending_queue_enabled = False
        # 关闭 message 语义去重，保证每次测试都能正常走到发送
        p_cfg.message_dedupe_enabled = False
        # 关闭 semantic dedup
        p_cfg.semantic_dedupe_enabled = False
        return p_cfg

    def _build_reflector(
        self,
        provider: LLMProvider,
        p_cfg: ProactiveConfig,
        spec: ProactiveScenarioSpec,
        *,
        fallback_model: str = "",
    ) -> ProactiveReflector:
        memory_text = spec.memory_text

        def collect_global_memory() -> str:
            return memory_text

        return ProactiveReflector(
            provider=provider,
            model=p_cfg.model or fallback_model,
            max_tokens=1024,
            cfg=p_cfg,
            memory_store=None,
            presence=None,
            fitbit_url="",
            hooks=ReflectHooks(
                format_items=_format_items,
                format_recent=_format_recent,
                parse_decision=_parse_decision,
                collect_global_memory=collect_global_memory,
                sample_random_memory=lambda n: [],
                target_session_key=lambda: "test:scenario",
                on_reflect_error=lambda e: _Decision(
                    score=0.0, should_send=False, message="", reasoning=str(e)
                ),
            ),
        )

    def _build_decide_port(
        self,
        provider: LLMProvider,
        p_cfg: ProactiveConfig,
        reflector: ProactiveReflector,
    ) -> DefaultDecidePort:
        return DefaultDecidePort(
            reflector=reflector,
            randomize_fn=lambda decision: _decision_with_randomized_score(
                decision,
                strength=0.0,  # 测试中不加随机扰动，保证结果稳定
            ),
            source_key_fn=_source_key,
            item_id_fn=_item_id,
            semantic_text_fn=_semantic_text,
            semantic_text_max_chars=p_cfg.semantic_dedupe_text_max_chars,
            feature_scorer=None,
            message_composer=None,
        )

    # ── 断言 ──────────────────────────────────────────────────────────────────

    def _check_assertions(
        self,
        assertions: ProactiveScenarioAssertions,
        *,
        sent_messages: list[dict],
        decision_snapshot: dict,
    ) -> list[str]:
        errors: list[str] = []
        should_send_in_snapshot = decision_snapshot.get("should_send")
        message = sent_messages[-1]["message"] if sent_messages else ""

        if assertions.expected_should_send is not None:
            actual_sent = len(sent_messages) > 0
            if actual_sent != assertions.expected_should_send:
                errors.append(
                    f"should_send 不符: expected={assertions.expected_should_send} "
                    f"actual={actual_sent} "
                    f"(snapshot.should_send={should_send_in_snapshot})"
                )

        # score 断言：reflect mode 下直接用 should_send 推断（sent=True → score≥threshold）
        # feature_final_score 只在 feature_scorer 启用时有效，测试中禁用了
        if assertions.min_score is not None:
            if assertions.min_score > 0.0 and not sent_messages:
                errors.append(
                    f"score 不足（期望 score≥{assertions.min_score} 但消息未发出）"
                )
        if assertions.max_score is not None:
            if assertions.max_score < 1.0 and sent_messages:
                # max_score<0.5 通常表示不应该发消息，若发了则违反预期
                if assertions.max_score < 0.5:
                    errors.append(
                        f"score 超限（期望 score≤{assertions.max_score} 但消息已发出）"
                    )

        if message and assertions.message_contains:
            for needle in assertions.message_contains:
                if needle not in message:
                    errors.append(f"message 缺少关键字: {needle!r}")

        if message and assertions.message_not_contains:
            for needle in assertions.message_not_contains:
                if needle in message:
                    errors.append(f"message 包含了不应出现的内容: {needle!r}")

        return errors

    # ── Judge ─────────────────────────────────────────────────────────────────

    async def _run_judge(
        self,
        *,
        provider: LLMProvider,
        model: str,
        spec: ProactiveScenarioSpec,
        judge: ScenarioJudgeSpec,
        sent_messages: list[dict],
        decision_snapshot: dict,
    ) -> ScenarioJudgeVerdict:
        final_message = sent_messages[-1]["message"] if sent_messages else ""
        rubric_text = "\n".join(f"- {line}" for line in judge.rubric)
        decision_signals_text = json.dumps(
            spec.build_decision_signals(), ensure_ascii=False, indent=2
        )
        prompt = (
            "你是 ProactiveLoop 场景测试的 judge，只能输出 JSON。\n"
            "请根据场景目标、预期结果和 LLM 决策结果判断是否通过。\n\n"
            f"[场景目标]\n{judge.goal}\n\n"
            f"[预期结果]\n{judge.expected_result}\n\n"
            f"[评分标准]\n{rubric_text}\n\n"
            f"[决策信号（注入值）]\n{decision_signals_text}\n\n"
            f"[LLM 决策快照]\n{json.dumps(decision_snapshot, ensure_ascii=False, indent=2)}\n\n"
            f"[最终发送的 message]\n{final_message}\n\n"
            '输出格式：{{"passed": true, "score": 0.95, "reasons": ["..."]}}'
        )
        response = await provider.chat(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            model=model,
            max_tokens=400,
        )
        import re as _re

        raw = (response.content or "").strip()
        if raw.startswith("```"):
            raw = _re.sub(r"^```(?:json)?\s*", "", raw)
            raw = _re.sub(r"\s*```$", "", raw)
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {}
        return ScenarioJudgeVerdict(
            passed=bool(payload.get("passed", False)),
            score=float(payload.get("score", 0.0) or 0.0),
            reasons=[str(r) for r in payload.get("reasons", [])],
            raw_text=response.content or "",
        )

    # ── 工具方法 ──────────────────────────────────────────────────────────────

    def _extract_decision_snapshot(self, trace_lines: list[dict]) -> dict:
        """从 stage trace 里提取 decide 阶段的快照（stage_trace_writer 写的格式）。"""
        for line in reversed(trace_lines):
            payload = line.get("payload", {})
            if payload.get("stage") == "decide":
                result = payload.get("result", {})
                return {
                    "should_send": result.get("should_send"),
                    "score": result.get("feature_final_score") or 0.0,
                    "decision_mode": result.get("decision_mode"),
                    "decision_message": result.get("decision_message", ""),
                    "raw_result": result,
                }
        # 兜底：返回最后一条 trace
        return trace_lines[-1] if trace_lines else {}

    def _prepare_artifact_dir(self, scenario_id: str) -> Path:
        latest_dir = self._artifact_root / scenario_id / "latest"
        if latest_dir.exists():
            shutil.rmtree(latest_dir)
        latest_dir.mkdir(parents=True, exist_ok=True)
        return latest_dir

    def _write_artifacts(
        self,
        artifact_dir: Path,
        spec: ProactiveScenarioSpec,
        result: ProactiveScenarioResult,
    ) -> None:
        def _write(name: str, data: Any) -> None:
            (artifact_dir / name).write_text(
                json.dumps(data, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

        _write("input.json", {
            "id": spec.id,
            "description": spec.description,
            "feed_items": [
                {
                    "title": item.title,
                    "source_name": item.source_name,
                    "content": item.content,
                    "url": item.url,
                    "published_at": item.published_at.isoformat() if item.published_at else None,
                }
                for item in spec.feed_items
            ],
            "recent_messages": spec.recent_messages,
            "memory_text": spec.memory_text,
            "decision_signals": spec.build_decision_signals(),
            "overrides": asdict(spec.overrides),
        })
        _write("summary.json", {
            "spec_id": result.spec_id,
            "passed": result.passed,
            "tick_result": result.tick_result,
            "final_message": result.final_message,
            "error": result.error,
            "assertion_errors": result.assertion_errors,
            "judge_verdict": asdict(result.judge_verdict) if result.judge_verdict else None,
        })
        _write("decision_snapshot.json", result.decision_snapshot)
        _write("sent_messages.json", result.sent_messages)
