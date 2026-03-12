from __future__ import annotations

import asyncio
import logging
import random as _random_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

from feeds.buffer import FeedBuffer
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
from proactive.engine import ProactiveEngine
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
    DefaultSensePort,
)
from proactive.source_scorer import SourceScorer
from proactive.state import ProactiveStateStore

if TYPE_CHECKING:
    from agent.provider import LLMProvider
    from agent.tools.message_push import MessagePushTool
    from core.memory.port import MemoryPort
    from feeds.registry import FeedRegistry
    from proactive.config import ProactiveConfig
    from proactive.presence import PresenceStore
    from proactive.schedule import ScheduleStore
    from session.manager import SessionManager

logger = logging.getLogger(__name__)


class ProactiveLoopRuntimeMixin:
    def _init_runtime_state(self, config: "ProactiveConfig") -> None:
        self._running = False
        self._manual_trigger_event = asyncio.Event()
        self._manual_trigger_lock = asyncio.Lock()
        self.feed_buffer = self._build_feed_buffer(config)

    def _build_state_store(
        self,
        state_store: ProactiveStateStore | None,
        state_path: Path | None,
    ) -> ProactiveStateStore:
        if state_store is not None:
            return state_store
        return ProactiveStateStore(state_path or Path("proactive_state.json"))

    def _build_feed_buffer(self, config: "ProactiveConfig") -> FeedBuffer | None:
        if not config.feed_poller_enabled:
            return None
        return FeedBuffer(
            ttl_hours=config.feed_poller_buffer_ttl_hours,
            max_per_source=config.feed_poller_buffer_max_per_source,
        )

    def _build_source_scorer(
        self,
        source_scorer: SourceScorer | None,
    ) -> SourceScorer | None:
        if source_scorer is not None:
            return source_scorer
        if not self._cfg.source_scorer_enabled:
            return None
        cache_path = self._resolve_source_scorer_cache_path()
        scorer = SourceScorer(
            light_provider=self._light_provider,
            light_model=self._light_model,
            cache_path=cache_path,
        )
        logger.info(
            "[proactive] SourceScorer 已初始化 model=%s cache=%s total_budget=%d",
            self._light_model,
            cache_path,
            self._cfg.source_scorer_total_budget,
        )
        return scorer

    def _resolve_source_scorer_cache_path(self) -> Path:
        if self._cfg.source_scorer_cache_path:
            return Path(self._cfg.source_scorer_cache_path).expanduser()
        if hasattr(self._state, "path"):
            return self._state.path.parent / "source_scores.json"
        return Path("source_scores.json")

    def _fitbit_url(self) -> str:
        if not getattr(self._cfg, "fitbit_enabled", False):
            return ""
        return self._cfg.fitbit_url

    def _build_fitbit_provider(self):
        if not getattr(self._cfg, "fitbit_enabled", False):
            return None
        from proactive.fitbit_sleep import FitbitSleepProvider

        provider = FitbitSleepProvider(
            url=self._cfg.fitbit_url,
            poll_interval=self._cfg.fitbit_poll_seconds,
            sleeping_modifier=self._cfg.sleep_modifier_sleeping,
        )
        logger.info(
            "[proactive] FitbitSleepProvider 已启动 url=%s interval=%ds sleeping_modifier=%.2f",
            self._cfg.fitbit_url,
            self._cfg.fitbit_poll_seconds,
            self._cfg.sleep_modifier_sleeping,
        )
        return provider

    def _build_item_filter(self) -> ProactiveItemFilter:
        return ProactiveItemFilter(
            cfg=self._cfg,
            state=self._state,
            source_key_fn=_source_key,
            item_id_fn=_item_id,
            semantic_text_fn=_semantic_text,
            build_tfidf_vectors_fn=_build_tfidf_vectors,
            cosine_fn=_cosine_sparse,
        )

    def _build_reflector(self, fitbit_url: str) -> ProactiveReflector:
        return ProactiveReflector(
            provider=self._provider,
            model=self._model,
            max_tokens=self._max_tokens,
            cfg=self._cfg,
            memory_store=self._memory,
            presence=self._presence,
            fitbit_url=fitbit_url,
            hooks=ReflectHooks(
                format_items=_format_items,
                format_recent=_format_recent,
                parse_decision=_parse_decision,
                collect_global_memory=self._build_context_block,
                sample_random_memory=self._sample_random_memory,
                target_session_key=self._target_session_key,
                on_reflect_error=lambda e: _Decision(
                    score=0.0,
                    should_send=False,
                    message="",
                    reasoning=str(e),
                ),
            ),
        )

    def _build_sender(self) -> ProactiveSender:
        return ProactiveSender(
            cfg=self._cfg,
            push_tool=self._push,
            sessions=self._sessions,
            presence=self._presence,
        )

    def _build_feature_scorer(self) -> ProactiveFeatureScorer:
        return ProactiveFeatureScorer(
            provider=self._provider,
            model=self._model,
            max_tokens=self._max_tokens,
            format_items=_format_items,
            format_recent=_format_recent,
            collect_global_memory=self._build_context_block,
        )

    def _build_message_composer(self, fitbit_url: str) -> ProactiveMessageComposer:
        return ProactiveMessageComposer(
            provider=self._provider,
            model=self._model,
            max_tokens=self._max_tokens,
            format_items=_format_items,
            format_recent=_format_recent,
            collect_global_memory=self._build_context_block,
            fitbit_url=fitbit_url,
        )

    def _build_anyaction_gate(self) -> AnyActionGate:
        if hasattr(self._state, "path"):
            quota_path = self._state.path.parent / "proactive_quota.json"
        else:
            quota_path = Path("proactive_quota.json")
        return AnyActionGate(
            cfg=self._cfg,
            quota_store=QuotaStore(quota_path),
            rng=self._rng,
        )

    def _build_sense_port(self, fitbit_provider) -> DefaultSensePort:
        return DefaultSensePort(
            cfg=self._cfg,
            feeds=self._feeds,
            sessions=self._sessions,
            state=self._state,
            item_filter=self._item_filter,
            memory=self._memory,
            presence=self._presence,
            schedule=self._schedule,
            rng=self._rng,
            feed_buffer=self.feed_buffer,
            source_scorer=self._source_scorer,
            feed_store=self._feed_store,
            fitbit=fitbit_provider,
        )

    def _build_decide_port(self) -> DefaultDecidePort:
        return DefaultDecidePort(
            reflector=self._reflector,
            randomize_fn=lambda decision: _decision_with_randomized_score(
                decision,
                strength=self._cfg.decision_score_random_strength,
                rng=self._rng,
            ),
            source_key_fn=_source_key,
            item_id_fn=_item_id,
            semantic_text_fn=_semantic_text,
            semantic_text_max_chars=self._cfg.semantic_dedupe_text_max_chars,
            feature_scorer=self._feature_scorer,
            message_composer=self._message_composer,
        )

    def _build_memory_retrieval_port(self) -> DefaultMemoryRetrievalPort:
        return DefaultMemoryRetrievalPort(
            cfg=self._cfg,
            memory=self._memory,
            item_id_fn=_item_id,
            trace_writer=self._trace_proactive_memory_retrieve,
        )

    def _build_message_deduper(self) -> ProactiveMessageDeduper | None:
        if not self._cfg.message_dedupe_enabled:
            return None
        return ProactiveMessageDeduper(
            provider=self._provider,
            model=self._model,
            max_tokens=self._max_tokens,
        )

    def _build_engine(self) -> ProactiveEngine:
        return ProactiveEngine(
            cfg=self._cfg,
            state=self._state,
            presence=self._presence,
            rng=self._rng,
            sense=self._sense,
            decide=self._decide,
            act=self._sender,
            memory_retrieval=self._memory_retrieval,
            anyaction=self._anyaction,
            message_deduper=self._message_deduper,
            skill_action_runner=self._build_skill_action_runner(),
            light_provider=self._light_provider,
            light_model=self._light_model,
            passive_busy_fn=self._passive_busy_fn,
            stage_trace_writer=(
                lambda payload: self._append_trace_line(
                    "proactive_strategy_trace.jsonl", payload
                )
            ),
        )

    def _log_runtime_config(self) -> None:
        logger.info(
            "[proactive] 去重配置 seen_ttl=%dh delivery_window=%dh semantic_enabled=%s semantic_threshold=%.2f semantic_window=%dh ngram=%d pending_enabled=%s pending_ttl=%dh pending_limit=%d pending_max_per_source=%d pending_max_total=%d use_global_memory=%s memory_max_chars=%d",
            self._cfg.dedupe_seen_ttl_hours,
            self._cfg.delivery_dedupe_hours,
            self._cfg.semantic_dedupe_enabled,
            self._cfg.semantic_dedupe_threshold,
            self._cfg.semantic_dedupe_window_hours,
            self._cfg.semantic_dedupe_ngram,
            self._cfg.pending_queue_enabled,
            self._cfg.pending_item_ttl_hours,
            self._cfg.pending_candidate_limit,
            self._cfg.pending_max_per_source,
            self._cfg.pending_max_total,
            self._cfg.use_global_memory,
            self._cfg.global_memory_max_chars,
        )
        logger.info(
            "[proactive] interest_filter enabled=%s min_score=%.2f top_k=%d explore=%.2f",
            self._cfg.interest_filter.enabled,
            self._cfg.interest_filter.min_score,
            self._cfg.interest_filter.top_k,
            self._cfg.interest_filter.exploration_ratio,
        )

    def _init_runtime_components(self, source_scorer: SourceScorer | None) -> None:
        self._source_scorer = self._build_source_scorer(source_scorer)
        self._log_runtime_config()
        self._item_filter = self._build_item_filter()
        fitbit_url = self._fitbit_url()
        self._reflector = self._build_reflector(fitbit_url)
        self._sender = self._build_sender()
        self._feature_scorer = self._build_feature_scorer()
        self._message_composer = self._build_message_composer(fitbit_url)
        self._anyaction = self._build_anyaction_gate()
        fitbit_provider = self._build_fitbit_provider()
        self._sense = self._build_sense_port(fitbit_provider)
        self._decide = self._build_decide_port()
        self._memory_retrieval = self._build_memory_retrieval_port()
        self._message_deduper = self._build_message_deduper()
        self._engine = self._build_engine()
        self._trace_proactive_config_snapshot()
