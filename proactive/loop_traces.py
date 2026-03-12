from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from core.common.strategy_trace import build_strategy_trace_envelope

logger = logging.getLogger("proactive.loop")


class ProactiveLoopTraceMixin:
    def _trace_proactive_memory_retrieve(self, payload: dict[str, Any]) -> None:
        try:
            memory_dir = self._sessions.workspace / "memory"
            memory_dir.mkdir(parents=True, exist_ok=True)
            trace_file = memory_dir / "proactive_memory_retrieve_trace.jsonl"
            line = {
                **build_strategy_trace_envelope(
                    trace_type="proactive_stage",
                    source="proactive.memory",
                    subject_kind="global",
                    subject_id="proactive-memory",
                    payload=payload,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                **payload,
            }
            with trace_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("[proactive] write proactive memory trace failed: %s", e)

    def _trace_proactive_config_snapshot(self) -> None:
        payload = {
            "enabled": self._cfg.enabled,
            "feature_scoring_enabled": self._cfg.feature_scoring_enabled,
            "threshold": self._cfg.threshold,
            "feature_send_threshold": self._cfg.feature_send_threshold,
            "score_llm_threshold": self._cfg.score_llm_threshold,
            "score_pre_threshold": self._cfg.score_pre_threshold,
            "tick_interval_s0": self._cfg.tick_interval_s0,
            "tick_interval_s1": self._cfg.tick_interval_s1,
            "tick_interval_s2": self._cfg.tick_interval_s2,
            "tick_interval_s3": self._cfg.tick_interval_s3,
            "tick_jitter": self._cfg.tick_jitter,
            "anyaction_enabled": self._cfg.anyaction_enabled,
            "anyaction_min_interval_seconds": self._cfg.anyaction_min_interval_seconds,
            "anyaction_probability_min": self._cfg.anyaction_probability_min,
            "anyaction_probability_max": self._cfg.anyaction_probability_max,
            "memory_retrieval_enabled": self._cfg.memory_retrieval_enabled,
            "memory_top_k_procedure": self._cfg.memory_top_k_procedure,
            "memory_top_k_history": self._cfg.memory_top_k_history,
            "memory_history_gate_enabled": self._cfg.memory_history_gate_enabled,
            "sleep_modifier_sleeping": self._cfg.sleep_modifier_sleeping,
        }
        self._append_trace_line("proactive_config_trace.jsonl", payload)

    def _trace_proactive_rate_decision(
        self,
        *,
        base_score: float | None,
        interval: int,
        mode: str,
    ) -> None:
        payload = {
            "mode": mode,
            "base_score": round(base_score, 4) if base_score is not None else None,
            "interval_seconds": int(interval),
            "threshold": self._cfg.threshold,
            "feature_send_threshold": self._cfg.feature_send_threshold,
            "score_llm_threshold": self._cfg.score_llm_threshold,
            "tick_interval_s0": self._cfg.tick_interval_s0,
            "tick_interval_s1": self._cfg.tick_interval_s1,
            "tick_interval_s2": self._cfg.tick_interval_s2,
            "tick_interval_s3": self._cfg.tick_interval_s3,
            "tick_jitter": self._cfg.tick_jitter,
        }
        self._append_trace_line("proactive_rate_trace.jsonl", payload)

    def _append_trace_line(self, filename: str, payload: dict[str, Any]) -> None:
        try:
            memory_dir = self._sessions.workspace / "memory"
            memory_dir.mkdir(parents=True, exist_ok=True)
            trace_file = memory_dir / filename
            if (
                "trace_type" in payload
                and "payload" in payload
                and "subject" in payload
            ):
                line = payload
            else:
                trace_type = (
                    "proactive_config" if "config" in filename else "proactive_rate"
                )
                source = (
                    "proactive.config"
                    if trace_type == "proactive_config"
                    else "proactive.rate"
                )
                line = {
                    **build_strategy_trace_envelope(
                        trace_type=trace_type,  # type: ignore[arg-type]
                        source=source,
                        subject_kind="global",
                        subject_id=filename.removesuffix(".jsonl"),
                        payload=payload,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                    **payload,
                }
            with trace_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("[proactive] write trace failed %s: %s", filename, e)
