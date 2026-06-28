from __future__ import annotations

from typing import Any, Callable

from proactive_v2.config import ProactiveConfig
from proactive_v2.energy import compute_energy, d_energy, next_tick_from_score


class ProactiveScheduler:
    def __init__(
        self,
        *,
        cfg: ProactiveConfig,
        presence: Any,
        rng: Any,
        target_session_key_fn: Callable[[], str],
        trace_fn: Callable[..., None],
    ) -> None:
        self._cfg = cfg
        self._presence = presence
        self._rng = rng
        self._target_session_key_fn = target_session_key_fn
        self._trace_fn = trace_fn

    def next_interval(self, base_score: float | None = None) -> int:
        if not self._presence:
            interval = self._cfg.interval_seconds
            self._trace_fn(
                base_score=base_score,
                interval=interval,
                mode="fixed_no_presence",
            )
            return interval
        if base_score is None:
            session_key = self._target_session_key_fn()
            last_user_at = self._presence.get_last_user_at(session_key)
            energy = compute_energy(last_user_at)
            base_score = d_energy(energy) * self._cfg.score_weight_energy
        interval = next_tick_from_score(
            base_score,
            tick_s1=self._cfg.tick_interval_s1,
            tick_s0=self._cfg.tick_interval_s0,
            tick_jitter=self._cfg.tick_jitter,
            rng=self._rng,
        )
        self._trace_fn(
            base_score=base_score,
            interval=interval,
            mode="adaptive",
        )
        return interval
