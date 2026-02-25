from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Protocol

from feeds.base import FeedItem
from proactive.energy import (
    composite_score,
    d_content,
    d_energy,
    d_recent,
    random_weight,
)
from proactive.interest import select_interesting_items
from proactive.presence import PresenceStore
from proactive.anyaction import AnyActionGate
from proactive.ports import ActPort, DecidePort, SensePort
from proactive.state import ProactiveStateStore
from proactive.skill_action import SkillActionRunner

logger = logging.getLogger(__name__)


class DecisionLike(Protocol):
    score: float
    should_send: bool
    message: str
    reasoning: str
    evidence_item_ids: list[str]


class _PseudoDecision:
    def __init__(self, message: str) -> None:
        self.message = message
        self.evidence_item_ids: list[str] = []


class ProactiveEngine:
    """主动循环引擎：编排一次完整 tick，不直接依赖 ProactiveLoop。"""

    def __init__(
        self,
        *,
        cfg: Any,
        state: ProactiveStateStore,
        presence: PresenceStore | None,
        rng: Any,
        sense: SensePort,
        decide: DecidePort,
        act: ActPort,
        anyaction: AnyActionGate | None = None,
        message_deduper: Any | None = None,
        skill_action_runner: SkillActionRunner | None = None,
    ) -> None:
        self._cfg = cfg
        self._state = state
        self._presence = presence
        self._rng = rng
        self._sense = sense
        self._decide = decide
        self._act = act
        self._anyaction = anyaction
        self._message_deduper = message_deduper
        self._skill_action_runner = skill_action_runner

    async def tick(self) -> float | None:
        """执行一次主动判断循环。
        返回 base_score（float）供调度器调整下次 tick 间隔；
        返回 None 表示 gate 以 min_interval/probability 拒绝，由调度器按能量自算间隔。
        """
        logger.info("[proactive] tick 开始")
        self._state.cleanup(
            seen_ttl_hours=self._cfg.dedupe_seen_ttl_hours,
            delivery_ttl_hours=self._cfg.delivery_dedupe_hours,
            semantic_ttl_hours=max(
                self._cfg.dedupe_seen_ttl_hours, self._cfg.semantic_dedupe_window_hours
            ),
            rejection_cooldown_ttl_hours=getattr(
                self._cfg, "llm_reject_cooldown_hours", 0
            ),
        )
        now_utc = datetime.now(timezone.utc)
        if self._cfg.anyaction_enabled and self._anyaction:
            should_act, meta = self._anyaction.should_act(
                now_utc=now_utc,
                last_user_at=self._sense.last_user_at(),
            )
            if not should_act:
                logger.info(
                    "[proactive] gate_result=reject selected_action=null meta=%s", meta
                )
                reason = meta.get("reason", "")
                if reason == "quota_exhausted":
                    return 0.0  # 今日配额已满，用最长间隔（tick_s0）
                return None  # min_interval / probability：调度器按能量自算
            logger.info("[proactive] gate_result=pass meta=%s", meta)

        energy = self._sense.compute_energy()
        now_hour = datetime.now().hour

        recent = self._sense.collect_recent()
        de = d_energy(energy)
        dr = d_recent(len(recent), self._cfg.score_recent_scale)
        interruptibility, interrupt_detail = self._sense.compute_interruptibility(
            now_hour=now_hour,
            now_utc=now_utc,
            recent_msg_count=len(recent),
        )
        interrupt_factor = 0.6 + 0.4 * interruptibility
        w_sum = self._cfg.score_weight_energy + self._cfg.score_weight_recent
        pre_score = (
            (self._cfg.score_weight_energy * de + self._cfg.score_weight_recent * dr)
            / w_sum
            if w_sum > 0
            else 0.0
        ) * interrupt_factor

        logger.info(
            "[proactive] pre_score=%.3f interrupt=%.3f factor=%.3f"
            " (time=%.2f reply=%.2f activity=%.2f fatigue=%.2f rand=%+.2f)"
            " D_energy=%.3f D_recent=%.3f energy=%.3f msg_count=%d",
            pre_score,
            interruptibility,
            interrupt_factor,
            interrupt_detail["f_time"],
            interrupt_detail["f_reply"],
            interrupt_detail["f_activity"],
            interrupt_detail["f_fatigue"],
            interrupt_detail["random_delta"],
            de,
            dr,
            energy,
            len(recent),
        )
        if pre_score < self._cfg.score_pre_threshold:
            logger.info(
                "[proactive] pre_score 过低（%.3f < %.2f），跳过本轮",
                pre_score,
                self._cfg.score_pre_threshold,
            )
            logger.info("[proactive] selected_action=idle reason=pre_score")
            return pre_score

        items = await self._sense.fetch_items(self._cfg.items_per_source)
        logger.info("[proactive] 拉取到 %d 条信息", len(items))
        new_items, new_entries, semantic_duplicate_entries = (
            self._sense.filter_new_items(items)
        )
        logger.info(
            "[proactive] 去重后剩余新信息 %d 条（过滤重复 %d 条）",
            len(new_items),
            len(items) - len(new_items),
        )
        if semantic_duplicate_entries:
            # 语义重复条目不写入 seen_items（14天 TTL），避免误判导致长期压制。
            # 这些条目仍受 semantic_items（72h TTL）窗口抑制；窗口到期后可重新参与决策。
            logger.info(
                "[proactive] 语义重复条目 count=%d 不写入 seen_items，72h 窗口自然抑制",
                len(semantic_duplicate_entries),
            )

        if self._cfg.interest_filter.enabled and new_items:
            memory_text = self._sense.read_memory_text()
            if memory_text:
                filtered_items, ranked = select_interesting_items(
                    new_items, memory_text, self._cfg.interest_filter
                )
                keep_ids = {self._decide.item_id_for(item) for item in filtered_items}
                old_count = len(new_items)
                new_items = filtered_items
                new_entries = [
                    (source_key, item_id)
                    for source_key, item_id in new_entries
                    if item_id in keep_ids
                ]
                top_preview = ", ".join(
                    f"{(pair[0].title or '')[:28]}:{pair[1]:.2f}" for pair in ranked[:3]
                )
                logger.info(
                    "[proactive] memory 兴趣筛选 old=%d kept=%d min_score=%.2f top=%s",
                    old_count,
                    len(new_items),
                    self._cfg.interest_filter.min_score,
                    top_preview or "-",
                )
            else:
                logger.info("[proactive] memory 兴趣筛选跳过：memory 为空")

        has_memory = self._sense.has_global_memory()
        if (
            self._cfg.only_new_items_trigger
            and not new_items
            and not self._presence
            and not has_memory
        ):
            logger.info(
                "[proactive] 无新信息且 only_new_items_trigger=true（无 presence），跳过本轮反思"
            )
            logger.info("[proactive] selected_action=idle reason=no_new_items")
            return pre_score

        dc = d_content(len(new_items), self._cfg.score_content_halfsat)
        base_score = (
            composite_score(
                de,
                dc,
                dr,
                self._cfg.score_weight_energy,
                self._cfg.score_weight_content,
                self._cfg.score_weight_recent,
            )
            * interrupt_factor
        )

        w_random = random_weight(rng=self._rng)
        draw_score = base_score * w_random
        session_key = self._sense.target_session_key()
        target_last_user = (
            self._presence.get_last_user_at(session_key)
            if self._presence and session_key
            else None
        )
        last_proactive_at = (
            self._presence.get_last_proactive_at(session_key)
            if self._presence and session_key
            else None
        )
        force_reflect = (
            energy < 0.05
            or (
                self._presence is not None
                and bool(session_key)
                and target_last_user is None
            )
            or (energy < 0.20 and has_memory)
        )
        logger.info(
            "[proactive] base_score=%.3f  D_energy=%.3f D_content=%.3f D_recent=%.3f"
            "  interrupt=%.3f W_random=%.2f → draw_score=%.3f 阈值=%.2f force_reflect=%s",
            base_score,
            de,
            dc,
            dr,
            interruptibility,
            w_random,
            draw_score,
            self._cfg.score_llm_threshold,
            force_reflect,
        )
        if draw_score < self._cfg.score_llm_threshold and not force_reflect:
            logger.info("[proactive] draw_score 未过门槛，跳过本轮反思")
            logger.info("[proactive] selected_action=idle reason=draw_score")
            return base_score
        if draw_score < self._cfg.score_llm_threshold and force_reflect:
            logger.info("[proactive] draw_score 未过门槛，但命中兜底条件，继续反思")

        is_crisis = energy < 0.05
        q_start, q_end, q_weight = self._sense.quiet_hours()
        in_quiet = (
            (now_hour >= q_start or now_hour < q_end)
            if q_start > q_end
            else (q_start <= now_hour < q_end)
        )
        sent_24h = (
            self._state.count_deliveries_in_window(session_key, 24, now=now_utc)
            if session_key
            else 0
        )
        replied_after_last_proactive = bool(
            target_last_user
            and last_proactive_at
            and target_last_user > last_proactive_at
        )
        mins_since_last_user = (
            int((now_utc - target_last_user).total_seconds() / 60)
            if target_last_user
            else None
        )
        mins_since_last_proactive = (
            int((now_utc - last_proactive_at).total_seconds() / 60)
            if last_proactive_at
            else None
        )
        fresh_items_24h = sum(
            1
            for item in new_items
            if item.published_at
            and (now_utc - item.published_at).total_seconds() <= 24 * 3600
        )
        decision_signals: dict[str, object] = {
            "quiet_window_local": f"{q_start:02d}:00-{q_end:02d}:00",
            "in_quiet_hours": in_quiet,
            "quiet_hours_weight": q_weight,
            "minutes_since_last_user": mins_since_last_user,
            "minutes_since_last_proactive": mins_since_last_proactive,
            "user_replied_after_last_proactive": replied_after_last_proactive,
            "proactive_sent_24h": sent_24h,
            "interruptibility": round(interruptibility, 3),
            "interrupt_breakdown": {
                "time": round(interrupt_detail["f_time"], 3),
                "reply": round(interrupt_detail["f_reply"], 3),
                "activity": round(interrupt_detail["f_activity"], 3),
                "fatigue": round(interrupt_detail["f_fatigue"], 3),
            },
            "scores": {
                "pre_score": round(pre_score, 3),
                "base_score": round(base_score, 3),
                "draw_score": round(draw_score, 3),
                "llm_threshold": round(self._cfg.score_llm_threshold, 3),
                "send_threshold": round(self._cfg.threshold, 3),
            },
            "candidate_items": len(new_items),
            "fresh_items_24h": fresh_items_24h,
        }
        feature_final_score: float | None = None
        feature_payload: dict[str, float | str] = {}
        if self._cfg.feature_scoring_enabled:
            features = await self._decide.score_features(
                items=new_items,  # 不回退到全量 items，尊重 interest_filter 过滤结果
                recent=recent,
                decision_signals=decision_signals,
            )
            feature_payload = features or {}
            feature_final_score = _feature_final_score(
                cfg=self._cfg,
                features=feature_payload,
                de=de,
                dc=dc,
                dr=dr,
                interruptibility=interruptibility,
            )
            logger.info(
                "[proactive] feature_score enabled final=%.3f threshold=%.3f features=%s",
                feature_final_score,
                self._cfg.feature_send_threshold,
                feature_payload,
            )
            if feature_final_score < self._cfg.feature_send_threshold:
                logger.info("[proactive] selected_action=idle reason=feature_score")
                # Fix C：feature_score 拒绝 = LLM 已评估，写拒绝冷却防止短期重入
                self._state.mark_rejection_cooldown(
                    new_entries,
                    hours=getattr(self._cfg, "llm_reject_cooldown_hours", 0),
                )
                return base_score
        decision = None
        decision_message = ""
        should_send = False
        if self._cfg.feature_scoring_enabled:
            decision_message = await self._decide.compose_message(
                items=new_items,  # 不回退，尊重 interest_filter 过滤结果
                recent=recent,
                decision_signals=decision_signals,
            )
            should_send = bool(decision_message.strip()) and (
                feature_final_score is not None
                and feature_final_score >= self._cfg.feature_send_threshold
            )
            logger.info(
                "[proactive] feature_mode compose_len=%d should_send=%s reasons={topic:%r,interest:%r,novel:%r,reconnect:%r,disturb:%r,readiness:%r,conf:%r}",
                len(decision_message),
                should_send,
                feature_payload.get("topic_continuity_reason", ""),
                feature_payload.get("interest_match_reason", ""),
                feature_payload.get("content_novelty_reason", ""),
                feature_payload.get("reconnect_value_reason", ""),
                feature_payload.get("disturb_risk_reason", ""),
                feature_payload.get("message_readiness_reason", ""),
                feature_payload.get("confidence_reason", ""),
            )
        else:
            decision = await self._decide.reflect(
                new_items,
                recent,
                energy=energy,
                urge=draw_score,
                is_crisis=is_crisis,
                decision_signals=decision_signals,
            )
            decision, decision_delta = self._decide.randomize_decision(decision)
            logger.info(
                f"[proactive] score={decision.score:.2f}  "
                f"score_delta={decision_delta:+.2f}  "
                f"send={decision.should_send}  "
                f"reasoning={decision.reasoning[:80]!r}"
            )
            should_send = decision.should_send and decision.score >= self._cfg.threshold
            decision_message = decision.message
        if should_send:
            channel = (self._cfg.default_channel or "").strip()
            chat_id = self._cfg.default_chat_id.strip()
            session_key = f"{channel}:{chat_id}" if channel and chat_id else ""
            evidence_ids = self._decide.resolve_evidence_item_ids(
                decision
                if decision is not None
                else _PseudoDecision(message=decision_message),
                new_items,  # 与 score_features/compose_message 保持一致，不回退全量
            )
            delivery_key = self._decide.build_delivery_key(
                evidence_ids, decision_message
            )
            logger.info(
                "[proactive] 发送前去重检查 session=%s evidence_count=%d delivery_key=%s",
                session_key or "（未配置）",
                len(evidence_ids),
                delivery_key[:16],
            )
            if session_key and self._state.is_delivery_duplicate(
                session_key=session_key,
                delivery_key=delivery_key,
                window_hours=self._cfg.delivery_dedupe_hours,
            ):
                logger.info("[proactive] 命中发送去重，跳过发送")
                self._state.mark_items_seen(new_entries)
                self._state.mark_semantic_items(
                    self._decide.semantic_entries(new_items)
                )
                logger.info(
                    "[proactive] 已按去重命中标记本轮条目为 seen（视为已送达过同等内容）"
                )
                logger.info("[proactive] selected_action=idle reason=delivery_dedupe")
                return base_score
            if self._message_deduper is not None:
                recent_proactive = self._sense.collect_recent_proactive(
                    getattr(self._cfg, "message_dedupe_recent_n", 5)
                )
                is_dup, dup_reason = await self._message_deduper.is_duplicate(
                    decision_message, recent_proactive
                )
                if is_dup:
                    logger.info(
                        "[proactive] 消息语义去重命中，跳过发送 reason=%r", dup_reason
                    )
                    logger.info(
                        "[proactive] selected_action=idle reason=message_dedupe"
                    )
                    # Fix B：message_dedupe 误判率较高，只写 semantic_items（72h 软抑制）
                    # 不写 seen_items，避免误判导致 14 天硬压制
                    self._state.mark_semantic_items(
                        self._decide.semantic_entries(new_items)
                    )
                    return base_score
            sent = await self._act.send(decision_message)
            if sent:
                # 配额记录：动作成功即消耗，与 session_key 无关
                if self._cfg.anyaction_enabled and self._anyaction:
                    self._anyaction.record_action(now_utc=now_utc)
                # item 标记：全局去重，不依赖 session
                self._state.mark_items_seen(new_entries)
                self._state.mark_semantic_items(
                    self._decide.semantic_entries(new_items)
                )
                # delivery 去重：仅 chat 类 action 有 session_key
                if session_key:
                    self._state.mark_delivery(session_key, delivery_key)
                logger.info("[proactive] 已发送成功并标记本轮条目为 seen")
                logger.info("[proactive] selected_action=chat")
            else:
                logger.info("[proactive] 本轮发送未成功，不标记 seen，后续可再次尝试")
                logger.info("[proactive] selected_action=idle reason=send_failed")
        else:
            logger.info("[proactive] 决定不主动发送")
            logger.info("[proactive] 本轮未发送，不标记 seen，后续可再次尝试")
            # Fix C：LLM 已评估并拒绝，写拒绝冷却防止短期内重复进 LLM
            self._state.mark_rejection_cooldown(
                new_entries, hours=getattr(self._cfg, "llm_reject_cooldown_hours", 0)
            )
            # Skill Action 分支：chat idle 时尝试执行注册的 skill action
            await self._try_skill_action(now_utc=now_utc)
        return base_score

    async def _try_skill_action(self, *, now_utc: datetime) -> None:
        """在 chat idle 时，尝试从注册的 skill actions 中随机抽取并执行一个。"""
        if not self._skill_action_runner:
            logger.info("[proactive] selected_action=idle reason=decision")
            return
        avail = self._skill_action_runner.available_count()
        logger.info("[proactive] skill_action 可用数量=%d", avail)
        action = self._skill_action_runner.pick()
        if action is None:
            logger.info(
                "[proactive] selected_action=idle reason=decision "
                "(无可用 skill action，配额已满或间隔未到)"
            )
            return
        logger.info(
            "[proactive] selected_action=skill_action id=%s name=%r",
            action.id,
            action.name,
        )
        # 消耗 anyaction 配额（与 chat 共享）
        if self._cfg.anyaction_enabled and self._anyaction:
            self._anyaction.record_action(now_utc=now_utc)
        success = await self._skill_action_runner.run(action)
        logger.info(
            "[proactive] skill_action 完成 id=%s success=%s",
            action.id,
            success,
        )


def _feature_final_score(
    *,
    cfg: Any,
    features: dict[str, float],
    de: float,
    dc: float,
    dr: float,
    interruptibility: float,
) -> float:
    def f(name: str, default: float = 0.5) -> float:
        try:
            val = float(features.get(name, default))
        except Exception:
            val = default
        return max(0.0, min(1.0, val))

    topic_continuity = f("topic_continuity")
    interest_match = f("interest_match")
    content_novelty = f("content_novelty")
    reconnect_value = f("reconnect_value")
    disturb_risk = f("disturb_risk")
    message_readiness = f("message_readiness")
    confidence = f("confidence")

    utility = (
        cfg.feature_weight_topic_continuity * topic_continuity
        + cfg.feature_weight_interest_match * interest_match
        + cfg.feature_weight_content_novelty * content_novelty
        + cfg.feature_weight_reconnect_value * reconnect_value
        + cfg.feature_weight_message_readiness * message_readiness
    )
    risk = (
        cfg.feature_weight_disturb_risk * disturb_risk
        + cfg.feature_weight_interrupt_penalty
        * (1.0 - max(0.0, min(1.0, interruptibility)))
    )
    system_bonus = (
        cfg.feature_weight_d_recent_bonus * dr
        + cfg.feature_weight_d_content_bonus * dc
        + cfg.feature_weight_d_energy_bonus * de
    )
    raw = utility - risk + system_bonus
    conf_adjusted = raw * (0.7 + 0.3 * confidence)
    return max(0.0, min(1.0, conf_adjusted))
