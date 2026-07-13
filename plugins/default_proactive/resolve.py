from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Awaitable, Callable, Protocol

from agent.turns.result import TurnOutbound, TurnResult, TurnTrace
from core.common.diagnostic_log import diagnostic_line
from proactive_v2.config import ProactiveConfig
from proactive_v2.sensor import RecentProactiveMessage
from plugins.default_proactive.context import AgentTickContext

logger = logging.getLogger(__name__)


class MessageDeduper(Protocol):
    async def is_duplicate(
        self,
        new_message: str,
        recent_proactive: list[RecentProactiveMessage],
        new_state_summary_tag: str = "none",
    ) -> tuple[bool, str]: ...

@dataclass
class ResolveResult:
    action: str
    result: TurnResult


@dataclass
class CallbackSideEffect:
    callback: Callable[[], Awaitable[None]]
    name: str = "callback"

    async def run(self) -> None:
        await self.callback()


def _normalize_delivery_url(raw: str) -> str:
    from urllib.parse import urlsplit, urlunsplit

    text = str(raw or "").strip()
    if not text:
        return ""
    parts = urlsplit(text)
    path = parts.path.rstrip("/") or parts.path
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def _build_delivery_refs(ctx: AgentTickContext) -> list[str]:
    if not ctx.cited_item_ids:
        return []
    content_map = {
        f"{e.get('ack_server', '')}:{e.get('event_id') or e.get('id', '')}": e
        for e in ctx.fetched_contents
        if e.get("ack_server") and (e.get("event_id") or e.get("id"))
    }
    refs: list[str] = []
    for key in sorted(set(ctx.cited_item_ids)):
        meta = content_map.get(key)
        if meta is None:
            refs.append(f"id:{key}")
            continue
        url = _normalize_delivery_url(str(meta.get("url") or ""))
        if url:
            refs.append(f"url:{url}")
            continue
        source = str(meta.get("source") or "").strip().lower()
        title = str(meta.get("title") or "").strip().lower()
        if title:
            refs.append(f"title:{source}|{title}")
            continue
        refs.append(f"id:{key}")
    return sorted(set(refs))


def build_delivery_key(ctx: AgentTickContext) -> str:
    refs = _build_delivery_refs(ctx)
    if refs and any(not ref.startswith("id:") for ref in refs):
        key_src = json.dumps(refs)
    elif ctx.cited_item_ids:
        key_src = json.dumps(sorted(ctx.cited_item_ids))
    else:
        key_src = ctx.final_message[:500]
    return sha1(key_src.encode()).hexdigest()[:16]


async def ack_discarded(ctx: AgentTickContext, ack_fn: Any) -> None:
    if ack_fn is None:
        return
    for key in ctx.discarded_item_ids:
        await ack_fn(key, "not_interesting")


async def ack_post_guard_fail(
    ctx: AgentTickContext,
    ack_fn: Any,
    *,
    alert_ack_fn: Any = None,
) -> None:
    if ack_fn is None and alert_ack_fn is None:
        return
    fetched_alert_keys = {
        f"{e['ack_server']}:{e.get('event_id') or e.get('id', '')}"
        for e in ctx.fetched_alerts
    }
    cited_set = set(ctx.cited_item_ids)

    async def _ack_alert(key: str) -> None:
        if alert_ack_fn is not None:
            await alert_ack_fn(key)
        elif ack_fn is not None:
            await ack_fn(key, "interesting")

    for key in cited_set - fetched_alert_keys:
        if ack_fn is not None:
            await ack_fn(key, "interesting")
    for key in cited_set & fetched_alert_keys:
        await _ack_alert(key)
    for key in fetched_alert_keys - cited_set:
        await _ack_alert(key)
    for key in (ctx.interesting_item_ids - cited_set) - fetched_alert_keys:
        if ack_fn is not None:
            await ack_fn(key, "interesting")
    for key in ctx.discarded_item_ids:
        if ack_fn is not None:
            await ack_fn(key, "not_interesting")


async def ack_on_success(
    ctx: AgentTickContext,
    ack_fn: Any,
    *,
    alert_ack_fn: Any = None,
) -> None:
    if ack_fn is None and alert_ack_fn is None:
        return
    fetched_alert_keys = {
        f"{e['ack_server']}:{e.get('event_id') or e.get('id', '')}"
        for e in ctx.fetched_alerts
    }
    fetched_content_keys = {
        f"{e['ack_server']}:{e.get('event_id') or e.get('id', '')}"
        for e in ctx.fetched_contents
    }
    cited_set = set(ctx.cited_item_ids)
    for key in cited_set & fetched_content_keys:
        if ack_fn is not None:
            await ack_fn(key, "interesting")
    for key in cited_set & fetched_alert_keys:
        if alert_ack_fn is not None:
            await alert_ack_fn(key)
        else:
            await ack_fn(key, "interesting")
    for key in (ctx.interesting_item_ids - cited_set) - fetched_alert_keys:
        if ack_fn is not None:
            await ack_fn(key, "interesting")
    for key in ctx.discarded_item_ids:
        if ack_fn is not None:
            await ack_fn(key, "not_interesting")


async def _mark_delivery(
    *,
    state_store: Any,
    session_key: str,
    delivery_key: str,
) -> None:
    state_store.mark_delivery(session_key, delivery_key)


async def _mark_context_only_send(
    *,
    state_store: Any,
    session_key: str,
    context_as_fallback_open: bool,
    has_cited: bool,
) -> None:
    if context_as_fallback_open and not has_cited:
        state_store.mark_context_only_send(session_key)


class ProactiveResolver:
    def __init__(
        self,
        *,
        cfg: ProactiveConfig,
        session_key: str,
        state_store: Any,
        deduper: MessageDeduper,
        recent_proactive_fn: Callable[[], list[RecentProactiveMessage]] | None,
        ack_fn: Any,
        alert_ack_fn: Any,
    ) -> None:
        self._cfg = cfg
        self._session_key = session_key
        self._state_store = state_store
        self._deduper = deduper
        self._recent_proactive_fn = recent_proactive_fn
        self._ack_fn = ack_fn
        self._alert_ack_fn = alert_ack_fn

    async def resolve(self, ctx: AgentTickContext) -> ResolveResult:
        if ctx.terminal_action != "reply":
            return self._build_skip_result(ctx)

        delivery_key = build_delivery_key(ctx)
        if self._state_store.is_delivery_duplicate(
            self._session_key,
            delivery_key,
            self._cfg.delivery_dedupe_hours,
        ):
            return self._build_delivery_dedupe_result(ctx)

        if self._cfg.message_dedupe_enabled and self._deduper is not None:
            recent_proactive = (
                self._recent_proactive_fn()
                if self._recent_proactive_fn is not None
                else []
            )
            is_dup, reason = await self._deduper.is_duplicate(
                new_message=ctx.final_message,
                recent_proactive=recent_proactive,
                new_state_summary_tag="none",
            )
            if is_dup:
                return self._build_message_dedupe_result(ctx, reason)

        return self._build_send_result(ctx, delivery_key)

    def _build_skip_result(self, ctx: AgentTickContext) -> ResolveResult:
        logger.info(
            diagnostic_line(
                "ProactiveResolver.resolve",
                event="resolve",
                flow="proactive",
                phase="resolve",
                session=self._session_key,
                tick=ctx.tick_id,
                action="skip",
                reason=ctx.skip_reason or "no_content",
                counts=self._counts(ctx),
                note=ctx.skip_note or "-",
            )
        )
        logger.info(
            "[proactive_v2] resolve: action=%s steps=%d discarded=%d interesting=%d skip_reason=%s note=%s",
            ctx.terminal_action or "none",
            ctx.steps_taken,
            len(ctx.discarded_item_ids),
            len(ctx.interesting_item_ids),
            ctx.skip_reason,
            ctx.skip_note,
        )
        return ResolveResult(
            action="skip",
            result=TurnResult(
                decision="skip",
                outbound=None,
                trace=TurnTrace(
                    source="proactive",
                    extra={
                        "steps_taken": ctx.steps_taken,
                        "skip_reason": ctx.skip_reason,
                        "skip_note": ctx.skip_note,
                    },
                ),
                side_effects=[
                    CallbackSideEffect(
                        callback=lambda: ack_discarded(ctx, self._ack_fn),
                        name="ack_discarded_skip",
                    )
                ],
            ),
        )

    def _build_delivery_dedupe_result(self, ctx: AgentTickContext) -> ResolveResult:
        logger.info(
            diagnostic_line(
                "ProactiveResolver.resolve",
                event="resolve",
                flow="proactive",
                phase="resolve",
                session=self._session_key,
                tick=ctx.tick_id,
                action="skip",
                reason="already_sent_similar",
                counts=self._counts(ctx),
                note="delivery_dedupe",
            )
        )
        logger.info("[proactive_v2] resolve: delivery_dedupe hit")
        return ResolveResult(
            action="skip",
            result=TurnResult(
                decision="skip",
                outbound=None,
                evidence=list(ctx.cited_item_ids),
                trace=TurnTrace(
                    source="proactive",
                    extra={
                        "steps_taken": ctx.steps_taken,
                        "skip_reason": "already_sent_similar",
                        "dedupe": "delivery",
                    },
                ),
                side_effects=[
                    CallbackSideEffect(
                        callback=lambda: ack_post_guard_fail(
                            ctx,
                            self._ack_fn,
                            alert_ack_fn=self._alert_ack_fn,
                        ),
                        name="ack_post_guard_delivery",
                    )
                ],
            ),
        )

    def _build_message_dedupe_result(
        self,
        ctx: AgentTickContext,
        reason: str,
    ) -> ResolveResult:
        logger.info(
            diagnostic_line(
                "ProactiveResolver.resolve",
                event="resolve",
                flow="proactive",
                phase="resolve",
                session=self._session_key,
                tick=ctx.tick_id,
                action="skip",
                reason="already_sent_similar",
                counts=self._counts(ctx),
                note=reason[:160],
            )
        )
        logger.info("[proactive_v2] resolve: message_dedupe hit: %s", reason)
        return ResolveResult(
            action="skip",
            result=TurnResult(
                decision="skip",
                outbound=None,
                evidence=list(ctx.cited_item_ids),
                trace=TurnTrace(
                    source="proactive",
                    extra={
                        "steps_taken": ctx.steps_taken,
                        "skip_reason": "already_sent_similar",
                        "dedupe": "message",
                        "dedupe_note": reason,
                    },
                ),
                side_effects=[
                    CallbackSideEffect(
                        callback=lambda: ack_post_guard_fail(
                            ctx,
                            self._ack_fn,
                            alert_ack_fn=self._alert_ack_fn,
                        ),
                        name="ack_post_guard_message",
                    )
                ],
            ),
        )

    def _build_send_result(
        self,
        ctx: AgentTickContext,
        delivery_key: str,
    ) -> ResolveResult:
        logger.info(
            diagnostic_line(
                "ProactiveResolver.resolve",
                event="resolve",
                flow="proactive",
                phase="resolve",
                session=self._session_key,
                tick=ctx.tick_id,
                action="send",
                reason="-",
                counts=f"{self._counts(ctx)},cited:{len(ctx.cited_item_ids)}",
            )
        )
        return ResolveResult(
            action="send",
            result=TurnResult(
                decision="reply",
                outbound=TurnOutbound(
                    session_key=self._session_key,
                    content=ctx.final_message,
                ),
                evidence=list(ctx.cited_item_ids),
                trace=TurnTrace(
                    source="proactive",
                    extra={
                        "steps_taken": ctx.steps_taken,
                        "skip_reason": "",
                        "state_summary_tag": "none",
                    },
                ),
                success_side_effects=[
                    CallbackSideEffect(
                        callback=lambda: _mark_delivery(
                            state_store=self._state_store,
                            session_key=self._session_key,
                            delivery_key=delivery_key,
                        ),
                        name="mark_delivery",
                    ),
                    CallbackSideEffect(
                        callback=lambda: _mark_context_only_send(
                            state_store=self._state_store,
                            session_key=self._session_key,
                            context_as_fallback_open=ctx.context_as_fallback_open,
                            has_cited=bool(ctx.cited_item_ids),
                        ),
                        name="mark_context_only_send",
                    ),
                    CallbackSideEffect(
                        callback=lambda: ack_on_success(
                            ctx,
                            self._ack_fn,
                            alert_ack_fn=self._alert_ack_fn,
                        ),
                        name="ack_on_success",
                    ),
                ],
                failure_side_effects=[
                    CallbackSideEffect(
                        callback=lambda: ack_discarded(ctx, self._ack_fn),
                        name="ack_discarded_send_fail",
                    )
                ],
            ),
        )

    @staticmethod
    def _counts(ctx: AgentTickContext) -> str:
        return (
            f"steps:{ctx.steps_taken},"
            f"interesting:{len(ctx.interesting_item_ids)},"
            f"discarded:{len(ctx.discarded_item_ids)}"
        )
