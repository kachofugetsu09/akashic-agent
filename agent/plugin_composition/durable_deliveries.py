from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

from agent.control.scoped_turn import TurnAcceptedReceipt
from agent.plugin_composition.channels import ChannelDeliveryReceipt, DeliveryStatus
from agent.plugin_composition.durable_delivery_store import DurableDeliveryStore
from agent.plugin_composition.model import ServiceKey
from session.store import validate_message_delivery_id


def _empty_metadata() -> dict[str, object]:
    return {}


@dataclass(frozen=True, slots=True)
class DurableDeliveryRequest:
    """Describe one immutable source-neutral delivery and Session projection."""

    logical_delivery_id: str
    accepted_turn: TurnAcceptedReceipt
    target_service: str
    channel: str
    recipient: str
    projection_session_id: str
    body: str
    metadata: Mapping[str, object] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _ = validate_message_delivery_id(self.logical_delivery_id)
        for name in (
            "target_service",
            "channel",
            "recipient",
            "projection_session_id",
        ):
            value = getattr(self, name)
            if not value or value.strip() != value:
                raise ValueError(f"{name} 必须非空且无首尾空白")
        if not isinstance(self.body, str) or not self.body:
            raise ValueError("body 必须是非空字符串")
        if not isinstance(self.accepted_turn, TurnAcceptedReceipt):
            raise TypeError("accepted_turn 必须是 TurnAcceptedReceipt")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class DurableBindingAttempt:
    """Freeze the exact Channel binding that is about to perform provider I/O."""

    attempt_id: str
    snapshot_id: str
    generation_id: str
    binding_token: str


@dataclass(frozen=True, slots=True)
class DurableDeliveryView:
    """Expose one immutable envelope with its current forward-only state."""

    logical_delivery_id: str
    accepted_turn: TurnAcceptedReceipt
    target_service: str
    channel: str
    recipient: str
    projection_session_id: str
    body: str
    metadata: Mapping[str, object]
    state: Literal[
        "prepared",
        "provider_started",
        "delivered",
        "projected",
        "settled",
        "rejected",
        "uncertain",
    ]
    attempt_id: str | None
    snapshot_id: str | None
    generation_id: str | None
    binding_token: str | None
    provider_receipt: Mapping[str, object] | None
    projection_message_id: str | None
    domain_receipt: str | None


ProviderStarted = Callable[[DurableBindingAttempt], None]
DurableSender = Callable[
    [DurableDeliveryRequest, ProviderStarted], Awaitable[ChannelDeliveryReceipt]
]
DurableProjector = Callable[[DurableDeliveryRequest], Awaitable[str]]


class PluginDurableDeliveries:
    """Own provider receipt, Session projection, and durable settlement ordering."""

    def __init__(
        self,
        store: DurableDeliveryStore | None,
        sender: DurableSender | None,
        projector: DurableProjector | None,
        *,
        recover_started: bool = True,
    ) -> None:
        self._store = store
        self._sender = sender
        self._projector = projector
        self._lock = asyncio.Lock()
        if store is not None:
            store.initialize()
            if recover_started:
                _ = store.recover_interrupted_provider_calls()

    @classmethod
    def candidate_validation(cls) -> PluginDurableDeliveries:
        return cls(None, None, None, recover_started=False)

    @property
    def formal(self) -> bool:
        return self._store is not None

    async def submit(self, request: DurableDeliveryRequest) -> DurableDeliveryView:
        """Prepare once, then advance provider and Session effects in order."""

        return await _complete_critical(self._submit_owned(request))

    async def _submit_owned(
        self, request: DurableDeliveryRequest
    ) -> DurableDeliveryView:
        """Complete provider receipt and projection inside one Core-owned task."""

        store, sender, projector = self._require_formal()
        async with self._lock:
            row = store.prepare(_envelope(request))
            state = str(row["state"])
            if state == "prepared":
                row = await self._send(store, sender, request)
                state = str(row["state"])
            if state == "provider_started" and request.channel == "akashic":
                message_id = await projector(request)
                row = store.mark_akashic_session_recovered(
                    request.logical_delivery_id, message_id
                )
                state = str(row["state"])
            if state == "delivered":
                message_id = await projector(request)
                row = store.mark_projected(request.logical_delivery_id, message_id)
            return _view(row)

    def lookup(self, accepted_turn: TurnAcceptedReceipt) -> DurableDeliveryView | None:
        """Read the unique logical delivery associated with an accepted Turn."""

        store = self._require_store()
        row = store.lookup(accepted_turn.session_id, accepted_turn.turn_id)
        return None if row is None else _view(row)

    def recoverable(self) -> tuple[DurableDeliveryView, ...]:
        """Read rows Core can complete without repeating an external effect."""

        store = self._require_store()
        return tuple(_view(row) for row in store.recoverable())

    async def resume(self, accepted_turn: TurnAcceptedReceipt) -> DurableDeliveryView:
        """Advance one existing prepared or delivered row without changing identity."""

        current = self.lookup(accepted_turn)
        if current is None:
            raise KeyError(
                f"durable delivery missing: {accepted_turn.session_id}/{accepted_turn.turn_id}"
            )
        return await self.submit(_request(current))

    def cancel_prepared(
        self, accepted_turn: TurnAcceptedReceipt, *, reason: str
    ) -> DurableDeliveryView:
        """Persist a no-send result while provider I/O is still impossible."""

        current = self.lookup(accepted_turn)
        if current is None:
            raise KeyError(
                f"durable delivery missing: {accepted_turn.session_id}/{accepted_turn.turn_id}"
            )
        store = self._require_store()
        return _view(store.cancel_prepared(current.logical_delivery_id, reason=reason))

    def confirm_settled(
        self, settlement_ref: str, domain_receipt: str
    ) -> DurableDeliveryView:
        """Persist one opaque domain receipt after projection."""

        store = self._require_store()
        return _view(store.confirm_settled(settlement_ref, domain_receipt))

    def _require_store(self) -> DurableDeliveryStore:
        if self._store is None:
            raise RuntimeError("candidate 验证期禁止 durable delivery")
        return self._store

    async def _send(
        self,
        store: DurableDeliveryStore,
        sender: DurableSender,
        request: DurableDeliveryRequest,
    ) -> dict[str, object]:
        """Persist provider_started through the sender callback before external I/O."""

        started = False

        def mark_started(attempt: DurableBindingAttempt) -> None:
            nonlocal started
            _ = store.mark_provider_started(
                request.logical_delivery_id,
                attempt_id=attempt.attempt_id,
                snapshot_id=attempt.snapshot_id,
                generation_id=attempt.generation_id,
                binding_token=attempt.binding_token,
            )
            started = True

        try:
            receipt = await sender(request, mark_started)
        except BaseException:
            if started:
                _ = store.mark_provider_result(
                    request.logical_delivery_id,
                    state="uncertain",
                    receipt={"status": "unknown", "error": "provider call interrupted"},
                )
            raise
        if not started:
            raise RuntimeError("durable sender 未在 provider I/O 前提交 binding attempt")
        state = {
            DeliveryStatus.DELIVERED: "delivered",
            DeliveryStatus.REJECTED: "rejected",
            DeliveryStatus.UNKNOWN: "uncertain",
        }[receipt.status]
        return store.mark_provider_result(
            request.logical_delivery_id,
            state=state,
            receipt={
                "delivery_id": receipt.delivery_id,
                "status": receipt.status.value,
                "provider_ids": list(receipt.provider_ids),
                "error": receipt.error,
            },
        )

    def _require_formal(
        self,
    ) -> tuple[DurableDeliveryStore, DurableSender, DurableProjector]:
        if self._store is None or self._sender is None or self._projector is None:
            raise RuntimeError("durable delivery provider/Session boundary 尚未绑定")
        return self._store, self._sender, self._projector


def _envelope(request: DurableDeliveryRequest) -> dict[str, object]:
    return {
        "logical_delivery_id": request.logical_delivery_id,
        "accepted_session_id": request.accepted_turn.session_id,
        "accepted_turn_id": request.accepted_turn.turn_id,
        "target_service": request.target_service,
        "channel": request.channel,
        "recipient": request.recipient,
        "projection_session_id": request.projection_session_id,
        "body": request.body,
        "metadata": dict(request.metadata),
    }


def _view(row: Mapping[str, object]) -> DurableDeliveryView:
    metadata = _mapping(row["metadata"], "metadata")
    provider_raw = row["provider_receipt"]
    provider_receipt = (
        None
        if provider_raw is None
        else MappingProxyType(dict(_mapping(provider_raw, "provider_receipt")))
    )
    return DurableDeliveryView(
        logical_delivery_id=_text(row["logical_delivery_id"], "logical_delivery_id"),
        accepted_turn=TurnAcceptedReceipt(
            _text(row["accepted_session_id"], "accepted_session_id"),
            _text(row["accepted_turn_id"], "accepted_turn_id"),
        ),
        target_service=_text(row["target_service"], "target_service"),
        channel=_text(row["channel"], "channel"),
        recipient=_text(row["recipient"], "recipient"),
        projection_session_id=_text(
            row["projection_session_id"], "projection_session_id"
        ),
        body=_text(row["body"], "body"),
        metadata=MappingProxyType(dict(metadata)),
        state=_state(row["state"]),
        attempt_id=_optional_text(row["attempt_id"], "attempt_id"),
        snapshot_id=_optional_text(row["snapshot_id"], "snapshot_id"),
        generation_id=_optional_text(row["generation_id"], "generation_id"),
        binding_token=_optional_text(row["binding_token"], "binding_token"),
        provider_receipt=provider_receipt,
        projection_message_id=_optional_text(
            row["projection_message_id"], "projection_message_id"
        ),
        domain_receipt=_optional_text(row["domain_receipt"], "domain_receipt"),
    )


def _request(view: DurableDeliveryView) -> DurableDeliveryRequest:
    return DurableDeliveryRequest(
        logical_delivery_id=view.logical_delivery_id,
        accepted_turn=view.accepted_turn,
        target_service=view.target_service,
        channel=view.channel,
        recipient=view.recipient,
        projection_session_id=view.projection_session_id,
        body=view.body,
        metadata=view.metadata,
    )


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"durable delivery row {field} invalid")
    return value


def _optional_text(value: object, field: str) -> str | None:
    return None if value is None else _text(value, field)


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise RuntimeError(f"durable delivery row {field} invalid")
    return cast(Mapping[str, object], value)


def _state(value: object) -> Literal[
    "prepared",
    "provider_started",
    "delivered",
    "projected",
    "settled",
    "rejected",
    "uncertain",
]:
    if value not in {
        "prepared",
        "provider_started",
        "delivered",
        "projected",
        "settled",
        "rejected",
        "uncertain",
    }:
        raise RuntimeError(f"durable delivery row state invalid: {value!r}")
    return cast(
        Literal[
            "prepared",
            "provider_started",
            "delivered",
            "projected",
            "settled",
            "rejected",
            "uncertain",
        ],
        value,
    )


DURABLE_DELIVERIES = ServiceKey[PluginDurableDeliveries]("core.durable_deliveries")


async def _complete_critical(
    awaitable: Awaitable[DurableDeliveryView],
) -> DurableDeliveryView:
    """Finish durable forward progress before restoring caller cancellation."""

    task = asyncio.ensure_future(awaitable)
    cancelled = False
    while not task.done():
        try:
            _ = await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result
