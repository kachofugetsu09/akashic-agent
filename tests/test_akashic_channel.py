from types import SimpleNamespace
from pathlib import Path
from typing import Any, cast

import pytest

from agent.plugin_composition.channels import (
    ChannelDeliveryReceipt,
    ChannelFactoryContext,
    ChannelReady,
    DeliveryStatus,
    ProviderDeliveryReceipt,
    ProviderDeliveryRequest,
    StopReceipt,
)
from agent.control.scoped_turn import TurnAcceptedReceipt
from agent.plugin_composition.durable_deliveries import (
    DurableBindingAttempt,
    DurableDeliveryRequest,
    PluginDurableDeliveries,
)
from agent.plugin_composition.durable_delivery_store import DurableDeliveryStore
from infra.channels.akashic_channel import AkashicChannel
from session.manager import SessionManager


class _Adapter:
    def __init__(
        self,
        binding_token: str,
        status: DeliveryStatus,
        *,
        error: BaseException | None = None,
        provider_ids: tuple[str, ...] | None = None,
    ) -> None:
        self.binding_token = binding_token
        self.status = status
        self.error = error
        self.provider_ids = provider_ids
        self.requests: list[ProviderDeliveryRequest] = []
        self.runtime: object | None = None
        self.admission_open = False

    async def start(self) -> ChannelReady:
        return ChannelReady(self.binding_token)

    def attach_runtime(self, ports: object) -> None:
        self.runtime = ports

    def open_admission(self) -> None:
        self.admission_open = True

    def close_admission(self) -> None:
        self.admission_open = False

    async def deliver(
        self,
        request: ProviderDeliveryRequest,
    ) -> ProviderDeliveryReceipt:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return ProviderDeliveryReceipt(
            request.delivery_id,
            self.status,
            provider_ids=self.provider_ids or (f"{self.status.value}-id",),
            error=(
                "unavailable" if self.status is not DeliveryStatus.DELIVERED else None
            ),
        )

    async def stop(self) -> StopReceipt:
        return StopReceipt(self.binding_token, resources_closed=True)


class _Child:
    name = "transport-detail"

    def __init__(
        self,
        status: DeliveryStatus,
        *,
        error: BaseException | None = None,
        provider_ids: tuple[str, ...] | None = None,
    ) -> None:
        self.status = status
        self.error = error
        self.provider_ids = provider_ids
        self.adapter: _Adapter | None = None

    def build_v3_adapter(self, context: ChannelFactoryContext) -> _Adapter:
        self.adapter = _Adapter(
            context.binding_token,
            self.status,
            error=self.error,
            provider_ids=self.provider_ids,
        )
        return self.adapter


class _FailingLifecycleAdapter(_Adapter):
    def __init__(
        self,
        binding_token: str,
        *,
        fail_start: bool = False,
        fail_stop: bool = False,
        fail_open: bool = False,
        fail_close: bool = False,
    ) -> None:
        super().__init__(binding_token, DeliveryStatus.DELIVERED)
        self.fail_start = fail_start
        self.fail_stop = fail_stop
        self.fail_open = fail_open
        self.fail_close = fail_close
        self.close_calls = 0

    async def start(self) -> ChannelReady:
        if self.fail_start:
            raise RuntimeError("start failed")
        return await super().start()

    async def stop(self) -> StopReceipt:
        if self.fail_stop:
            raise RuntimeError("stop failed")
        return await super().stop()

    def open_admission(self) -> None:
        if self.fail_open:
            raise RuntimeError("open failed")
        super().open_admission()

    def close_admission(self) -> None:
        self.close_calls += 1
        if self.fail_close:
            raise RuntimeError("close failed")
        super().close_admission()


class _LifecycleChild:
    name = "transport-detail"

    def __init__(self, adapter: _FailingLifecycleAdapter) -> None:
        self.adapter = adapter

    def build_v3_adapter(
        self, context: ChannelFactoryContext
    ) -> _FailingLifecycleAdapter:
        return self.adapter


def _context() -> ChannelFactoryContext:
    return ChannelFactoryContext(
        snapshot_id="snapshot",
        generation_id="generation",
        binding_token="binding",
        config={},
        credentials={},
        provider_client_factory=cast(Any, SimpleNamespace()),
        ingress=None,
        identity=None,
    )


def _request() -> ProviderDeliveryRequest:
    return ProviderDeliveryRequest(
        binding_token="binding",
        delivery_id="delivery",
        recipient="chat-id",
        body="hello",
    )


@pytest.mark.asyncio
async def test_projects_one_core_delivery_to_web_and_mobile() -> None:
    web = _Child(DeliveryStatus.DELIVERED)
    mobile = _Child(DeliveryStatus.REJECTED)
    channel = AkashicChannel(cast(Any, web), cast(Any, mobile))
    adapter = channel.build_v3_adapter(_context())

    receipt = await adapter.deliver(_request())

    assert channel.name == "akashic"
    assert receipt.status is DeliveryStatus.DELIVERED
    assert web.adapter is not None and len(web.adapter.requests) == 1
    assert mobile.adapter is not None and len(mobile.adapter.requests) == 1
    assert web.adapter.requests[0] is mobile.adapter.requests[0]


@pytest.mark.asyncio
async def test_preserves_unknown_when_another_client_delivered() -> None:
    web = _Child(DeliveryStatus.DELIVERED)
    mobile = _Child(DeliveryStatus.UNKNOWN, error=RuntimeError("offline"))
    adapter = AkashicChannel(cast(Any, web), cast(Any, mobile)).build_v3_adapter(
        _context()
    )

    receipt = await adapter.deliver(_request())

    assert receipt.status is DeliveryStatus.UNKNOWN
    assert "offline" in str(receipt.error)


@pytest.mark.asyncio
async def test_provider_ids_are_unique_in_first_seen_order() -> None:
    web = _Child(
        DeliveryStatus.DELIVERED,
        provider_ids=("shared", "web"),
    )
    mobile = _Child(
        DeliveryStatus.DELIVERED,
        provider_ids=("shared", "mobile", "web"),
    )
    adapter = AkashicChannel(cast(Any, web), cast(Any, mobile)).build_v3_adapter(
        _context()
    )

    receipt = await adapter.deliver(_request())

    assert receipt.provider_ids == ("shared", "web", "mobile")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("web_status", "mobile_status", "expected_state", "expected_messages"),
    (
        (DeliveryStatus.DELIVERED, DeliveryStatus.DELIVERED, "projected", 1),
        (DeliveryStatus.DELIVERED, DeliveryStatus.REJECTED, "projected", 1),
        (DeliveryStatus.DELIVERED, DeliveryStatus.UNKNOWN, "uncertain", 0),
    ),
)
async def test_durable_projection_follows_composite_delivery_matrix(
    tmp_path: Path,
    web_status: DeliveryStatus,
    mobile_status: DeliveryStatus,
    expected_state: str,
    expected_messages: int,
) -> None:
    """Unknown blocks projection; one delivered UI is enough after rejection."""

    web = _Child(web_status)
    mobile = _Child(mobile_status)
    adapter = AkashicChannel(cast(Any, web), cast(Any, mobile)).build_v3_adapter(
        _context()
    )
    sessions = SessionManager(tmp_path / "workspace")
    store = DurableDeliveryStore(tmp_path / "settlements.sqlite")
    provider_calls = 0

    async def sender(request: DurableDeliveryRequest, started: Any) -> Any:
        nonlocal provider_calls
        provider_calls += 1
        started(DurableBindingAttempt("attempt", "snapshot", "generation", "binding"))
        receipt = await adapter.deliver(
            ProviderDeliveryRequest(
                binding_token="binding",
                delivery_id=request.logical_delivery_id,
                recipient=request.recipient,
                body=request.body,
            )
        )
        return ChannelDeliveryReceipt(
            receipt.delivery_id,
            receipt.status,
            receipt.provider_ids,
            receipt.error,
        )

    async def project(request: DurableDeliveryRequest) -> str:
        return await sessions.append_durable_delivery(
            session_key=request.projection_session_id,
            content=request.body,
            delivery_id=request.logical_delivery_id,
            control_turn_id=request.accepted_turn.turn_id,
        )

    request = DurableDeliveryRequest(
        logical_delivery_id="schedule:matrix",
        accepted_turn=TurnAcceptedReceipt("scheduler:job", "turn:matrix"),
        target_service="scheduler.delivery.v1",
        channel="akashic",
        recipient="a" * 32,
        projection_session_id="akashic:" + "a" * 32,
        body="scheduled result",
    )
    service = PluginDurableDeliveries(store, sender, project)

    result = await service.submit(request)
    duplicate = await service.submit(request)

    assert result.state == duplicate.state == expected_state
    assert provider_calls == 1
    assert (
        len(
            sessions.control_store.fetch_session_messages(request.projection_session_id)
        )
        == expected_messages
    )
    assert web.adapter is not None and len(web.adapter.requests) == 1
    assert mobile.adapter is not None and len(mobile.adapter.requests) == 1
    sessions.close()


@pytest.mark.asyncio
async def test_delegates_one_binding_lifecycle_to_both_adapters() -> None:
    web = _Child(DeliveryStatus.DELIVERED)
    mobile = _Child(DeliveryStatus.DELIVERED)
    adapter = AkashicChannel(cast(Any, web), cast(Any, mobile)).build_v3_adapter(
        _context()
    )
    runtime = SimpleNamespace()

    ready = await adapter.start()
    adapter.attach_runtime(cast(Any, runtime))
    adapter.open_admission()
    adapter.close_admission()
    stopped = await adapter.stop()

    assert ready.binding_token == "binding"
    assert stopped.resources_closed is True
    assert web.adapter is not None and web.adapter.runtime is runtime
    assert mobile.adapter is not None and mobile.adapter.runtime is runtime
    assert web.adapter.admission_open is False
    assert mobile.adapter.admission_open is False


@pytest.mark.asyncio
async def test_start_rollback_preserves_primary_and_stop_failure() -> None:
    first = _FailingLifecycleAdapter("binding", fail_stop=True)
    second = _FailingLifecycleAdapter("binding", fail_start=True)
    adapter = AkashicChannel(
        cast(Any, _LifecycleChild(first)),
        cast(Any, _LifecycleChild(second)),
    ).build_v3_adapter(_context())

    with pytest.raises(BaseExceptionGroup) as raised:
        await adapter.start()

    assert {str(error) for error in raised.value.exceptions} == {
        "start failed",
        "stop failed",
    }


def test_admission_rollback_closes_every_open_child_and_preserves_failures() -> None:
    first = _FailingLifecycleAdapter("binding", fail_close=True)
    second = _FailingLifecycleAdapter("binding", fail_open=True)
    adapter = AkashicChannel(
        cast(Any, _LifecycleChild(first)),
        cast(Any, _LifecycleChild(second)),
    ).build_v3_adapter(_context())

    with pytest.raises(BaseExceptionGroup) as raised:
        adapter.open_admission()

    assert first.close_calls == 1
    assert {str(error) for error in raised.value.exceptions} == {
        "open failed",
        "close failed",
    }
