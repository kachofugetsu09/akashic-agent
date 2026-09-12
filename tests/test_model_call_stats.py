from dataclasses import asdict
from collections.abc import Mapping
from typing import cast

import httpx
import pytest
from fastapi import FastAPI

from agent.plugin_composition import CompositionRoot
from agent.plugin_composition.models import (
    MODEL_CALL_STATS,
    ChatModelSelection,
    ModelRequest,
)
from agent.plugin_composition.rpc import rpc_method_key
from agent.plugin_composition.model_settings_http import ModelControlUnavailable
from agent.plugins.model_control import RuntimeModelControl
from agent.plugins.snapshot import RuntimeSnapshotCompiler, RuntimeSnapshotStore
from plugins.models.selection import MODEL_SELECTION, SelectionOwner
from plugins.models.model_settings_http import (
    ModelControl,
    create_model_settings_router,
    rpc_methods,
)
from bootstrap.chat_api import _model_rpc_response, create_chat_app
from infra.channels.web_chat_channel import WebChatChannel
from tests.test_model_call_records import descriptor, store, dump
from tests.test_mobile_message_log import mobile
from tests.mobile_realtime.test_channel import _generic_frame


@pytest.mark.asyncio
async def test_model_selection_reader_uses_current_snapshot_and_reports_missing_owner():
    root = CompositionRoot("selection")

    async def plugin(ctx):
        await ctx.provide(MODEL_SELECTION, SelectionOwner())

    await root.mount(plugin, name="models")
    snapshot = RuntimeSnapshotCompiler().compile(
        {}, composition_root=root, snapshot_revision="selection"
    )
    snapshots = RuntimeSnapshotStore()
    snapshots.install(snapshot)
    control = RuntimeModelControl(snapshots)
    try:
        assert await control.read_saved({"model_runtime_override": "model-a"}) == (
            ChatModelSelection("model-a", None)
        )
        assert snapshot.lease_count == 0

        absent_root = CompositionRoot("absent")
        absent = RuntimeSnapshotCompiler().compile(
            {}, composition_root=absent_root, snapshot_revision="absent"
        )
        await snapshots.commit(snapshots.begin_publish(absent))
        with pytest.raises(ModelControlUnavailable):
            await control.read_saved({"model_runtime_override": "model-a"})
        assert absent.lease_count == 0
        await absent_root.dispose()
    finally:
        await snapshots.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_runtime_model_control_propagates_reader_programming_errors():
    """Core model adapters must release the lease without masking provider bugs."""

    root = CompositionRoot("stats-error")

    def broken(_call_id: str):
        raise ValueError("provider invariant broken")

    async def plugin(ctx):
        await ctx.provide(MODEL_CALL_STATS, broken)

    await root.mount(plugin, name="models")
    snapshot = RuntimeSnapshotCompiler().compile(
        {}, composition_root=root, snapshot_revision="stats-error"
    )
    snapshots = RuntimeSnapshotStore()
    snapshots.install(snapshot)
    control = RuntimeModelControl(snapshots)
    try:
        with pytest.raises(ValueError, match="provider invariant broken"):
            await control.call_stats("call")
        assert snapshot.lease_count == 0
    finally:
        await snapshots.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_http_and_mobile_read_same_call_without_receipts_or_credentials(store, descriptor, mobile):
    log, runtime, channel, device = mobile
    call_id = store.start_call(descriptor, ModelRequest(({'role': 'user', 'content': 'private input'},)))
    store.record_first_token(call_id, 250)
    root = CompositionRoot('stats')
    async def plugin(ctx):
        await ctx.provide(MODEL_CALL_STATS, store.read_call_stats)
    await root.mount(plugin, name='models')
    snapshot = RuntimeSnapshotCompiler().compile({}, composition_root=root, snapshot_revision='stats')
    snapshots = RuntimeSnapshotStore()
    snapshots.install(snapshot)
    control = RuntimeModelControl(snapshots)
    channel.bind_model_stats(control.call_stats)
    app = FastAPI()
    # 此夹具只请求 call stats 路由，其他设置方法不在本用例范围内。
    app.include_router(create_model_settings_router(cast(ModelControl, control)))
    before = dump(store.path), dump(runtime.storage.db_path)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            reply = await client.get(f'/api/chat/model-settings/calls/{call_id}')
            assert reply.status_code == 200
            frame = _generic_frame(frame_id='01ARZ3NDEKTSV4RRFFQ69G5FAV', command_type='model.call.get', payload={'call_record_id': call_id})
            mobile_reply = await channel.handle_command(device_id=device, frame=frame)
            assert mobile_reply.type == 'model.call.get.ok'
            assert reply.json() == mobile_reply.payload == asdict(store.read_call_stats(call_id))
            assert snapshot.lease_count == 0
            missing = await client.get('/api/chat/model-settings/calls/missing')
            assert missing.status_code == 404
            bad_frame = frame.model_copy(update={'payload': {'call_record_id': 'missing'}})
            assert (await channel.handle_command(device_id=device, frame=bad_frame)).payload['code'] == 'model_call_not_found'
        assert (dump(store.path), dump(runtime.storage.db_path)) == before
        assert 'private input' not in reply.text and 'binding' not in reply.text and descriptor.auth_identity not in reply.text
        absent_root = CompositionRoot('absent')
        absent = RuntimeSnapshotCompiler().compile({}, composition_root=absent_root, snapshot_revision='absent')
        await snapshots.commit(snapshots.begin_publish(absent))
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                assert (await client.get(f'/api/chat/model-settings/calls/{call_id}')).status_code == 503
            assert (await channel.handle_command(device_id=device, frame=frame)).payload['code'] == 'model_stats_unavailable'
            assert absent.lease_count == 0
        finally:
            await absent_root.dispose()
    finally:
        await snapshots.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_chat_model_route_dispatches_plugin_rpc_under_one_snapshot_lease(
    store, descriptor, tmp_path
):
    call_id = store.start_call(descriptor, ModelRequest(()))
    root = CompositionRoot("rpc")

    class Control:
        async def call_stats(self, call_id: str):
            return store.read_call_stats(call_id)

    async def plugin(ctx):
        for name, operation in rpc_methods(cast(ModelControl, Control())).items():
            await ctx.provide(rpc_method_key(name), operation)

    await root.mount(plugin, name="models")
    snapshot = RuntimeSnapshotCompiler().compile(
        {}, composition_root=root, snapshot_revision="rpc"
    )
    snapshots = RuntimeSnapshotStore()
    snapshots.install(snapshot)
    control = RuntimeModelControl(snapshots)
    app = create_chat_app(
        workspace=tmp_path / "chat",
        channel=WebChatChannel(),
        model_control=control,
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(f"/api/chat/model-settings/calls/{call_id}")
            missing = await client.get("/api/chat/model-settings/calls/missing")
        assert response.status_code == 200
        assert response.json() == asdict(store.read_call_stats(call_id))
        assert missing.status_code == 404
        assert snapshot.lease_count == 0
        absent_root = CompositionRoot("absent-rpc")
        absent = RuntimeSnapshotCompiler().compile(
            {}, composition_root=absent_root, snapshot_revision="absent-rpc"
        )
        await snapshots.commit(snapshots.begin_publish(absent))
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                unavailable = await client.get(
                    f"/api/chat/model-settings/calls/{call_id}"
                )
            assert unavailable.status_code == 503
            assert absent.lease_count == 0
        finally:
            await absent_root.dispose()
    finally:
        await snapshots.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_chat_model_route_does_not_hide_provider_programming_errors():
    """Only plugin-owned validation errors become HTTP 422 responses."""

    class BrokenControl:
        async def invoke_rpc(
            self, method: str, params: Mapping[str, object]
        ) -> object:
            del method, params
            raise ValueError("provider invariant broken")

    with pytest.raises(ValueError, match="provider invariant broken"):
        await _model_rpc_response(BrokenControl(), "models/catalog", {})
