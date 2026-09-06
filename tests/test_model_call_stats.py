from dataclasses import asdict

import httpx
import pytest
from fastapi import FastAPI

from agent.plugin_composition import CompositionRoot
from agent.plugin_composition.models import MODEL_CALL_STATS, ModelRequest
from agent.plugin_composition.model_settings_http import create_model_settings_router
from agent.plugins.model_control import RuntimeModelControl
from agent.plugins.snapshot import RuntimeSnapshotCompiler, RuntimeSnapshotStore
from tests.test_model_call_records import descriptor, store, dump
from tests.test_mobile_message_log import mobile
from tests.mobile_realtime.test_channel import _generic_frame


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
    app.include_router(create_model_settings_router(control))
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
