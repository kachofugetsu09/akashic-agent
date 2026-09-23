from dataclasses import asdict
from collections.abc import Mapping

import httpx
import pytest

from agent.plugin_composition.models import (
    MODEL_CALL_STATS,
    ModelRequest,
)
from agent.plugin_composition.ui import UI
from agent.config_models import Config
from bootstrap.dashboard_api import create_dashboard_app
from bootstrap.tools import build_core_runtime
from core.net.http import SharedHttpResources
from plugins.akashic_clients.chat_api import _model_rpc_response
from plugins.models.store import ModelsStore
from tests.test_model_call_records import descriptor, dump
from tests.test_mobile_message_log import mobile
from tests.mobile_realtime.test_channel import _generic_frame
from tests.fixtures.formal_plugins import FULL_RUNTIME_PLUGINS, install_formal_plugins


@pytest.mark.asyncio
async def test_http_and_mobile_read_same_call_without_receipts_or_credentials(
    tmp_path, monkeypatch, descriptor, mobile,
):
    """真实 Dashboard 与 Mobile 窄 reader 读取同一 ModelsStore 调用账。"""

    _, runtime, channel, device = mobile
    workspace = tmp_path / "workspace"
    plugin_home, _ = install_formal_plugins(
        tmp_path, FULL_RUNTIME_PLUGINS, configure_materials=True,
        initialize_persona=True,
    )
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(plugin_home))
    http = SharedHttpResources()
    core = build_core_runtime(Config(), workspace, http, plugin_dirs=[])
    app = create_dashboard_app(workspace, plugin_manager=core.plugin_manager)
    try:
        await core.start()
        await core.plugin_manager.start_runtime()
        root = core.plugin_manager.live_root
        assert root is not None
        ui_context, ui = root._service_provider(UI)
        async with ui_context.runtime_scope():
            catalog = ui.catalog()
        module = next(
            item for item in catalog.modules if item.plugin_id == "models@fixture"
        )
        headers = {
            "x-akashic-web-snapshot": root.generation_id,
            "x-akashic-web-catalog": catalog.identity,
            "x-akashic-web-module": module.plugin_id,
            "x-akashic-web-generation": module.generation_id,
        }
        store = ModelsStore(
            workspace / "model-registry.sqlite3",
            backup_dir=workspace / "runtime/model-backups",
            writable=True,
        )
        store.initialize()
        call_id = store.start_call(
            descriptor,
            ModelRequest(({"role": "user", "content": "private input"},)),
        )
        store.record_first_token(call_id, 250)

        stats_context, reader = root._service_provider(MODEL_CALL_STATS)
        stats_store = reader.__self__
        assert stats_store.path == store.path

        async def read_stats(model_call_id: str):
            async with stats_context.runtime_scope():
                return reader(model_call_id)

        frame = _generic_frame(
            frame_id='01ARZ3NDEKTSV4RRFFQ69G5FAV',
            command_type='model.call.get',
            payload={'call_record_id': call_id},
        )
        unbound = await channel.handle_command(device_id=device, frame=frame)
        assert unbound.type == 'model.call.get.error'
        assert unbound.payload['code'] == 'model_stats_unavailable'
        channel.bind_model_stats(read_stats)
        before = dump(store.path), dump(runtime.storage.db_path)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url='http://test',
            headers=headers,
        ) as client:
            reply = await client.get(f'/api/dashboard/models/calls/{call_id}')
            assert reply.status_code == 200
            mobile_reply = await channel.handle_command(device_id=device, frame=frame)
            assert mobile_reply.type == 'model.call.get.ok'
            assert reply.json() == mobile_reply.payload == asdict(store.read_call_stats(call_id))
            missing = await client.get('/api/dashboard/models/calls/missing')
            assert missing.status_code == 404
            bad_frame = frame.model_copy(update={'payload': {'call_record_id': 'missing'}})
            assert (await channel.handle_command(device_id=device, frame=bad_frame)).payload['code'] == 'model_call_not_found'
            with monkeypatch.context() as patch:
                def broken_read_call(_call_id: str):
                    raise ValueError("provider invariant broken")

                patch.setattr(stats_store, "read_call", broken_read_call)
                with pytest.raises(ValueError, match="provider invariant broken"):
                    await client.get(f'/api/dashboard/models/calls/{call_id}')
                with pytest.raises(ValueError, match="provider invariant broken"):
                    await channel.handle_command(device_id=device, frame=frame)
            assert not stats_context._fiber._in_flight_calls
            assert not ui_context._fiber._in_flight_calls
            restored = await client.get(f'/api/dashboard/models/calls/{call_id}')
            assert restored.status_code == 200
            restored_mobile = await channel.handle_command(device_id=device, frame=frame)
            assert restored.json() == restored_mobile.payload == asdict(store.read_call_stats(call_id))
            assert not stats_context._fiber._in_flight_calls
            assert not ui_context._fiber._in_flight_calls
        assert (dump(store.path), dump(runtime.storage.db_path)) == before
        assert 'private input' not in reply.text and 'binding' not in reply.text and descriptor.auth_identity not in reply.text
    finally:
        try:
            await core.bus.aclose()
        finally:
            try:
                await core.stop()
            finally:
                await http.aclose()


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
