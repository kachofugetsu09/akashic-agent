"""set_model_enabled：逻辑停用保留持久行，在用引用拒绝停用。"""

import pytest

from agent.plugin_composition.models import CapabilitySources, ModelCapabilities, ModelKind
from plugins.models.model_settings_http import CommandParams, _command
from plugins.models.settings import (
    AddConnection,
    AddModel,
    CreateConnectionWithModel,
    SetDefaultModel,
    SetModelEnabled,
)
from plugins.models.state import ModelUnavailableError
from plugins.models.store import ModelsStore, RevisionConflictError
from tests.support.material_models import material_models


def _chat_model(model_id: str, connection_id: str, model: str) -> AddModel:
    return AddModel(
        expected_revision=0,
        model_id=model_id,
        connection_id=connection_id,
        kind=ModelKind.CHAT,
        model=model,
        capabilities=ModelCapabilities(input_modalities=("text",)),
        capability_sources=CapabilitySources(),
    )


def _connection(connection_id: str) -> AddConnection:
    return AddConnection(
        expected_revision=0,
        connection_id=connection_id,
        name="T",
        driver_id="openai-compatible",
        endpoint="https://api.example.com/v1",
        auth_identity=f"api:{connection_id}",
        credential={"driver": "api_key", "access_token": "key"},
        driver_config={"format_version": 1},
    )


@pytest.fixture()
def store(tmp_path):
    instance = ModelsStore(tmp_path / "registry.sqlite3", backup_dir=tmp_path / "backups", writable=True)
    instance.initialize()
    yield instance
    instance.close()


def _create(store: ModelsStore) -> int:
    return store.create_connection_with_model(
        CreateConnectionWithModel(
            connection=_connection("c1"),
            model=_chat_model("c1__m1", "c1", "m1"),
        )
    )


def test_disable_bound_model_is_refused(store):
    revision = _create(store)
    revision = store.set_default(SetDefaultModel(expected_revision=revision, role="default", model_id="c1__m1"))
    with pytest.raises(ValueError, match="still in use"):
        store.set_model_enabled(SetModelEnabled(expected_revision=revision, model_id="c1__m1", enabled=False))
    assert store.read_snapshot().models["c1__m1"].enabled


def test_disable_preserves_row_and_reenable(store):
    revision = _create(store)
    revision = store.set_model_enabled(SetModelEnabled(expected_revision=revision, model_id="c1__m1", enabled=False))
    snapshot = store.read_snapshot()
    assert not snapshot.models["c1__m1"].enabled
    revision = store.set_model_enabled(SetModelEnabled(expected_revision=revision, model_id="c1__m1", enabled=True))
    assert store.read_snapshot().models["c1__m1"].enabled


def test_noop_keeps_revision(store):
    revision = _create(store)
    next_revision = store.set_model_enabled(SetModelEnabled(expected_revision=revision, model_id="c1__m1", enabled=True))
    assert next_revision == revision


def test_missing_model_is_rejected(store):
    revision = _create(store)
    with pytest.raises(ValueError, match="does not exist"):
        store.set_model_enabled(SetModelEnabled(expected_revision=revision, model_id="missing", enabled=False))


def test_expected_revision_is_enforced(store):
    _create(store)
    with pytest.raises(RevisionConflictError):
        store.set_model_enabled(SetModelEnabled(expected_revision=0, model_id="c1__m1", enabled=False))


def test_http_payload_converts():
    payload = CommandParams.model_validate({
        "type": "set_model_enabled",
        "expected_revision": 3,
        "model_id": "c1__m1",
        "enabled": False,
    }).root
    command = _command(payload)
    assert command == SetModelEnabled(expected_revision=3, model_id="c1__m1", enabled=False)


@pytest.mark.asyncio()
async def test_reenable_runs_real_verification(tmp_path, monkeypatch):
    """重新开放必须向 provider 真实验证；失败时模型保持停用且 revision 不变。"""

    async with material_models(tmp_path) as models:
        # fixture 的默认角色占用 material-fixture-chat；再加一个未绑定的聊天模型。
        revision = models.store.read_snapshot().revision
        receipt = await models.settings.apply(
            AddModel(
                expected_revision=revision,
                model_id="c1__extra",
                connection_id="material-fixture-connection",
                kind=ModelKind.CHAT,
                model="fixture-extra",
                capabilities=ModelCapabilities(input_modalities=("text",)),
                capability_sources=CapabilitySources(),
            )
        )
        receipt = await models.settings.apply(
            SetModelEnabled(
                expected_revision=receipt.revision,
                model_id="c1__extra",
                enabled=False,
            )
        )
        assert not models.store.read_snapshot().models["c1__extra"].enabled

        # 重新开放前驱动先坏掉：验证失败则保持停用、revision 不动。
        driver_type = type(models.driver)
        original_bind_chat = driver_type.bind_chat

        def failing_bind(self, descriptor, config):
            if descriptor.model == "fixture-extra":
                raise ModelUnavailableError("provider rejected")
            return original_bind_chat(self, descriptor, config)

        monkeypatch.setattr(driver_type, "bind_chat", failing_bind)
        with pytest.raises(ModelUnavailableError, match="provider rejected"):
            await models.settings.apply(
                SetModelEnabled(
                    expected_revision=receipt.revision,
                    model_id="c1__extra",
                    enabled=True,
                )
            )
        assert not models.store.read_snapshot().models["c1__extra"].enabled
        assert models.store.read_snapshot().revision == receipt.revision

        # 驱动恢复后重新开放：fixture 记录了真实 complete 验证调用。
        monkeypatch.undo()
        models.driver.chat_requests.clear()
        await models.settings.apply(
            SetModelEnabled(
                expected_revision=receipt.revision,
                model_id="c1__extra",
                enabled=True,
            )
        )
        assert models.store.read_snapshot().models["c1__extra"].enabled
        assert len(models.driver.chat_requests) == 1
