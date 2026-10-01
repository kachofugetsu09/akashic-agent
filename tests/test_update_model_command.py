"""update_model: user-declared chat capabilities write all three fields at once."""
import pytest

from agent.plugin_composition.models import (
    CapabilitySources,
    ModelCapabilities,
    ModelKind,
)
from plugins.models.settings import (
    AddConnection,
    AddModel,
    CreateConnectionWithModel,
    UpdateModel,
)
from plugins.models.store import ModelsStore


def _store(tmp_path):
    store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
    store.initialize()
    return store


def _seed(store: ModelsStore, *, capabilities=None, sources=None) -> int:
    return store.create_connection_with_model(
        CreateConnectionWithModel(
            connection=AddConnection(
                0,
                "c1",
                "DeepSeek",
                "openai-compatible",
                "https://api.example/v1",
                "api:c1",
                {"driver": "api_key", "access_token": "k"},
                {"format_version": 1},
            ),
            model=AddModel(
                0,
                "c1__m1",
                "c1",
                ModelKind.CHAT,
                "m1",
                capabilities if capabilities is not None else ModelCapabilities(
                    input_modalities=("text",),
                    context_window=200000,
                    supported_reasoning_efforts=("low", "high"),
                    supports_tool_calls=True,
                ),
                sources if sources is not None else CapabilitySources(
                    context_window="litellm",
                    input_modalities="litellm",
                    reasoning_efforts="litellm",
                    tool_calls="litellm",
                ),
            ),
        )
    )


def test_update_model_rewrites_three_fields_and_marks_sources_user(tmp_path):
    store = _store(tmp_path)
    revision = _seed(store)

    revision = store.update_model(
        UpdateModel(revision, "c1__m1", 1_000_000, 32768, True)
    )

    model = store.read_snapshot().models["c1__m1"]
    assert model.capabilities.context_window == 1_000_000
    assert model.capabilities.max_output_tokens == 32768
    assert model.capabilities.input_modalities == ("text", "image")
    assert model.capability_sources.context_window == "user"
    assert model.capability_sources.max_output_tokens == "user"
    assert model.capability_sources.input_modalities == "user"


def test_update_model_preserves_unrelated_capability_fields(tmp_path):
    store = _store(tmp_path)
    revision = _seed(store)

    store.update_model(UpdateModel(revision, "c1__m1", 64000, 4096, False))

    model = store.read_snapshot().models["c1__m1"]
    assert model.capabilities.supported_reasoning_efforts == ("low", "high")
    assert model.capabilities.supports_tool_calls is True
    assert model.capability_sources.reasoning_efforts == "litellm"
    assert model.capability_sources.tool_calls == "litellm"


def test_update_model_none_clears_numeric_capability(tmp_path):
    store = _store(tmp_path)
    revision = _seed(store)

    store.update_model(UpdateModel(revision, "c1__m1", None, None, True))

    model = store.read_snapshot().models["c1__m1"]
    assert model.capabilities.context_window is None
    assert model.capabilities.max_output_tokens is None
    assert model.capabilities.input_modalities == ("text", "image")


def test_update_model_image_off_keeps_text_modality(tmp_path):
    store = _store(tmp_path)
    revision = _seed(store)
    revision = store.update_model(
        UpdateModel(revision, "c1__m1", 200000, 8192, True)
    )

    store.update_model(UpdateModel(revision, "c1__m1", 200000, 8192, False))

    model = store.read_snapshot().models["c1__m1"]
    assert model.capabilities.input_modalities == ("text",)


def test_update_model_synthesizes_payload_for_legacy_row(tmp_path):
    store = _store(tmp_path)
    revision = _seed(store)
    with store._connect() as connection:
        connection.execute(
            "UPDATE model_definitions SET capabilities_json = NULL WHERE id = 'c1__m1'"
        )
        connection.commit()

    store.update_model(UpdateModel(revision, "c1__m1", 128000, 8192, True))

    model = store.read_snapshot().models["c1__m1"]
    assert model.capabilities.context_window == 128000
    assert model.capabilities.max_output_tokens == 8192
    assert model.capabilities.input_modalities == ("text", "image")
    assert model.capabilities.supported_reasoning_efforts == ("low", "high")


def test_update_model_rejects_embedding_model(tmp_path):
    store = _store(tmp_path)
    revision = _seed(store)
    revision = store.add_model(
        AddModel(
            revision,
            "c1__emb",
            "c1",
            ModelKind.EMBEDDING,
            "emb-1",
            ModelCapabilities(embedding_dimensions=768),
            CapabilitySources(embedding_dimensions="probe"),
        )
    )

    with pytest.raises(ValueError, match="chat capabilities"):
        store.update_model(UpdateModel(revision, "c1__emb", 128000, 8192, True))


def test_update_model_rejects_non_positive_and_missing_model(tmp_path):
    store = _store(tmp_path)
    revision = _seed(store)

    with pytest.raises(ValueError, match="positive"):
        store.update_model(UpdateModel(revision, "c1__m1", 0, 8192, True))
    with pytest.raises(ValueError, match="positive"):
        store.update_model(UpdateModel(revision, "c1__m1", 128000, -1, True))
    with pytest.raises(ValueError, match="does not exist"):
        store.update_model(UpdateModel(revision, "missing", 128000, 8192, True))


def test_update_model_requires_current_revision(tmp_path):
    store = _store(tmp_path)
    _seed(store)

    with pytest.raises(Exception, match="revision"):
        store.update_model(UpdateModel(99, "c1__m1", 128000, 8192, True))
