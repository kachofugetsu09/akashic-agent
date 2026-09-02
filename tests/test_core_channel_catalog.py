from __future__ import annotations

import pytest

from agent.plugin_composition.channels import (
    ChannelCapability,
    ChannelDescriptor,
    ChannelFactoryProvenance,
    ChannelReady,
    ChannelRegistrySnapshot,
    CommittedChannelCatalog,
    CoreChannelDefinition,
    DeliveryStatus,
    InboundIdentity,
    ProviderDeliveryReceipt,
    ProviderDeliveryRequest,
    StopReceipt,
    _registry_identity,
)


class _RecordingAdapter:
    def __init__(self, binding_token: str = "core-binding") -> None:
        self.binding_token = binding_token

    async def start(self) -> ChannelReady:
        return ChannelReady(self.binding_token)

    async def deliver(self, request: ProviderDeliveryRequest) -> ProviderDeliveryReceipt:
        return ProviderDeliveryReceipt(request.delivery_id, DeliveryStatus.DELIVERED)

    async def stop(self) -> StopReceipt:
        return StopReceipt(self.binding_token, resources_closed=True)


def _build_adapter(_context: object) -> _RecordingAdapter:
    return _RecordingAdapter()


def _core_definition(name: str) -> CoreChannelDefinition:
    return CoreChannelDefinition(
        name=name,
        capabilities=frozenset({ChannelCapability.OUTBOUND}),
        factory=_build_adapter,
        inbound_identity=None,
        source_revision="core-source-1",
        config_revision="core-config-1",
        generation_id="core-generation-1",
        config={"channel": {"name": name}},
    )


def _plugin_registry(*, name: str = "feishu") -> ChannelRegistrySnapshot:
    descriptor = ChannelDescriptor(
        owner="plugin",
        name=name,
        capabilities=(ChannelCapability.OUTBOUND,),
        factory_export=f"{name}.build_channel",
        inbound_identity=None,
        credential_paths=("token",),
    )
    provenance = ChannelFactoryProvenance(
        plugin_id="plugin",
        generation_id="generation-1",
        channel_name=name,
        source_revision="source-1",
        config_revision="config-1",
        factory_export=descriptor.factory_export,
    )
    return ChannelRegistrySnapshot(
        descriptors=(descriptor,),
        factories=(provenance,),
        identity=_registry_identity((descriptor,), (provenance,)),
        root_instance_token=object(),
    )


def test_committed_catalog_merges_core_and_plugin_descriptors() -> None:
    plugin_registry = _plugin_registry()
    catalog = CommittedChannelCatalog(
        plugin_registry=plugin_registry,
        core_definitions=(_core_definition("telegram"), _core_definition("web")),
    )

    assert tuple(item.name for item in catalog.descriptors) == (
        "feishu",
        "telegram",
        "web",
    )
    assert catalog.descriptors[1].owner == "core"
    assert catalog.definition("web") is not None
    assert catalog.definition("feishu") is None
    assert catalog.registry.root_instance_token is plugin_registry.root_instance_token
    assert catalog.identity == catalog.registry.identity


def test_committed_catalog_fails_loud_on_core_plugin_collision() -> None:
    with pytest.raises(ValueError, match="名称冲突: telegram"):
        CommittedChannelCatalog(
            plugin_registry=_plugin_registry(name="telegram"),
            core_definitions=(_core_definition("telegram"),),
        )


def test_committed_catalog_canonicalizes_core_definition_order() -> None:
    catalog = CommittedChannelCatalog(
        plugin_registry=_plugin_registry(),
        core_definitions=(_core_definition("web"), _core_definition("telegram")),
    )

    assert tuple(item.name for item in catalog.core_definitions) == (
        "telegram",
        "web",
    )


def test_core_definition_validates_inbound_identity_and_freezes_config() -> None:
    with pytest.raises(ValueError, match="必须声明 inbound_identity"):
        CoreChannelDefinition(
            name="mobile",
            capabilities=frozenset({ChannelCapability.INBOUND}),
            factory=_build_adapter,
            inbound_identity=None,
            source_revision="core-source-1",
            config_revision="core-config-1",
            generation_id="core-generation-1",
        )

    definition = CoreChannelDefinition(
        name="web",
        capabilities=frozenset({ChannelCapability.OUTBOUND}),
        factory=_build_adapter,
        inbound_identity=None,
        source_revision="core-source-1",
        config_revision="core-config-1",
        generation_id="core-generation-1",
        config={"nested": {"enabled": True}},
    )
    assert definition.config["nested"]["enabled"] is True  # type: ignore[index]
    with pytest.raises(TypeError):
        definition.config["nested"] = {}  # type: ignore[index]


def test_committed_catalog_can_start_from_core_only_definitions() -> None:
    root = object()
    catalog = CommittedChannelCatalog(
        core_definitions=(_core_definition("mobile"),),
        root_instance_token=root,
    )

    assert catalog.plugin_registry is None
    assert catalog.root_instance_token is root
    assert catalog.registry.descriptors[0].owner == "core"


def test_core_config_changes_catalog_identity_but_input_order_does_not() -> None:
    first = _core_definition("telegram")
    changed = CoreChannelDefinition(
        name=first.name,
        capabilities=first.capabilities,
        factory=first.factory,
        inbound_identity=first.inbound_identity,
        source_revision=first.source_revision,
        config_revision=first.config_revision,
        generation_id=first.generation_id,
        config={"channel": {"name": "changed"}},
    )
    first_catalog = CommittedChannelCatalog(
        core_definitions=(first, _core_definition("web")),
    )
    reordered_catalog = CommittedChannelCatalog(
        core_definitions=(_core_definition("web"), first),
    )
    changed_catalog = CommittedChannelCatalog(
        core_definitions=(changed, _core_definition("web")),
    )

    assert first_catalog.identity == reordered_catalog.identity
    assert first_catalog.identity != changed_catalog.identity


def test_plugin_channel_definition_still_requires_credentials() -> None:
    with pytest.raises(ValueError, match="非空 tuple"):
        ChannelDescriptor(
            owner="plugin",
            name="plugin_channel",
            capabilities=(ChannelCapability.OUTBOUND,),
            factory_export="plugin.build_channel",
            inbound_identity=None,
            credential_paths=(),
        )
