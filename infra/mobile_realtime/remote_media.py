"""Compatibility exports for the retired mobile channel path."""

from infra.channels.remote_media import (
    AddressResolver,
    BackendFactory,
    PinnedNetworkBackend,
    RemoteMediaError,
    RemoteMediaSnapshot,
    snapshot_remote_media,
)

__all__ = [
    "AddressResolver",
    "BackendFactory",
    "PinnedNetworkBackend",
    "RemoteMediaError",
    "RemoteMediaSnapshot",
    "snapshot_remote_media",
]
