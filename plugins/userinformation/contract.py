"""User information queries: providers own collection, storage and domain semantics."""
from collections.abc import Mapping
from typing import Protocol
from agent.plugin_composition import Context, Effect, ServiceKey


class InformationProvider(Protocol):
    async def describe(self) -> Mapping[str, object]:
        """Return available queries and their JSON object parameter schemas."""
        ...

    async def query(self, query: str, arguments: Mapping[str, object]) -> Mapping[str, object]:
        """Read evidence; never silently turn missing/stale data into a current fact."""
        ...


class InformationProviders(Protocol):
    async def register(self, ctx: Context, provider: InformationProvider) -> Effect: ...


USER_INFORMATION = ServiceKey[InformationProviders]("userinformation.providers.v1")


class InformationRead(Protocol):
    async def query(self, provider: str, query: str, arguments: Mapping[str, object]) -> Mapping[str, object]:
        """Bounded read of one provider; returns timestamped evidence or explicit failure."""
        ...


USER_INFORMATION_READ = ServiceKey[InformationRead]("userinformation.read.v1")
