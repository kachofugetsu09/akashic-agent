from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ClientInfo(StrictModel):
    name: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)


class ClientCapabilities(StrictModel):
    reasoningEvents: bool = False


class InitializeParams(StrictModel):
    protocolVersion: Literal["1.0"]
    clientInfo: ClientInfo
    capabilities: ClientCapabilities = Field(default_factory=ClientCapabilities)
    workspaceToken: str | None = None


class ThreadStartParams(StrictModel):
    metadata: dict[str, Any] = Field(default_factory=dict)


class ThreadIdParams(StrictModel):
    threadId: str = Field(min_length=1, max_length=512)


class ThreadReadParams(ThreadIdParams):
    includeTurns: bool = False


class ThreadListParams(StrictModel):
    cursor: str | None = None
    limit: int = Field(default=50, ge=1, le=200)


class TurnStartParams(ThreadIdParams):
    input: str = Field(min_length=1, max_length=1_048_576)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TurnIdParams(ThreadIdParams):
    turnId: str = Field(min_length=1, max_length=128)


class PluginDrainParams(StrictModel):
    pluginId: str = Field(min_length=1, max_length=256)


METHOD_PARAMS: dict[str, type[StrictModel]] = {
    "initialize": InitializeParams,
    "server/status": StrictModel,
    "thread/start": ThreadStartParams,
    "thread/resume": ThreadIdParams,
    "thread/list": ThreadListParams,
    "thread/read": ThreadReadParams,
    "thread/delete": ThreadIdParams,
    "thread/consolidate/start": ThreadIdParams,
    "turn/start": TurnStartParams,
    "turn/read": TurnIdParams,
    "turn/interrupt": TurnIdParams,
    "plugin/disable-and-drain": PluginDrainParams,
}
