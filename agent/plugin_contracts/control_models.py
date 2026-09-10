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
    protocolVersion: Literal["2.0"]
    clientInfo: ClientInfo
    capabilities: ClientCapabilities = Field(default_factory=ClientCapabilities)
    workspaceToken: str | None = None


class ThreadStartParams(StrictModel):
    metadata: dict[str, Any] = Field(default_factory=dict)
    runtime: Literal["stable", "latest"] = "stable"
    pluginRolloutCapability: str = Field(default="", max_length=256)


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
    runtime: Literal["stable", "latest"] | None = None
    detached: bool = False


class TurnIdParams(ThreadIdParams):
    turnId: str = Field(min_length=1, max_length=128)


class PluginDrainParams(StrictModel):
    pluginId: str = Field(min_length=1, max_length=256)
    ownerTurnId: str = Field(default="", max_length=128)


class PluginInstallParams(StrictModel):
    source: str = Field(min_length=1, max_length=4096)
    marketplace: str = Field(default="local", min_length=1, max_length=128)
    ref: str = Field(default="", max_length=1024)
    sparse: list[str] = Field(default_factory=list, max_length=128)
    ownerTurnId: str = Field(default="", max_length=128)


class PluginRevertParams(StrictModel):
    ownerTurnId: str = Field(min_length=1, max_length=128)


class SessionIdParams(StrictModel):
    session_id: str = Field(min_length=1, max_length=512)


class SessionListParams(StrictModel):
    cursor: list[str] | None = Field(default=None, min_length=2, max_length=2)
    limit: int = Field(default=50, ge=1, le=200)


class MessageReadParams(SessionIdParams):
    after_seq: int = Field(default=-1, ge=-1)
    through_seq: int | None = Field(default=None, ge=-1)
    limit: int = Field(default=50, ge=1, le=200)


class MessageSendParams(SessionIdParams):
    message_id: str = Field(min_length=1, max_length=256)
    text: str = Field(default="", max_length=1_048_576)
    attachment_ids: list[str] = Field(default_factory=list, max_length=64)
    reply_to_message_id: str | None = Field(default=None, min_length=1, max_length=256)
    model_id: str | None = Field(default=None, max_length=256)
    reasoning_effort: str | None = Field(default=None, max_length=128)
    retry_of: str | None = Field(default=None, min_length=1, max_length=256)


class SessionFollowParams(SessionIdParams):
    after_seq: int = Field(default=-1, ge=-1)
    subscription_id: str = Field(min_length=1, max_length=128)


class SessionUnfollowParams(SessionIdParams):
    subscription_id: str = Field(min_length=1, max_length=128)


class PluginIdParams(StrictModel):
    plugin_id: str = Field(min_length=1, max_length=256)


class UpdateIdParams(StrictModel):
    update_id: str = Field(min_length=1, max_length=256)


class InstallParams(UpdateIdParams):
    source: str = Field(min_length=1, max_length=4096)
    marketplace: str = Field(default="local", min_length=1, max_length=128)
    ref: str = Field(default="", max_length=1024)
    sparse: list[str] = Field(default_factory=list, max_length=128)


METHOD_PARAMS: dict[str, type[StrictModel]] = {
    "initialize": InitializeParams,
    "server/status": StrictModel,
    "session/create": StrictModel,
    "session/list": SessionListParams,
    "message/read": MessageReadParams,
    "message/send": MessageSendParams,
    "session/follow": SessionFollowParams,
    "session/unfollow": SessionUnfollowParams,
    "plugin/install": InstallParams,
    "plugin/status": StrictModel,
    "plugin/update": UpdateIdParams,
    "plugin/promote": UpdateIdParams,
    "plugin/discard": UpdateIdParams,
    "plugin/disable-and-drain": PluginIdParams,
    "plugin/uninstall": PluginIdParams,
}
