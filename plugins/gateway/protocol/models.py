from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from ..contract import NAMES


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
    NAMES.initialize: InitializeParams,
    NAMES.server_status: StrictModel,
    NAMES.session_create: StrictModel,
    NAMES.session_list: SessionListParams,
    NAMES.message_read: MessageReadParams,
    NAMES.message_send: MessageSendParams,
    NAMES.session_follow: SessionFollowParams,
    NAMES.session_unfollow: SessionUnfollowParams,
    NAMES.plugin_install: InstallParams,
    NAMES.plugin_status: StrictModel,
    NAMES.plugin_update: UpdateIdParams,
    NAMES.plugin_drain: PluginIdParams,
    NAMES.plugin_uninstall: PluginIdParams,
}
