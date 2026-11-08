"""Read host deployment failures from the existing shared task directory."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from .contract import JobStatus


class HostNoticeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(min_length=1, max_length=512)
    job_id: str = Field(pattern=r"^[0-9A-Za-z_-]{1,64}$")
    kind: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=200)
    status: JobStatus
    detail: str = Field(max_length=8000)
    collect_hint: str = Field(default="", max_length=2000)


class SavedNotice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    notice: HostNoticeParams
    handled: bool = False
    delivery_error: str | None = None


class _Stop(BaseModel):
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    session_id: str = Field(min_length=1, max_length=512)


class _Request(BaseModel):
    stop: _Stop
    targetCommit: str = Field(pattern=r"^[0-9a-f]{40}$")
    unit: str = Field(pattern=r"^akashic-deploy-[0-9a-f]{32}\.service$")
    notifySession: str | None = Field(default=None, min_length=1, max_length=512)


class _Failure(BaseModel):
    request: _Request
    error: str


# Producer and recovery use the same validated payload and stable job identity.
def failure_notice(record: Mapping[str, object]) -> HostNoticeParams:
    failure = _Failure.model_validate(record)
    request = failure.request
    identity = request.stop.request_id
    if request.unit != f"akashic-deploy-{identity}.service":
        raise ValueError("部署任务编号与 service 不一致")
    return HostNoticeParams(
        session_id=request.notifySession or request.stop.session_id, job_id=identity,
        kind="self-deploy", label=f"部署 {request.targetCommit[:12]}", status="failed", detail=failure.error[-7000:],
        collect_hint=(f"用 Shell 运行 `akashic-release status {identity}` 和 "
            f"`journalctl --user -u {request.unit} --no-pager -n 80` 查明原因。"
            "仅在原授权范围内修复；送达失败、来源不支持或次数上限时等待用户新消息，不要换 SHA 重交。"
            "默认自部署通知来源没有最终送达 provider；修复后如实汇报，等用户新对话回合再提交。"
            "说明保留现场或回退失败时交给部署者处理。"))


# The task file remains host-owned; Core only stores its own durable acknowledgement.
def read_failure(path: Path) -> HostNoticeParams | None:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("部署记录必须是对象")
    record = cast(dict[str, object], raw)
    if record.get("status") != "failed" or record.get("notice") == "not-accepted":
        return None
    notice = failure_notice(record)
    if path.stem != notice.job_id:
        raise ValueError("部署记录文件名与任务编号不一致")
    return notice
