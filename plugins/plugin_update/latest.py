"""显式候选程序复用普通工具、Task 和 Message，不启动后台裁判。"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.plugin_composition import Context
from agent.plugin_composition.messages import OWNER_STATE
from agent.plugin_composition.plugin_updates import PLUGIN_UPDATES
from agent.plugin_contracts import ContentPart, Output, body_to_dict, json_value

from .inputs import CallSource, Result
from .tool import Request, receipt
from .validation import PLUGIN_VALIDATION


class LatestInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    update_id: str = Field(min_length=1)
    action: Literal["run", "status", "revert"]


class Latest:
    """发起 Agent 显式执行、观察或撤销同一更新。"""

    idempotent = False

    def __init__(self, ctx: Context):
        self._ctx = ctx

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object] | str:
        try:
            request = LatestInput.model_validate(json_value(arguments))
        except ValidationError as error:
            return str(error)
        return request.model_dump(mode="json")

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        """只有本次实际程序正常完成才请求发布；重查不重跑程序。"""
        request = LatestInput.model_validate(json_value(arguments))
        ctx = self._ctx
        updates = ctx.require(PLUGIN_UPDATES)
        identity = request.update_id
        store = ctx.require(OWNER_STATE).open(ctx)
        record = store.read(identity)
        if record is None:
            raise ValueError("没有对应的插件安装请求")
        install = Request.model_validate(json_value(record.value))
        if request.action == "status":
            status = updates.read(ctx, identity)
            if status is None:
                raise ValueError("更新没有候选收据")
            messages = updates.messages(ctx, identity, "plugin-validation:" + identity)
            text = json.dumps({"update": asdict(status), "messages": [
                {"message_id": message.message_id, "body": body_to_dict(message.body)}
                for message in messages
            ]}, ensure_ascii=False)
            return Result("success", (ContentPart("text", text),))
        if request.action == "revert":
            await updates.discard(ctx, identity, reason="发起 Agent 已 revert")
            status = updates.read(ctx, identity)
            assert status is not None
            return Result("success", (ContentPart("text", json.dumps(asdict(status), ensure_ascii=False)),))

        # 1. 普通工具恢复只查询原请求，未知结果没有第二次执行权。
        run_key = "latest-call:" + key
        if store.read(run_key) is not None:
            raise RuntimeError("已有 latest 调用只能查询，不得重跑")
        store.transact(lambda tx: tx.save(run_key, request.model_dump(mode="json"), expected_version=None))
        async with updates.open_validation(ctx, identity) as scope:
            output = await scope.require(PLUGIN_VALIDATION).run(identity, install.install)
        # 2. 候选程序及隔离资源均已退出；宿主等待调用租约释放后才换代。
        updates.publish(ctx, identity)
        return Result("success", cast(tuple[ContentPart, ...], cast(Output, output.body).parts))

    async def query(self, key: str) -> Result | None:
        record = self._ctx.require(OWNER_STATE).open(self._ctx).read("latest-call:" + key)
        if record is None:
            return None
        request = LatestInput.model_validate(json_value(record.value))
        status = self._ctx.require(PLUGIN_UPDATES).read(self._ctx, request.update_id)
        if status is None:
            return Result("error", (ContentPart("text", "latest 调用结果未知；请查询原更新，不自动重跑。"),))
        if status.phase == "committed":
            return receipt(status)
        return Result("error", (ContentPart("text", json.dumps(asdict(status), ensure_ascii=False)),))
