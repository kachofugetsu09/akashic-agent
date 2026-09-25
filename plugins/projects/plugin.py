"""Project 是 Session 宽键中的 project 维度；只拥有项目记录，不解释记忆或展示。"""
from __future__ import annotations

from datetime import UTC, datetime
import re
from typing import cast

from agent.plugin_composition import UI_SLOTS, Context, MobileUiDefinition, MobileUiRpcInvalidRequest
from agent.plugin_composition.messages import (
    OWNER_STATE,
    SESSION_ADMISSION,
    MessageConflict,
    OwnerStore,
    OwnerTransaction,
)

api_version = 3
name = "projects"
version = "1.0.0"
desc = "按项目组织对话，并为 Session 宽键提供 project 维度"
inject = (SESSION_ADMISSION, OWNER_STATE, UI_SLOTS)

DIMENSION = "project"
_PREFIX = "project:"
_NAME_LIMIT = 80
_PROJECT_ID = re.compile(r"p_[0-9a-f]{32}")


class Projects:
    """项目记录只追加或原位改名、归档；从不删除，历史 Session 永远能解析。"""

    def __init__(self, store: OwnerStore):
        self._store = store

    def list(self) -> list[dict[str, object]]:
        rows = self._store.scan(start=_PREFIX, stop=_PREFIX + "\uffff", limit=1000)
        projects = [_project_row(key[len(_PREFIX):], dict(record.value)) for key, record in rows]
        return sorted(projects, key=lambda row: cast(str, row["created_at"]))

    def create(self, project_id: str, project_name: str) -> dict[str, object]:
        """由请求的稳定 ID 创建一次；响应丢失后同名重放返回原记录。"""
        if _PROJECT_ID.fullmatch(project_id) is None:
            raise MobileUiRpcInvalidRequest("项目 ID 无效")
        name = _check_name(project_name)
        value: dict[str, object] = {
            "name": name, "created_name": name, "archived": False,
            "created_at": datetime.now(UTC).isoformat(),
        }
        def save(transaction: OwnerTransaction) -> dict[str, object]:
            current = transaction.read(_PREFIX + project_id)
            if current is not None:
                if current.value.get("created_name", current.value["name"]) != name:
                    raise MobileUiRpcInvalidRequest("项目 ID 已用于其他名称")
                return _project_row(project_id, dict(current.value))
            _ = transaction.save(_PREFIX + project_id, value, expected_version=None)
            return _project_row(project_id, value)
        try:
            return self._store.transact(save)
        except MessageConflict as error:
            raise MobileUiRpcInvalidRequest("项目正在并发创建，请重试") from error

    def update(self, project_id: str, **changes: object) -> dict[str, object]:
        key = _PREFIX + project_id
        def save(transaction: OwnerTransaction) -> dict[str, object]:
            current = transaction.read(key)
            if current is None:
                raise MobileUiRpcInvalidRequest("项目不存在")
            value = {**dict(current.value), **changes}
            _ = transaction.save(key, value, expected_version=current.version)
            return _project_row(project_id, value)
        try:
            return self._store.transact(save)
        except MessageConflict as error:
            raise MobileUiRpcInvalidRequest("项目已被并发修改，请刷新后重试") from error

    # Session 首次接纳时由 Core 调用；归档项目不再接纳新对话。
    def check(self, project_id: str) -> None:
        record = self._store.read(_PREFIX + project_id)
        if record is None:
            raise MessageConflict(f"项目不存在: {project_id}")
        if record.value.get("archived") is True:
            raise MessageConflict(f"项目已归档: {project_id}")


def _check_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > _NAME_LIMIT:
        raise MobileUiRpcInvalidRequest(f"项目名称必须是 1 到 {_NAME_LIMIT} 个字符")
    return value.strip()


def _project_id(payload: dict[str, object]) -> str:
    value = payload.get("project_id")
    if not isinstance(value, str) or not value:
        raise MobileUiRpcInvalidRequest("请选择一个项目")
    return value


def _project_row(project_id: str, value: dict[str, object]) -> dict[str, object]:
    return {
        "id": project_id, "name": value["name"],
        "archived": value["archived"], "created_at": value["created_at"],
    }


async def apply(ctx: Context) -> None:
    """注册 project 维度的唯一 owner，并经插件查询通道提供项目增改。"""
    projects = Projects(ctx.require(OWNER_STATE).open(ctx))
    _ = await ctx.require(SESSION_ADMISSION).register_dimension(ctx, name=DIMENSION, check=projects.check)

    def query(method: str, payload: dict[str, object], *, session_id: str | None,
              turn_id: str | None) -> dict[str, object]:
        if method == "project.list" and not payload:
            return {"dimension": DIMENSION, "items": projects.list()}
        if method == "project.create" and set(payload) == {"project_id", "name"}:
            return projects.create(_project_id(payload), _check_name(payload["name"]))
        if method == "project.rename" and set(payload) == {"project_id", "name"}:
            return projects.update(_project_id(payload), name=_check_name(payload["name"]))
        if method == "project.archive" and set(payload) == {"project_id"}:
            return projects.update(_project_id(payload), archived=True)
        raise MobileUiRpcInvalidRequest(f"不支持的项目查询：{method}")

    _ = await ctx.require(UI_SLOTS).register_mobile(
        ctx, MobileUiDefinition(module="mobile_ui.js"), query=query,
    )
