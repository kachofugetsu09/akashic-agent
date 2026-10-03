"""Project 是 Session 宽键中的 project 维度；只拥有项目记录，不解释记忆或展示。"""
from __future__ import annotations

from datetime import UTC, datetime
import re
from typing import cast
from pathlib import Path
from core.common.file_io import run_file_io

from agent.plugin_contracts.directories import WORKING_DIRECTORY

from agent.plugin_composition import UI_SLOTS, Context, PluginUiDefinition, PluginUiRpcInvalidRequest
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

    def create(self, project_id: str, project_name: str, directory: str | None = None) -> dict[str, object]:
        """由请求的稳定 ID 创建一次；响应丢失后同名重放返回原记录。"""
        if _PROJECT_ID.fullmatch(project_id) is None:
            raise PluginUiRpcInvalidRequest("项目 ID 无效")
        name = _check_name(project_name)
        value: dict[str, object] = {
            "name": name, "created_name": name, "archived": False,
            "created_at": datetime.now(UTC).isoformat(), "created_directory": directory,
        }
        if directory is not None:
            value["directory"] = directory
        def save(transaction: OwnerTransaction) -> dict[str, object]:
            current = transaction.read(_PREFIX + project_id)
            if current is not None:
                if current.value.get("created_name", current.value["name"]) != name:
                    raise PluginUiRpcInvalidRequest("项目 ID 已用于其他名称")
                if current.value.get("created_directory") != directory:
                    raise PluginUiRpcInvalidRequest("项目 ID 已用于其他初始目录")
                return _project_row(project_id, dict(current.value))
            _ = transaction.save(_PREFIX + project_id, value, expected_version=None)
            return _project_row(project_id, value)
        try:
            return self._store.transact(save)
        except MessageConflict as error:
            raise PluginUiRpcInvalidRequest("项目正在并发创建，请重试") from error

    def exists(self, project_id: str) -> bool:
        return self._store.read(_PREFIX + project_id) is not None

    def update(self, project_id: str, **changes: object) -> dict[str, object]:
        if set(changes) - {"name", "archived"}:
            raise PluginUiRpcInvalidRequest("项目更新只允许改名或归档")
        key = _PREFIX + project_id
        def save(transaction: OwnerTransaction) -> dict[str, object]:
            current = transaction.read(key)
            if current is None:
                raise PluginUiRpcInvalidRequest("项目不存在")
            value = {**dict(current.value), **changes}
            _ = transaction.save(key, value, expected_version=current.version)
            return _project_row(project_id, value)
        try:
            return self._store.transact(save)
        except MessageConflict as error:
            raise PluginUiRpcInvalidRequest("项目已被并发修改，请刷新后重试") from error

    def directory(self, project_id: str) -> str | None:
        record = self._store.read(_PREFIX + project_id)
        if record is None:
            raise PluginUiRpcInvalidRequest("项目不存在")
        path = record.value.get("directory")
        if path is not None and (not isinstance(path, str) or not Path(path).is_absolute()):
            raise ValueError("Project 目录记录损坏")
        return path

    def bind_directory(self, project_id: str, path: str) -> dict[str, object]:
        """Set the default once while preserving concurrent rename/archive fields."""
        def save(transaction: OwnerTransaction) -> dict[str, object]:
            record = transaction.read(_PREFIX + project_id)
            if record is None:
                raise PluginUiRpcInvalidRequest("项目不存在")
            current = record.value.get("directory")
            if current is not None:
                if current != path:
                    raise PluginUiRpcInvalidRequest("项目默认目录已固定，不可更改或清空")
                return _project_row(project_id, dict(record.value))
            value = {**dict(record.value), "directory": path}
            _ = transaction.save(_PREFIX + project_id, value, expected_version=record.version)
            return _project_row(project_id, value)

        return self._store.transact(save)

    # Session 首次接纳时由 Core 调用；归档项目不再接纳新对话。
    def check(self, project_id: str) -> None:
        record = self._store.read(_PREFIX + project_id)
        if record is None:
            raise MessageConflict(f"项目不存在: {project_id}")
        if record.value.get("archived") is True:
            raise MessageConflict(f"项目已归档: {project_id}")


def _check_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > _NAME_LIMIT:
        raise PluginUiRpcInvalidRequest(f"项目名称必须是 1 到 {_NAME_LIMIT} 个字符")
    return value.strip()


def _project_id(payload: dict[str, object]) -> str:
    value = payload.get("project_id")
    if not isinstance(value, str) or not value:
        raise PluginUiRpcInvalidRequest("请选择一个项目")
    return value


def _project_row(project_id: str, value: dict[str, object]) -> dict[str, object]:
    return {
        "id": project_id, "name": value["name"],
        "archived": value["archived"], "created_at": value["created_at"],
        "directory": value.get("directory"),
    }


async def apply(ctx: Context) -> None:
    """注册 project 维度的唯一 owner，并经插件查询通道提供项目增改。"""
    projects = Projects(ctx.require(OWNER_STATE).open(ctx))
    def check_project(project_id: str) -> None:
        projects.check(project_id)
        if projects.directory(project_id) is not None:
            with ctx.borrow(WORKING_DIRECTORY) as directories:
                if directories is None:
                    raise MessageConflict("已关联目录的项目需要当前目录能力才能创建 Session")

    _ = await ctx.require(SESSION_ADMISSION).register_dimension(ctx, name=DIMENSION, check=check_project)

    async def defaults(child: Context) -> None:
        _ = await child.require(WORKING_DIRECTORY).register_default(
            child, dimension=DIMENSION, read=projects.directory,
        )

    _ = await ctx.inject((WORKING_DIRECTORY,), defaults, name="directory-default")

    async def query(method: str, payload: dict[str, object], *, session_id: str | None,
              turn_id: str | None) -> dict[str, object]:
        if method == "project.list" and not payload:
            return {"dimension": DIMENSION, "items": await run_file_io(projects.list)}
        if method == "project.create" and set(payload) in (
            {"project_id", "name"}, {"project_id", "name", "directory"},
        ):
            project_id, project_name = _project_id(payload), _check_name(payload["name"])
            directory = payload.get("directory")
            if directory is not None:
                if not isinstance(directory, str) or not Path(directory).is_absolute():
                    raise PluginUiRpcInvalidRequest("请选择执行主机上的绝对目录")
                # A replay acknowledges the original commit even if its path moved.
                if not await run_file_io(lambda: projects.exists(project_id)):
                    with ctx.borrow(WORKING_DIRECTORY) as directories:
                        if directories is None:
                            raise PluginUiRpcInvalidRequest("当前组合没有可用的目录工具")
                        try:
                            _ = await directories.check_directory(directory)
                        except ValueError as error:
                            raise PluginUiRpcInvalidRequest(str(error)) from error
            return await run_file_io(lambda: projects.create(project_id, project_name, directory))
        if method == "project.rename" and set(payload) == {"project_id", "name"}:
            project_id, project_name = _project_id(payload), _check_name(payload["name"])
            return await run_file_io(lambda: projects.update(project_id, name=project_name))
        if method == "project.archive" and set(payload) == {"project_id"}:
            project_id = _project_id(payload)
            return await run_file_io(lambda: projects.update(project_id, archived=True))
        if method in {"project.bind_directory", "project.directory", "directory.browse"}:
            with ctx.borrow(WORKING_DIRECTORY) as directories:
                if directories is None:
                    raise PluginUiRpcInvalidRequest("当前组合没有可用的目录工具")
                if method == "project.bind_directory" and set(payload) == {"project_id", "path"}:
                    project_id = _project_id(payload)
                    path = payload["path"]
                    if not isinstance(path, str) or not Path(path).is_absolute():
                        raise PluginUiRpcInvalidRequest("请选择执行主机上的绝对目录")
                    if await run_file_io(lambda: projects.directory(project_id)) == path:
                        return await run_file_io(lambda: projects.bind_directory(project_id, path))
                    try:
                        checked = await directories.check_directory(path)
                    except ValueError as error:
                        raise PluginUiRpcInvalidRequest(str(error)) from error
                    return await run_file_io(lambda: projects.bind_directory(project_id, checked))
                if method == "project.directory" and set(payload) == {"project_id"}:
                    project_id = _project_id(payload)
                    path = await run_file_io(lambda: projects.directory(project_id))
                    return dict(await directories.inspect(path))
                if method == "directory.browse" and not set(payload) - {"path", "after"}:
                    path, after = payload.get("path", "~"), payload.get("after")
                    if not isinstance(path, str) or not path or (after is not None and not isinstance(after, str)):
                        raise PluginUiRpcInvalidRequest("目录路径或翻页参数无效")
                    return dict(await directories.browse(path, after=after))
        raise PluginUiRpcInvalidRequest(f"不支持的项目查询：{method}")

    _ = await ctx.require(UI_SLOTS).register_plugin_ui(
        ctx, PluginUiDefinition(module="plugin_ui.js"), query=query,
    )
