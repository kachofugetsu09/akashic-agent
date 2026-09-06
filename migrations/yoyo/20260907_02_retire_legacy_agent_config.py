from __future__ import annotations

import hashlib
import json
import os
import stat
import tomllib
from pathlib import Path
from typing import Mapping
from uuid import uuid4

import tomlkit
from yoyo import step

from agent.migrations.context import current_migration_context
from agent.plugins.manifest import builtin_plugin_data_dir, validate_workspace_plugin_data_path

__depends__ = {"20260907_01_context_material_grants"}
__transactional__ = False

_MIGRATION = "retire-legacy-agent-config"
_DEFAULT_SYSTEM_PROMPT = (
    "You are Akashic, a helpful AI assistant with access to tools. "
    "Always respond in the same language the user uses."
)
_DEFAULT_TOOLSETS = ("meta_common",)


class _Snapshot:
    """保存文件字节、权限和软链身份，供双文件发布和恢复使用。"""

    def __init__(
        self,
        path: Path,
        target: Path,
        content: bytes | None,
        mode: int | None,
        symlink_target: str | None,
    ) -> None:
        self.path = path
        self.target = target
        self.content = content
        self.mode = mode
        self.symlink_target = symlink_target


class _Plan:
    def __init__(
        self,
        config: _Snapshot,
        reply: _Snapshot,
        config_bytes: bytes | None,
        reply_bytes: bytes | None,
    ) -> None:
        self.config = config
        self.reply = reply
        self.config_bytes = config_bytes
        self.reply_bytes = reply_bytes


def _snapshot(path: Path, *, label: str) -> _Snapshot:
    """读取普通文件或软链目标，并拒绝目录和悬空软链。"""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return _Snapshot(path, path, None, None, None)
    except OSError as error:
        raise RuntimeError(f"无法读取 {label} 元数据: {path}") from error
    if stat.S_ISLNK(metadata.st_mode):
        link = os.readlink(path)
        try:
            target = path.resolve(strict=True)
            target_metadata = target.stat()
        except OSError as error:
            raise RuntimeError(f"{label} 软链接目标不可读: {path}") from error
        if not stat.S_ISREG(target_metadata.st_mode):
            raise RuntimeError(f"{label} 软链接目标必须是普通文件: {path}")
        return _Snapshot(
            path, target, path.read_bytes(), stat.S_IMODE(target_metadata.st_mode), link
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{label} 必须是普通文件或软链接: {path}")
    return _Snapshot(path, path, path.read_bytes(), stat.S_IMODE(metadata.st_mode), None)


def _same_snapshot(left: _Snapshot, right: _Snapshot) -> bool:
    """比较文件存在性、类型、字节、权限和软链身份。"""

    return (
        left.content == right.content
        and left.mode == right.mode
        and left.target == right.target
        and left.symlink_target == right.symlink_target
    )


def _check_snapshot(snapshot: _Snapshot, *, label: str) -> None:
    """确认发布边界仍等于迁移预检时的完整快照。"""

    current = _snapshot(snapshot.path, label=label)
    if not _same_snapshot(snapshot, current):
        raise RuntimeError(f"{label} 迁移期间发生变化: {snapshot.path}")


def _matches_published(
    snapshot: _Snapshot,
    current: _Snapshot,
    payload: bytes,
) -> bool:
    """确认目标仍是本次迁移刚发布的字节和文件身份。"""

    return (
        current.content == payload
        and current.mode == (snapshot.mode if snapshot.mode is not None else 0o600)
        and current.target == snapshot.target
        and current.symlink_target == snapshot.symlink_target
    )


def _same_existing_target(left: Path, right: Path) -> bool:
    """拒绝两个既存配置目标共享同一 inode。"""

    try:
        left_stat = left.stat()
        right_stat = right.stat()
    except FileNotFoundError:
        return False
    if (left_stat.st_dev, left_stat.st_ino) == (right_stat.st_dev, right_stat.st_ino):
        return True
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_atomic(path: Path, payload: bytes, mode: int) -> None:
    """在同一目录完成 fsync 后原子替换。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _backup(snapshot: _Snapshot, root: Path, name: str) -> str | None:
    if snapshot.content is None:
        return None
    target = root / name
    _write_atomic(target, snapshot.content, 0o600)
    if target.read_bytes() != snapshot.content:
        raise RuntimeError(f"配置备份校验失败: {snapshot.path}")
    return target.name


def _write_manifest(
    root: Path,
    config: _Snapshot,
    reply: _Snapshot,
    config_backup: str | None,
    reply_backup: str | None,
) -> None:
    manifest = {
        "schema_version": 1,
        "migration": _MIGRATION,
        "sources": {
            "config": {
                "path": str(config.path),
                "kind": "symlink" if config.symlink_target is not None else "file",
                "target": str(config.target),
                "symlink_target": config.symlink_target,
                "mode": config.mode,
                "backup": config_backup,
                "sha256": None if config.content is None else _digest(config.content),
            },
            "reply": {
                "path": str(reply.path),
                "kind": (
                    "absent"
                    if reply.content is None
                    else "symlink" if reply.symlink_target is not None else "file"
                ),
                "target": str(reply.target),
                "symlink_target": reply.symlink_target,
                "mode": reply.mode,
                "backup": reply_backup,
                "sha256": None if reply.content is None else _digest(reply.content),
            },
        },
    }
    rendered = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    _write_atomic(root / "manifest.json", rendered, 0o600)
    if json.loads((root / "manifest.json").read_text(encoding="utf-8")) != manifest:
        raise RuntimeError(f"配置备份 manifest 校验失败: {root}")


def _strict_nonnegative(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise RuntimeError(f"{field} 必须是非负整数；请修复后重试")
    return value


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{field} 必须是 TOML table；原配置保持不变")
    return value


def _legacy_values(data: Mapping[str, object]) -> int | None:
    agent = _mapping(data.get("agent", {}), "agent")
    values: dict[str, object] = {}
    if "max_iterations" in data:
        values["max_iterations"] = data["max_iterations"]
    if "max_iterations" in agent:
        values["agent.max_iterations"] = agent["max_iterations"]
    parsed = {
        field: _strict_nonnegative(value, field) for field, value in values.items()
    }
    unique = set(parsed.values())
    if len(unique) > 1:
        raise RuntimeError("max_iterations 在顶层与 agent 中冲突；原配置保持不变")
    max_steps = next(iter(unique), None)
    return max_steps


def _check_safe_defaults(data: Mapping[str, object], agent: Mapping[str, object]) -> None:
    prompt_values = []
    if "system_prompt" in data:
        prompt_values.append(("system_prompt", data["system_prompt"]))
    if "system_prompt" in agent:
        prompt_values.append(("agent.system_prompt", agent["system_prompt"]))
    if prompt_values:
        if any(value != _DEFAULT_SYSTEM_PROMPT for _, value in prompt_values):
            raise RuntimeError(
                "自定义 system_prompt 与唯一 memory/VEDA.md 人格 owner 冲突；"
                "原配置保持不变，不覆盖或拼接 VEDA"
            )

    dev_values = []
    for field, container in (("dev_mode", data), ("dev_model", data),
                             ("agent.dev_mode", agent), ("agent.dev_model", agent)):
        key = field.rsplit(".", 1)[-1]
        if key in container:
            value = container[key]
            if type(value) is not bool:
                raise RuntimeError(f"{field} 必须是布尔值；原配置保持不变")
            dev_values.append((field, value))
    if any(value for _, value in dev_values):
        raise RuntimeError("dev_mode=true 没有当前 owner；原配置保持不变")

    wiring_values = []
    for field, container in (("wiring", data), ("agent.wiring", agent)):
        if field.rsplit(".", 1)[-1] not in container:
            continue
        wiring = _mapping(container[field.rsplit(".", 1)[-1]], field)
        toolsets = wiring.get("toolsets")
        if not wiring or (
            set(wiring) == {"toolsets"}
            and isinstance(toolsets, list)
            and tuple(toolsets) == _DEFAULT_TOOLSETS
        ) or (
            set(wiring) == {"context"}
            and wiring.get("context") == "default"
        ) or (
            set(wiring) == {"context", "toolsets"}
            and wiring.get("context") == "default"
            and isinstance(toolsets, list)
            and tuple(toolsets) == _DEFAULT_TOOLSETS
        ):
            wiring_values.append(field)
        else:
            raise RuntimeError(f"{field} 有自定义值且没有当前 owner；原配置保持不变")

    search_fields = []
    if "tool_search_enabled" in data:
        search_fields.append("tool_search_enabled")
    tools = agent.get("tools")
    if isinstance(tools, Mapping) and "search_enabled" in tools:
        search_fields.append("agent.tools.search_enabled")
    if search_fields:
        raise RuntimeError(
            "旧工具搜索布尔值无法无损映射: "
            + ", ".join(search_fields)
            + "; reply.tools 只接受插件明确的工具名，工具 discovery 负责候选与选择；"
            "请由这两个 owner 决定后再移除旧布尔值，原配置保持不变"
        )


def _render_config(content: bytes, max_steps: int | None) -> bytes | None:
    """只移除已证明默认值和旧预算键，保留其他用户配置与格式。"""

    parsed = tomllib.loads(content.decode("utf-8"))
    agent = _mapping(parsed.get("agent", {}), "agent")
    _check_safe_defaults(parsed, agent)
    discovered = _legacy_values(parsed)
    document = tomlkit.parse(content.decode("utf-8"))
    changed = False
    for key in ("system_prompt", "max_iterations", "dev_mode", "dev_model"):
        if key in document:
            del document[key]
            changed = True
    agent_document = document.get("agent")
    if isinstance(agent_document, dict):
        for key in ("system_prompt", "max_iterations", "dev_mode", "dev_model"):
            if key in agent_document:
                del agent_document[key]
                changed = True
        wiring = agent_document.get("wiring")
        if wiring is not None:
            del agent_document["wiring"]
            changed = True
        if not agent_document:
            del document["agent"]
            changed = True
    if "wiring" in document:
        del document["wiring"]
        changed = True
    if discovered != max_steps:
        raise RuntimeError("max_iterations 预检结果不一致；原配置保持不变")
    if not changed:
        return None
    rendered = tomlkit.dumps(document).encode("utf-8")
    final = tomllib.loads(rendered.decode("utf-8"))
    if any(key in final for key in ("system_prompt", "max_iterations", "dev_mode", "dev_model", "wiring")):
        raise RuntimeError("配置迁移后仍含旧全局字段")
    final_agent = final.get("agent")
    if isinstance(final_agent, Mapping) and any(
        key in final_agent
        for key in ("system_prompt", "max_iterations", "dev_mode", "dev_model", "wiring")
    ):
        raise RuntimeError("配置迁移后仍含旧 agent 字段")
    return rendered


def _reply_config(path: Path, max_steps: int | None, workspace: Path) -> tuple[_Snapshot, bytes | None]:
    validate_workspace_plugin_data_path(path, workspace)
    snapshot = _snapshot(path, label="reply 插件配置")
    if max_steps is None:
        return snapshot, None
    if snapshot.content is None:
        return snapshot, f"max_steps = {max_steps}\n".encode()
    try:
        raw = tomllib.loads(snapshot.content.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(f"reply 插件配置不是合法 TOML；原配置保持不变: {path}") from error
    existing = raw.get("max_steps")
    if existing is not None:
        current = _strict_nonnegative(existing, "reply.max_steps")
        if current != max_steps:
            raise RuntimeError(
                f"reply.max_steps={current} 与旧 max_iterations={max_steps} 冲突；不能覆盖"
            )
        return snapshot, None
    document = tomlkit.parse(snapshot.content.decode("utf-8"))
    document["max_steps"] = max_steps
    rendered = tomlkit.dumps(document).encode("utf-8")
    final = tomllib.loads(rendered.decode("utf-8"))
    if final.get("max_steps") != max_steps:
        raise RuntimeError("reply 配置迁移后 max_steps 校验失败")
    return snapshot, rendered


def _plan(config_path: Path, workspace: Path) -> _Plan | None:
    config = _snapshot(config_path, label="主配置")
    if config.content is None:
        return None
    try:
        data = tomllib.loads(config.content.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(f"主配置不是合法 TOML；原配置保持不变: {config_path}") from error
    if not isinstance(data, Mapping):
        raise RuntimeError("主配置必须是 TOML table；原配置保持不变")
    max_steps = _legacy_values(data)
    config_bytes = _render_config(config.content, max_steps)
    reply_path = builtin_plugin_data_dir("reply", workspace) / "config.local.toml"
    reply, reply_bytes = _reply_config(reply_path, max_steps, workspace)
    if reply.content is not None and reply.target == config.target:
        raise RuntimeError("主配置与 reply 插件配置解析到同一文件；拒绝双重发布")
    if reply.content is not None and _same_existing_target(reply.target, config.target):
        raise RuntimeError("主配置与 reply 插件配置共享同一 inode；拒绝双重发布")
    if config_bytes is None and reply_bytes is None:
        return None
    return _Plan(config, reply, config_bytes, reply_bytes)


def _publish(snapshot: _Snapshot, payload: bytes | None, *, label: str) -> None:
    if payload is None:
        return
    _check_snapshot(snapshot, label=label)
    target = snapshot.target
    mode = snapshot.mode if snapshot.mode is not None else 0o600
    _write_atomic(target, payload, mode)
    current = _snapshot(snapshot.path, label=label)
    if not _matches_published(snapshot, current, payload):
        raise RuntimeError(f"{label} 发布校验失败: {snapshot.path}")


def _restore(snapshot: _Snapshot, payload: bytes | None, *, label: str) -> None:
    if payload is None:
        return

    current = _snapshot(snapshot.path, label=label)
    if _same_snapshot(snapshot, current):
        return
    if not _matches_published(snapshot, current, payload):
        raise RuntimeError(f"{label} 发布目标已漂移，拒绝恢复: {snapshot.path}")
    if snapshot.content is None:
        snapshot.path.unlink()
        _fsync_directory(snapshot.path.parent)
    else:
        _write_atomic(snapshot.target, snapshot.content, snapshot.mode or 0o600)
    _check_snapshot(snapshot, label=label)
    if snapshot.content is not None and snapshot.path.read_bytes() != snapshot.content:
        raise RuntimeError(f"{label} 恢复校验失败: {snapshot.path}")


def retire_legacy_agent_config(_connection: object) -> None:
    """把旧全局预算交给 reply，并清除已证明无意图的默认字段。"""

    _ = _connection
    current = current_migration_context()
    plan = _plan(current.config_path, current.workspace)
    if plan is None:
        return
    backup_root = current.workspace / "backups" / _MIGRATION / uuid4().hex
    backup_root.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(backup_root, 0o700)
    config_backup = _backup(plan.config, backup_root, "config.toml.before")
    reply_backup = _backup(plan.reply, backup_root, "reply-config.local.toml.before")
    _write_manifest(backup_root, plan.config, plan.reply, config_backup, reply_backup)
    attempted: list[tuple[_Snapshot, bytes | None, str]] = []
    try:
        # 1. 所有冲突都已在第一次外部写入前检查。
        attempted.append((plan.reply, plan.reply_bytes, "reply 插件配置"))
        validate_workspace_plugin_data_path(plan.reply.path, current.workspace)
        _publish(plan.reply, plan.reply_bytes, label="reply 插件配置")
        attempted.append((plan.config, plan.config_bytes, "主配置"))
        _publish(plan.config, plan.config_bytes, label="主配置")
    except BaseException as migration_error:
        restore_error: BaseException | None = None
        for snapshot, payload, label in reversed(attempted):
            try:
                if label == "reply 插件配置":
                    validate_workspace_plugin_data_path(snapshot.path, current.workspace)
                _restore(snapshot, payload, label=label)
            except BaseException as error:
                if restore_error is None:
                    restore_error = error
        if restore_error is not None:
            raise RuntimeError(
                f"旧 Agent 配置迁移失败且恢复失败: {migration_error}; "
                f"请从 {backup_root} 恢复"
            ) from restore_error
        raise


steps = [step(retire_legacy_agent_config)]
