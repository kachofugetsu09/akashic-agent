"""唯一 workspace stable 指针；完整组件与历史记录由 PluginArchive 保存。"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Literal

from agent.plugin_composition.archive import PluginArchive, sync_directory


class SelectionFormatError(ValueError):
    """选择缺失或格式损坏，需要显式初始化、升级或恢复。"""


class SelectionConflictError(RuntimeError):
    """提交基线已经变化，本次候选不能发布。"""


class SelectionWriteError(RuntimeError):
    """写入失败；当前可读值不等于已经确认刷盘的提交结果。"""

    def __init__(
        self, *, operation: str, target_ref: str | None,
        outcome: Literal["unchanged", "uncertain"], observed_ref: str | None,
        observation_error: BaseException | None,
    ) -> None:
        super().__init__(f"stable {operation} 写入失败: outcome={outcome}, observed={observed_ref}")
        self.operation = operation
        self.target_ref = target_ref
        self.outcome = outcome
        self.observed_ref = observed_ref
        self.observation_error = observation_error


class PluginSelection:
    """调用者持有 workspace 单 writer 锁；此对象不启动或选择候选实例。"""

    def __init__(self, workspace: Path) -> None:
        self.path = workspace / "runtime" / "plugin-stable.json"
        self.archive = PluginArchive(workspace / "runtime" / "plugin-archives", create=False)

    def initialize(self) -> None:
        """显式新建空选择；调用者证明是新 workspace 或已批准的升级。"""
        self._check_path()
        if self.path.exists():
            raise SelectionFormatError("stable 已存在，初始化不得覆盖")
        # 不扫描业务数据，也不把历史目录猜成可初始化状态。
        self._write(None, initialize=True)

    def read(self) -> str | None:
        """只读当前完整选择；只有明确的 null 表示尚未首次提交。"""
        self._check_path()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise SelectionFormatError("stable 缺失；需要显式初始化或升级") from error
        except (UnicodeError, json.JSONDecodeError) as error:
            raise SelectionFormatError("stable 指针无法解析") from error
        if not isinstance(raw, dict) or set(raw) != {"version", "root_ref"}:
            raise SelectionFormatError("stable 指针格式无效")
        if type(raw["version"]) is not int or raw["version"] != 1:
            raise SelectionFormatError("stable 指针版本不支持；需要显式升级")
        ref = _reference(raw["root_ref"], nullable=True)
        if ref is not None:
            self._read_record(ref)
        return ref

    def commit(self, components: tuple[str, ...], *, expected_ref: str | None) -> str:
        """提交构造方保证完整的正式输入；不接受任意 binding 子集充当完整组合。"""
        # 1. 基线是前次提交记录，而不是可重复出现的组件集合。
        expected_ref = _reference(expected_ref, nullable=True)
        if self.read() != expected_ref:
            raise SelectionConflictError("stable 基线已变化")
        if not isinstance(components, tuple):
            raise TypeError("components 必须是完整输入引用的 tuple")
        for ref in components:
            _reference(ref)
            self.archive.read_descriptor(ref)
        if len(set(components)) != len(components):
            raise SelectionFormatError("完整选择不能重复包含组件引用")

        # 2. 只保存引用和前驱，不复制身份、配置或环境路径。
        try:
            self.archive = PluginArchive(self.archive.path)
            ref = self.archive.save_descriptor({
                "version": 1, "components": list(components), "previous": expected_ref,
            })
        except BaseException as error:
            raise self._write_error("commit", None, replacing=False) from error
        if self.read() != expected_ref:
            raise SelectionConflictError("保存记录期间 stable 基线已变化")
        # 3. 记录先耐久，再提交唯一指针；失败不得回写旧选择。
        self._write(ref, initialize=False)
        return ref

    def _read_record(self, ref: str) -> None:
        try:
            record = self.archive.read_descriptor(ref)
        except (ValueError, RuntimeError, FileNotFoundError) as error:
            raise SelectionFormatError("stable 完整记录缺失或损坏") from error
        if set(record) != {"version", "components", "previous"}:
            raise SelectionFormatError("stable 不是完整选择记录")
        if type(record["version"]) is not int or record["version"] != 1:
            raise SelectionFormatError("stable 记录版本不支持")
        components = record["components"]
        if not isinstance(components, tuple):
            raise SelectionFormatError("stable components 格式无效")
        for component in components:
            _reference(component)
        if len(set(components)) != len(components):
            raise SelectionFormatError("stable components 重复")
        _reference(record["previous"], nullable=True)

    def _check_path(self) -> None:
        if self.path.parent.is_symlink() or self.path.is_symlink():
            raise SelectionFormatError("stable 路径不能是符号链接")
        if self.path.exists() and not self.path.is_file():
            raise SelectionFormatError("stable 必须是普通文件")

    def _write(self, ref: str | None, *, initialize: bool) -> None:
        """同步临时文件后发布；发布尝试之后的异常一律保留不确定结果。"""
        replacing = False
        try:
            # 1. 只有显式初始化创建目录，并同步父目录中的新目录项。
            if initialize:
                self.path.parent.mkdir(mode=0o700, exist_ok=True)
                sync_directory(self.path.parent.parent)
            fd, name = tempfile.mkstemp(prefix=".plugin-stable-", dir=self.path.parent)
            temporary = Path(name)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"version": 1, "root_ref": ref}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            # 2. 初始化用无覆盖发布；提交在单 writer 锁内原子替换。
            replacing = True
            if initialize:
                os.link(temporary, self.path)
                temporary.unlink()
            else:
                os.replace(temporary, self.path)
            sync_directory(self.path.parent)
        except BaseException as error:
            # 失败临时文件保留供诊断；不自动清理或回退已发布选择。
            raise self._write_error(
                "initialize" if initialize else "commit", ref, replacing=replacing,
            ) from error

    def _write_error(self, operation: str, ref: str | None, *, replacing: bool) -> SelectionWriteError:
        """读取异常后的可见指针；读失败与明确的 null 分开报告。"""
        observed = None
        observation_error = None
        try:
            observed = self.read()
        except BaseException as error:
            observation_error = error
        return SelectionWriteError(
            operation=operation, target_ref=ref,
            outcome="uncertain" if replacing else "unchanged",
            observed_ref=observed, observation_error=observation_error,
        )


def _reference(value: object, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SelectionFormatError("选择引用必须是 SHA-256")
    return value
