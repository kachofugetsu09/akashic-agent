"""原子保存当前插件选择；不保存历史依赖图、代码或配置闭包。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import tempfile
from collections.abc import Mapping
from typing import Literal, cast

from agent.plugins.files import sync_directory
from session.message import freeze_json
from session.message_codec import json_value
from agent.plugins.static_manifest import check_entrypoints


class SelectionFormatError(ValueError):
    """选择缺失或格式损坏，需要显式初始化、升级或恢复。"""


class SelectionConflictError(RuntimeError):
    """提交基线已经变化，本次候选不能发布。"""


class SelectionWriteError(RuntimeError):
    """写入失败；当前可读值不等于已经确认刷盘的提交结果。"""

    def __init__(self, *, operation: str, target_ref: str | None,
                 outcome: Literal["unchanged", "uncertain"], observed_ref: str | None,
                 observation_error: BaseException | None) -> None:
        super().__init__(f"stable {operation} 写入失败: outcome={outcome}, observed={observed_ref}")
        self.operation, self.target_ref, self.outcome = operation, target_ref, outcome
        self.observed_ref, self.observation_error = observed_ref, observation_error


class PluginSelection:
    """单 writer 在内存准备输入，提交时替换唯一当前选择文件。"""

    def __init__(self, workspace: Path) -> None:
        self.path = workspace / "runtime/plugin-stable.json"
        self._inputs: dict[str, Mapping[str, object]] = {}
        self._current: Mapping[str, object] | None = None

    def initialize(self) -> None:
        """显式创建空选择，不扫描业务数据或猜测旧格式。"""
        self._check_path()
        if self.path.exists():
            raise SelectionFormatError("stable 已存在，初始化不得覆盖")
        self._write({"version": 2, "root_ref": None, "inputs": {},
                     "distribution_adoption": None, "transition": None}, initialize=True)

    def read(self) -> str | None:
        """读取当前输入；旧归档指针只允许经过离线显式升级。"""
        self._check_path()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise SelectionFormatError("stable 缺失；需要显式初始化或升级") from error
        except (UnicodeError, json.JSONDecodeError) as error:
            raise SelectionFormatError("stable 无法解析") from error
        if not isinstance(raw, dict) or type(raw.get("version")) is not int or raw["version"] != 2:
            raise SelectionFormatError("stable 版本不支持；请离线执行 upgrade_plugin_selection.py --from-archive")
        if set(raw) != {"version", "root_ref", "inputs", "distribution_adoption", "transition"}:
            raise SelectionFormatError("stable 当前选择格式无效")
        ref = _reference(raw["root_ref"], nullable=True)
        inputs = raw["inputs"]
        if not isinstance(inputs, dict):
            raise SelectionFormatError("stable inputs 必须是对象")
        seen: set[str] = set()
        for identity, value in inputs.items():
            _reference(identity)
            record = _input(value)
            plugin_id = cast(str, record["plugin_id"])
            if plugin_id in seen:
                raise SelectionFormatError("stable 重复插件身份")
            seen.add(plugin_id)
            self._inputs[identity] = record
        adoption = raw["distribution_adoption"]
        if adoption is not None and not isinstance(adoption, dict):
            raise SelectionFormatError("发行版归属凭证必须是对象")
        transition = raw["transition"]
        if ref is None:
            if inputs or transition is not None or adoption is not None:
                raise SelectionFormatError("null 选择不能包含输入、提交或归属凭证")
        elif not isinstance(transition, dict) or set(transition) != {"base", "components"}:
            raise SelectionFormatError("当前选择缺少最后一次提交证据")
        else:
            _reference(transition["base"], nullable=True)
            if transition["components"] != list(inputs):
                raise SelectionFormatError("最后一次提交与当前输入不一致")
        self._current = cast(Mapping[str, object], freeze_json(raw))
        return ref

    def prepare(self, value: Mapping[str, object]) -> str:
        """候选仅存在于当前 owner 内存；提交失败不产生持久历史副本。"""
        record = _input(value)
        payload = json.dumps(json_value(record), sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode()
        identity = hashlib.sha256(payload).hexdigest()
        self._inputs[identity] = record
        return identity

    def read_input(self, identity: str) -> Mapping[str, object]:
        """读取当前或本次准备的输入，不从历史文件恢复。"""
        return self._inputs[identity]

    def components(self, expected_ref: str) -> tuple[str, ...]:
        """完整组合只能从当前选择读取，不能把旧 revision 当历史入口。"""
        if self.read() != expected_ref:
            raise SelectionConflictError("stable 基线已变化")
        assert self._current is not None
        return tuple(cast(Mapping[str, object], self._current["inputs"]))

    def adoption(self) -> Mapping[str, object] | None:
        """归属凭证是历史事实，随当前选择保留，但不加载历史代码。"""
        self.read()
        assert self._current is not None
        return cast(Mapping[str, object] | None, self._current["distribution_adoption"])

    def transition_committed(self, base: str | None, components: tuple[str, ...]) -> bool | None:
        """只凭最后一次原子提交认定恢复结果；更旧的未决事实明确未知。"""
        current = self.read()
        if current == base:
            return False
        assert self._current is not None
        transition = self._current["transition"]
        if isinstance(transition, Mapping) and transition["base"] == base:
            return transition["components"] == components
        return None

    def commit(self, components: tuple[str, ...], *, expected_ref: str | None,
               distribution_adoption: Mapping[str, object] | None = None) -> str:
        """CAS 提交完整当前组合，旧选择文件由新选择替换。"""
        # 1. 构造方持有唯一 writer，输入来自实际安装或本次准备。
        _reference(expected_ref, nullable=True)
        if self.read() != expected_ref:
            raise SelectionConflictError("stable 基线已变化")
        if not isinstance(components, tuple) or len(set(components)) != len(components):
            raise SelectionFormatError("完整选择必须是无重复输入的 tuple")
        inputs = {ref: self.read_input(ref) for ref in components}
        if len({value["plugin_id"] for value in inputs.values()}) != len(inputs):
            raise SelectionFormatError("完整选择不能重复包含插件身份")
        assert self._current is not None
        previous = cast(Mapping[str, object] | None, self._current["distribution_adoption"])
        if distribution_adoption is None:
            distribution_adoption = previous
        elif previous is not None and freeze_json(distribution_adoption) != previous:
            raise SelectionConflictError("历史归属已经转换，不能覆盖凭证")
        # 2. 仅当前状态和最后一次提交证据原子落盘，不保存前驱链。
        ref = secrets.token_hex(32)
        self._write({"version": 2, "root_ref": ref, "inputs": inputs,
                     "distribution_adoption": distribution_adoption,
                     "transition": {"base": expected_ref, "components": components}}, initialize=False)
        return ref

    def _check_path(self) -> None:
        if self.path.parent.is_symlink() or self.path.is_symlink():
            raise SelectionFormatError("stable 路径不能是符号链接")
        if self.path.exists() and not self.path.is_file():
            raise SelectionFormatError("stable 必须是普通文件")

    def _write(self, value: Mapping[str, object], *, initialize: bool) -> None:
        """同步完整文件后原子发布；发布后的失败不回写旧选择。"""
        replacing = False
        ref = cast(str | None, value["root_ref"])
        try:
            if initialize:
                self.path.parent.mkdir(mode=0o700, exist_ok=True)
                sync_directory(self.path.parent.parent)
            fd, name = tempfile.mkstemp(prefix=".plugin-stable-", dir=self.path.parent)
            temporary = Path(name)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(json_value(value), stream, ensure_ascii=False, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            replacing = True
            if initialize:
                os.link(temporary, self.path)
                temporary.unlink()
            else:
                os.replace(temporary, self.path)
            sync_directory(self.path.parent)
        except BaseException as error:
            observed, observation_error = None, None
            try:
                observed = self.read()
            except BaseException as failure:
                observation_error = failure
            raise SelectionWriteError(operation="initialize" if initialize else "commit", target_ref=ref,
                                      outcome="uncertain" if replacing else "unchanged",
                                      observed_ref=observed, observation_error=observation_error) from error


def _reference(value: object, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SelectionFormatError("选择身份必须是 64 位十六进制字符串")
    return value


def _input(value: object) -> Mapping[str, object]:
    """在选择文件边界核对安装元数据，不读代码或配置内容。"""
    fields = {"version", "code", "plugin_id", "source_revision", "config_revision",
              "python_environments", "source_type", "data_dir", "runtime"}
    if not isinstance(value, Mapping):
        raise SelectionFormatError("插件输入结构无效")
    version = value.get("version")
    if type(version) is not int or version not in {5, 6}:
        raise SelectionFormatError("插件输入版本无效")
    if version == 6:
        fields.add("entrypoints")
    if set(value) not in (fields, fields | {"distribution_source"}):
        raise SelectionFormatError("插件输入结构无效")
    if value["source_type"] not in {"builtin", "installed"}:
        raise SelectionFormatError("插件输入版本或来源类型无效")
    if version == 6:
        try:
            check_entrypoints(value["entrypoints"])
        except ValueError as error:
            raise SelectionFormatError("插件命令输入无效") from error
    for key in ("code", "plugin_id", "data_dir"):
        if not isinstance(value[key], str) or not value[key]:
            raise SelectionFormatError(f"插件输入缺少 {key}")
    plugin_id = cast(str, value["plugin_id"])
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*(?:@[A-Za-z0-9][A-Za-z0-9._-]*)?", plugin_id) is None:
        raise SelectionFormatError("插件输入身份无效")
    name, _, marketplace = plugin_id.partition("@")
    if value["source_type"] == "installed" and not marketplace:
        raise SelectionFormatError("已安装来源缺少 marketplace")
    if not Path(cast(str, value["code"])).is_absolute():
        raise SelectionFormatError("插件代码路径必须是绝对路径")
    data = Path(cast(str, value["data_dir"]))
    if data.parts != ("plugin-data", f"{name}-{marketplace or 'builtin'}"):
        raise SelectionFormatError("插件数据路径与来源身份不一致")
    for key in ("source_revision", "config_revision"):
        _reference(value[key])
    runtime, environments = value["runtime"], value["python_environments"]
    if (not isinstance(runtime, Mapping) or set(runtime) != {"python_tag", "binding_api"}
        or not isinstance(runtime["python_tag"], str) or type(runtime["binding_api"]) is not int
        or not isinstance(environments, Mapping)):
        raise SelectionFormatError("插件运行环境元数据无效")
    for key, ref in environments.items():
        if (not isinstance(key, str) or not key or Path(key).is_absolute() or ".." in Path(key).parts
            or not isinstance(ref, str) or re.fullmatch(r"(?:[0-9a-f]{32}|[0-9a-f]{64})", ref) is None):
            raise SelectionFormatError("插件环境引用无效")
    return cast(Mapping[str, object], freeze_json(dict(value)))
