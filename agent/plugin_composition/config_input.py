"""显式配置命令与宿主共用的固定输入和私有凭据边界。"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from uuid import uuid4

from agent.plugin_composition.archive import decode_config, encode_config, sync_directory
from agent.plugin_composition.channels import CredentialRef

CONFIG_INPUT = "config.input.json"
_LEGACY = "config.local.toml"


def _data_dir(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if path.absolute() != resolved or resolved.parent.name != "plugin-data":
        raise ValueError("固定配置必须位于无符号链接的 workspace/plugin-data/<owner>")
    return resolved


def _private_root(data_dir: Path) -> Path:
    root = _data_dir(data_dir).parent.parent / ".plugin-credentials" / data_dir.name
    if root.resolve(strict=False) != root:
        raise ValueError("私有凭据目录不能包含符号链接")
    return root


def _private_directory(path: Path) -> None:
    boundary = next(parent for parent in (path, *path.parents) if parent.name == ".plugin-credentials")
    boundary.parent.mkdir(parents=True, exist_ok=True)
    directories = [boundary]
    current = boundary
    for part in path.relative_to(boundary).parts:
        current /= part
        directories.append(current)
    for directory in directories:
        directory.mkdir(exist_ok=True, mode=0o700)
        sync_directory(directory.parent)
        if directory.is_symlink() or directory.stat().st_mode & 0o077:
            raise PermissionError("私有配置目录必须是权限 0700 的实际目录")


def _legacy_files(data_dir: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in data_dir.rglob("*") if _LEGACY in path.name))


def check_config_format(data_dir: Path) -> None:
    """只核对固定配置入口；插件业务数据不参与格式判断。"""
    data_dir = _data_dir(data_dir)
    legacy = data_dir / _LEGACY
    if legacy.exists() or legacy.is_symlink():
        raise RuntimeError("发现 config.local.toml；须离线显式升级插件配置")
    path = data_dir / CONFIG_INPUT
    if path.is_symlink():
        raise ValueError("固定配置输入不能是符号链接")


def load_config(data_dir: Path) -> tuple[dict[str, object], str]:
    """从同一次读取还原固定映射和版本，不解释插件字段或解析凭据。"""
    check_config_format(data_dir)
    path = _data_dir(data_dir) / CONFIG_INPUT
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return {}, hashlib.sha256(b"<missing>").hexdigest()
    raw = json.loads(content)
    if not isinstance(raw, dict) or set(raw) != {"version", "config"} or type(raw["version"]) is not int or raw["version"] != 1:
        raise ValueError("固定配置输入格式错误；须显式升级")
    config = decode_config(raw["config"])
    if not isinstance(config, dict) or not all(isinstance(key, str) for key in config):
        raise ValueError("固定配置输入必须是字符串键映射")
    return config, hashlib.sha256(content).hexdigest()


def config_refs(config: object) -> frozenset[CredentialRef]:
    """仅收集已编码引用作为授权，不从插件字段名推断秘密。"""
    if isinstance(config, CredentialRef):
        return frozenset({config})
    if isinstance(config, Mapping):
        values = config.values()
    elif isinstance(config, (tuple, list)):
        values = config
    else:
        return frozenset()
    return frozenset(ref for value in values for ref in config_refs(value))


def _config_bytes(config: Mapping[str, object]) -> bytes:
    if not isinstance(config, Mapping) or not all(isinstance(key, str) for key in config):
        raise TypeError("固定配置输入必须是字符串键映射")
    return json.dumps({"version": 1, "config": encode_config(config)},
                      ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")


def _write(path: Path, content: bytes, *, replace: bool = False) -> None:
    """先刷盘，再发布完整文件；失败只清理本次临时文件。"""
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary_path, path)
        else:
            os.link(temporary_path, path)
        sync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_config(data_dir: Path, config: Mapping[str, object]) -> None:
    """显式发布插件解释后的输入；旧输入在私有目录保留独立恢复点。"""
    # 1. 编码和格式核对先于持久化，旧 TOML 只能经过显式升级入口。
    data_dir = _data_dir(data_dir)
    check_config_format(data_dir)
    content = _config_bytes(config)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / CONFIG_INPUT
    if path.is_symlink():
        raise ValueError("固定配置输入不能是符号链接")
    # 2. 备份与新输入分别刷盘；旧引用授权在新配置发布后失效。
    if path.exists():
        backup = _private_root(data_dir) / "config-history"
        _private_directory(backup)
        _write(backup / f"{uuid4().hex}.json", path.read_bytes())
    _write(path, content, replace=True)


def save_credential(data_dir: Path, value: str) -> CredentialRef:
    """显式保存不可变凭据版本；明文不进入 plugin-data 或配置归档。"""
    if not isinstance(value, str) or not value:
        raise ValueError("凭据必须是非空字符串")
    root = _private_root(data_dir)
    _private_directory(root)
    identity = uuid4().hex
    content = json.dumps({"owner": data_dir.name, "id": identity, "value": value},
                         ensure_ascii=False, sort_keys=True).encode("utf-8")
    revision = hashlib.sha256(content).hexdigest()
    _write(root / f"{identity}.json", content)
    return CredentialRef((identity, revision))


def _credential_path(data_dir: Path, ref: CredentialRef) -> Path:
    if len(ref.path) != 2 or re.fullmatch(r"[0-9a-f]{32}", ref.path[0]) is None or re.fullmatch(r"[0-9a-f]{64}", ref.path[1]) is None:
        raise ValueError("凭据引用不是固定版本；须显式升级")
    return _private_root(data_dir) / f"{ref.path[0]}.json"


def revoke_credential(data_dir: Path, ref: CredentialRef) -> None:
    """追加撤销标记，保留原凭据作为受保护恢复材料。"""
    path = _credential_path(data_dir, ref)
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != ref.path[1]:
        raise RuntimeError("凭据固定版本内容已漂移")
    _write(path.with_suffix(".revoked"), b"revoked\n")


def upgrade_config(data_dir: Path, convert: Callable[[bytes], Mapping[str, object]]) -> Path:
    """离线升级单个插件；转换函数由配置程序提供，所有原件保留。"""
    # 1. 先备份全部旧配置与命名备份，转换失败不改变原件。
    data_dir = _data_dir(data_dir)
    source = data_dir / _LEGACY
    legacy = _legacy_files(data_dir)
    if any(path.is_symlink() or not path.is_file() for path in legacy):
        raise ValueError("旧配置升级只接受实际文件")
    content = source.read_bytes()
    input_path = data_dir / CONFIG_INPUT
    if input_path.exists() or input_path.is_symlink():
        raise FileExistsError("已有固定配置输入；请用升级恢复点人工核对，不能重复转换")
    backup = _private_root(data_dir) / "upgrades" / uuid4().hex
    _private_directory(backup)
    for path in legacy:
        target = backup / "original" / path.relative_to(data_dir)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _write(target, path.read_bytes())
    if (backup / "original" / _LEGACY).read_bytes() != content:
        raise RuntimeError("备份期间旧配置变化；原件与恢复点均保留")
    # 2. 配置程序解释字段并保存私有凭据，传回只含引用的输入。
    encoded = _config_bytes(convert(content))
    _write(input_path, encoded)
    # 3. 顶层旧入口最后退役；此前中断仍会明确要求完成升级。
    for old in sorted(legacy, key=lambda path: path == source):
        original = backup / "original" / old.relative_to(data_dir)
        if old.read_bytes() != original.read_bytes():
            raise RuntimeError("升级期间旧配置变化；已停止，原件与恢复点均保留")
        target = backup / "retired" / old.relative_to(data_dir)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.rename(old, target)
        sync_directory(old.parent)
        sync_directory(target.parent)
    return backup


__all__ = ["CONFIG_INPUT", "load_config", "save_config", "save_credential", "revoke_credential", "upgrade_config"]
