"""核对 Gateway 固定输入并保留恢复点后，退休 Core 的控制配置表。"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import tomllib

import tomlkit
from yoyo import step
from agent.migrations.context import current_migration_context
from agent.plugin_composition.config_input import CONFIG_INPUT, load_config
from .helpers.settings import GatewayConfig

__depends__ = {"20261010_01_gateway_config_copy"}
__transactional__ = False
_BACKUP_SUFFIX = ".before-gateway-config-migration.bak"


def retire_config(connection: object) -> None:
    """只减少已经完整复制的配置表，不修改插件输入或任何业务数据库。"""
    context = current_migration_context()
    source = context.config_path
    if source.is_symlink() or not source.is_file():
        raise ValueError("Gateway 配置源必须是普通文件")
    before = source.read_bytes()
    raw = tomllib.loads(before.decode("utf-8")).get("app_server")
    if raw is None:
        return
    if not isinstance(raw, dict):
        raise ValueError("app_server 必须是 TOML table")

    # 1. 复制 step 已提交；目标缺席或不一致不能借默认值退休源表。
    data_root = context.bundle_data_roots["gateway_config"]
    if not (data_root / CONFIG_INPUT).is_file():
        raise ValueError("Gateway 固定输入缺席，保留旧 app_server")
    current, _ = load_config(data_root)
    if GatewayConfig.model_validate(current).model_dump() != GatewayConfig.from_legacy(raw).model_dump():
        raise ValueError("Gateway 固定输入与旧 app_server 不同，保留源表")
    document = tomlkit.parse(before.decode("utf-8"))
    del document["app_server"]
    after = tomlkit.dumps(document).encode("utf-8")

    # 2. 原始字节先刷盘；已有恢复点只能复用，不能覆盖。
    backup = source.with_name(source.name + _BACKUP_SUFFIX)
    if backup.exists() or backup.is_symlink():
        if backup.is_symlink() or not backup.is_file() or backup.read_bytes() != before:
            raise FileExistsError(f"Gateway 配置恢复点不同，拒绝覆盖: {backup}")
    else:
        descriptor = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(before)
            stream.flush()
            os.fsync(stream.fileno())
    _sync_directory(source.parent)

    # 3. 保留其余表、注释和源权限；失败时不假称外部文件已经回滚。
    if source.read_bytes() != before:
        raise RuntimeError("Gateway 迁移期间源配置已变化，拒绝替换")
    descriptor, name = tempfile.mkstemp(prefix=f".{source.name}.", dir=source.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.chmod(temporary, source.stat().st_mode & 0o777)
            stream.write(after)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, source)
        _sync_directory(source.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _sync_directory(path: Path) -> None:
    """恢复点和替换文件名刷盘后，Yoyo 才能记录完成。"""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


steps = [step(retire_config)]
