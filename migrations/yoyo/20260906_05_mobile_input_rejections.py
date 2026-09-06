"""Mobile 失败输入保留回执到交接完成，附件拒绝保留逻辑终态。"""
from pathlib import Path
from typing import cast
import tomllib
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.migrations.mobile_input import migrate

__depends__ = {'20260906_04_channel_identities'}
__transactional__ = False


def migrate_mobile_inputs(_ledger: object) -> None:
    context = current_migration_context()
    config = tomllib.loads(context.config_path.read_text()) if context.config_path.is_file() else {}
    mobile = config.get('mobile_realtime', {})
    if not isinstance(mobile, dict):
        raise TypeError('mobile_realtime 配置不是 table')
    value = cast(dict[str, object], mobile).get('database', 'data/mobile_realtime.db')
    if not isinstance(value, str) or not value:
        raise TypeError('mobile_realtime.database 必须为非空路径')
    configured = Path(value)
    path = configured if configured.is_absolute() else context.workspace / configured
    _ = migrate(path, context.workspace / 'backups/mobile-input-rejections' / uuid4().hex)


steps = [step(migrate_mobile_inputs)]
