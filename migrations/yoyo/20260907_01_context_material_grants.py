"""只为未配置的内置 Context 安装显式材料授权，既有配置保持原样。"""
import os
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context

__depends__ = {"20260906_06_model_call_timing"}
__transactional__ = False

_DEFAULT = (
    'prompt_sources = {default_prompt = "prompt", markdown_memory = "markdown_memory"}\n'
    'summary_source = ["compaction", "compaction"]\n'
)


def install_context_grants(_ledger):
    """临时文件刷盘后无覆盖发布；重试或操作者已有选择时保留原配置。"""
    path = current_migration_context().workspace / "plugin-data/context-builtin/config.local.toml"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".context-grants-" + uuid4().hex + ".tmp")
    try:
        # 1. 完整写入后才让正式名称可见；不存在覆盖或凭据复制。
        with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as output:
            output.write(_DEFAULT)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return
        # 2. 目录项耐久后 yoyo 才能落账；已有文件不做逻辑失效或物理减少。
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


steps = [step(install_context_grants)]
