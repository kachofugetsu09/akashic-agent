from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from infra.mobile_webui.store import MobileWebUiStore


logger = logging.getLogger(__name__)


def auto_publish_webui(
    source_repository: Path,
    workspace: Path,
    *,
    server_id: str,
) -> bool:
    """在受信任 main 首次启动时发布 Stable，并返回是否发生发布。"""

    # 1. 只让与 origin/main 精确一致的本地 main 拥有自动发布权限
    branch = _git(source_repository, "symbolic-ref", "--quiet", "--short", "HEAD", allow_missing=True)
    if branch != "main":
        return False
    if _git(source_repository, "status", "--porcelain", "--untracked-files=no"):
        return False
    head = _git(source_repository, "rev-parse", "HEAD")
    tracked_main = _git(source_repository, "rev-parse", "refs/remotes/origin/main")
    if head != tracked_main:
        raise RuntimeError(
            "Mobile WebUI Stable 自动对账拒绝未同步的 main："
            f"HEAD={head} origin/main={tracked_main}"
        )

    # 2. append-only 发布历史是 applied ledger，避免重启覆盖显式 rollback
    store = MobileWebUiStore(workspace / "mobile-webui", server_id=server_id)
    try:
        if store.has_stable_publication_for_source(head):
            return False
    finally:
        store.close()

    # 3. 复用唯一发布 CLI 的隔离构建、摘要校验和原子提交
    publisher = source_repository / "scripts" / "publish-mobile-webui.py"
    command = [
        sys.executable,
        str(publisher),
        "publish",
        "--source-repository",
        str(source_repository),
        "--workspace",
        str(workspace),
        "--server-id",
        server_id,
        "--source-commit",
        head,
        "--stable",
        "--actor",
        "mobile-gateway-main-reconciler",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=source_repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        reason = (error.stderr or error.stdout or str(error)).strip()
        raise RuntimeError(
            f"Mobile WebUI Stable 自动对账失败 source_commit={head}: {reason}"
        ) from error
    logger.info(
        "Mobile WebUI Stable 自动对账完成 source_commit=%s result=%s",
        head,
        completed.stdout.strip(),
    )
    return True


def _git(repository: Path, *args: str, allow_missing: bool = False) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=not allow_missing,
        capture_output=True,
        text=True,
    )
    if allow_missing and completed.returncode != 0:
        return ""
    return completed.stdout.strip()
