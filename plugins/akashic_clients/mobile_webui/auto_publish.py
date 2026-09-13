from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from .store import MobileWebUiStore


logger = logging.getLogger(__name__)


def auto_publish_webui(
    source_repository: Path,
    workspace: Path,
    *,
    server_id: str,
) -> bool:
    """对账 clean main 的 Stable，并返回是否发生发布。"""

    # 1. 非 clean main 不取得自动发布权限
    branch = _git(source_repository, "symbolic-ref", "--quiet", "--short", "HEAD", allow_missing=True)
    if branch != "main":
        return False
    if _git(source_repository, "status", "--porcelain", "--untracked-files=no"):
        return False
    head = _git(source_repository, "rev-parse", "HEAD")
    tracked_main = _git(source_repository, "rev-parse", "refs/remotes/origin/main")
    if head != tracked_main:
        if _current_stable_matches_head(workspace, server_id=server_id, head=head):
            return False
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
    publisher = Path(__file__).resolve().with_name("release_cli.py")
    if not publisher.is_file():
        raise RuntimeError(f"当前客户端 artifact 缺少发布 CLI: {publisher}")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("AKASHIC_EXTRA_PLUGIN_DIRS", None)
    environment["AKASHIC_CORE_ROOT"] = str(_running_core_root())
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
            env=environment,
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


def _running_core_root() -> Path:
    """Resolve the one Core root already hosting this plugin runtime."""

    import agent

    locations = tuple(
        Path(path).resolve(strict=True)
        for path in getattr(agent, "__path__", ())
    )
    candidates = tuple(
        location.parent
        for location in locations
        if (location / "plugin_composition").is_dir()
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "客户端自动发布需要唯一的运行中 Core root，"
            f"实际发现 {len(candidates)} 个: {candidates}"
        )
    return candidates[0]


def _current_stable_matches_head(workspace: Path, *, server_id: str, head: str) -> bool:
    """判断现有 Stable 是否由当前 Core HEAD 发布。"""

    publication_root = workspace / "mobile-webui"
    if not (publication_root / "publication.sqlite3").is_file():
        return False
    store = MobileWebUiStore(publication_root, server_id=server_id)
    try:
        release = store.get_release()
        if release.stable is None:
            return False
        manifest = store.get_manifest(release.stable.manifest_digest)
        return manifest.source_commit == head
    finally:
        store.close()


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
