from pathlib import Path

import pytest

from bootstrap.workspace_lock import WorkspaceInstanceLock


def test_workspace_lock_rejects_second_owner_and_releases(tmp_path: Path) -> None:
    first = WorkspaceInstanceLock(tmp_path)
    second = WorkspaceInstanceLock(tmp_path)
    first.acquire()
    with pytest.raises(RuntimeError, match="其他 runtime"):
        second.acquire()
    first.release()
    second.acquire()
    second.release()
