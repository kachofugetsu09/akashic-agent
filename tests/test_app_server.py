from pathlib import Path
from typing import cast

import pytest

from agent.config_models import Config
from bootstrap import app_server
from bootstrap.workspace_lock import WorkspaceInstanceLock


@pytest.mark.asyncio
async def test_stdio_runtime_clears_stale_admissions_only_after_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """确认 stdio runtime 以 workspace owner 身份清理遗留租约。"""
    observed: dict[str, bool] = {}

    def fail_after_build_request(*args, **kwargs):
        observed["clear"] = kwargs["clear_stale_session_admissions"]
        raise RuntimeError("stop after owner routing check")

    monkeypatch.setattr(app_server, "build_core_runtime", fail_after_build_request)

    with pytest.raises(RuntimeError, match="owner routing"):
        await app_server.run_stdio_app_server(cast(Config, object()), tmp_path)

    assert observed == {"clear": True}
    lock = WorkspaceInstanceLock(tmp_path)
    lock.acquire()
    lock.release()
