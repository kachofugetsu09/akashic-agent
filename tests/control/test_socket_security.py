from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any, cast

import pytest

from bootstrap.workspace_token import ensure_workspace_token
from infra.control.socket import SocketAppServer


def test_tcp_rejects_non_loopback_and_token_is_private(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="只允许 loopback"):
        _ = SocketAppServer("0.0.0.0:2236", cast(Any, object()))
    _ = ensure_workspace_token(tmp_path)
    if os.name != "nt":
        assert stat.S_IMODE((tmp_path / ".app-server-token").stat().st_mode) == 0o600
