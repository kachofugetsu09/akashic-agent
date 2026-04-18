from __future__ import annotations

from pathlib import Path

from eval.longmemeval.runtime import close_runtime as close_runtime
from eval.longmemeval.runtime import create_runtime as _create_runtime

async def create_runtime(config_path: Path, workspace: Path, persona_profile: str) -> object:
    _ = persona_profile
    return await _create_runtime(config_path, workspace)
