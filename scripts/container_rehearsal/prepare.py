"""Orchestrate and atomically publish a container rehearsal root."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.container_rehearsal.candidate import (
    copy_plugin_manifest,
    isolate_schedules,
    write_candidate_config,
)
from scripts.container_rehearsal.model import CopyRecord, sha256
from scripts.container_rehearsal.policy import (
    EXCLUDED_DIRECTORY_NAMES,
    EXCLUDED_RUNTIME_FILES,
    validate_roots,
)
from scripts.container_rehearsal.workspace_snapshot import copy_workspace


def prepare_rehearsal(
    *,
    source_workspace: Path,
    source_config: Path,
    plugin_home: Path,
    target: Path,
) -> Path:
    """Create an isolated rehearsal root and return its machine manifest."""

    source_workspace, source_config, plugin_home, target = validate_roots(
        source_workspace=source_workspace,
        source_config=source_config,
        plugin_home=plugin_home,
        target=target,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.preparing-", dir=target.parent))
    stage.chmod(0o700)
    try:
        # 1. Capture state and produce isolated runtime inputs.
        records, exclusions, databases, consistency = copy_workspace(
            source_workspace, stage / "workspace"
        )
        records, disabled_schedule_count = isolate_schedules(stage / "workspace", records)
        write_candidate_config(source_config, stage / "config.toml", target / "workspace")
        config_record = CopyRecord(
            path="config.toml",
            kind="webui_only_config",
            size=(stage / "config.toml").stat().st_size,
            sha256=sha256(stage / "config.toml"),
        )
        plugin_record, disabled_plugins = copy_plugin_manifest(
            plugin_home, stage / "plugin-home"
        )

        # 2. Record evidence and cleanup boundaries without serializing secrets.
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "workspace": str(source_workspace),
                "config": str(source_config),
                "plugin_home": str(plugin_home),
                "read_only": True,
            },
            "target": str(target),
            "candidate": {
                "workspace": "workspace",
                "config": "config.toml",
                "plugin_manifest": "plugin-home/manifest.toml",
                "config_channels": ["web"],
                "model_registry_preserved": True,
                "plugin_manifest_copied_unmodified": False,
                "plugins_disabled_until_rebuilt": disabled_plugins,
                "plugin_data_source": "workspace/plugin-data",
                "plugin_cache_copied": False,
                "schedules_disabled": disabled_schedule_count,
                "source_schedules": "workspace/schedules.source.json",
            },
            "exclusion_policy": {
                "directory_names": sorted(EXCLUDED_DIRECTORY_NAMES),
                "runtime_files": sorted(EXCLUDED_RUNTIME_FILES),
                "additional": [
                    "*_rebuild directories",
                    "mobile-webui-build-* directories",
                    "mobile-webui/staging and mobile-webui/trash",
                    "SQLite -wal/-shm/-journal sidecars",
                    "workspace skills and drift/skills cache symlinks",
                    "non-regular filesystem entries",
                ],
            },
            "excluded": exclusions,
            "databases": databases,
            "consistency": consistency,
            "files": [
                record.__dict__
                for record in sorted(
                    [*records, config_record, plugin_record], key=lambda item: item.path
                )
            ],
            "cleanup": {
                "exact_paths": [str(target)],
                "guard_manifest": str(target / "rehearsal-manifest.json"),
            },
        }
        manifest_path = stage / "rehearsal-manifest.json"
        _ = manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_path.chmod(0o600)

        # 3. Publish only after every validation has completed.
        os.replace(stage, target)
        return target / "rehearsal-manifest.json"
    finally:
        if stage.exists():
            shutil.rmtree(stage)
