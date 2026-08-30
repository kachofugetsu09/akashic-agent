from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from agent.plugins.scope import PluginScope, ScopedEventBus
    from agent.plugins.skill_host import PreparedSkillCatalog
    from agent.plugins.static_manifest import StaticPluginManifest
    from agent.plugins.snapshot import RuntimeSnapshot


GateStatus = Literal["passed", "failed"]


@dataclass(frozen=True)
class MobileUiAsset:
    module: str
    module_sha256: str
    module_bytes: int
    stylesheet: str
    stylesheet_sha256: str | None
    stylesheet_bytes: int
    navigation_label: str | None
    navigation_description: str | None
    slots: tuple[str, ...]


@dataclass(frozen=True)
class WebModuleAsset:
    module: str
    module_sha256: str
    module_bytes: int
    stylesheet: str
    stylesheet_sha256: str | None
    stylesheet_bytes: int
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    contract_digests: tuple[tuple[str, str], ...] = ()
    contract_sha256: str = ""


@dataclass(frozen=True)
class PluginSemanticCheck:
    check_id: str
    passed: bool
    evidence: object = ""


@dataclass(frozen=True)
class GateCheckResult:
    check_id: str
    status: GateStatus
    evidence: object = ""


@dataclass(frozen=True)
class GateResult:
    gate_id: str
    plugin_id: str
    candidate_revision: str
    status: GateStatus
    checks: tuple[GateCheckResult, ...]
    failure_reason: str = ""


@dataclass(frozen=True)
class PluginContributions:
    manifest: dict[str, object]
    skill_roots: tuple[Path, ...] = ()
    drift_skill_roots: tuple[Path, ...] = ()
    dashboard_module: Path | None = None
    web_module: WebModuleAsset | None = None


@dataclass
class PluginGeneration:
    plugin_id: str
    generation_id: str
    module_path: str
    source_revision: str
    config_revision: str
    plugin_dir: Path
    data_dir: Path
    config: object | None
    instance: object
    scope: PluginScope
    contributions: PluginContributions
    gate_result: GateResult
    config_projection: dict[str, object] = field(default_factory=dict)
    source_type: Literal["builtin", "installed"] = "builtin"
    static_manifest: StaticPluginManifest | None = None
    static_runtime_commands: tuple[tuple[str, tuple[str, ...]], ...] = ()
    composition_runtime_cleanup_registered: bool = False
    replaced_composition_runtime_generation: PluginGeneration | None = None
    formal_root_stopped: bool = False
    formal_root_released: bool = False
    entrypoint: str = "plugin.py"
    skill_catalog: PreparedSkillCatalog | None = None
    runtime_snapshot: RuntimeSnapshot | None = None
    staged_event_bus: ScopedEventBus | None = None
    prepare_started: bool = False
    retire_started: bool = False
    minimum_resource_count: int = 0
    state: str = "active"
    lease_count: int = 0
    reload_tx_id: str | None = None
    production_contributions: PluginContributions | None = None
    production_data_dir: Path | None = None
    boot_created_data_dir: bool = False
    publication_created_data_dir: bool = False
    validation_workspace: Path | None = None
    validation_data_inventory: tuple[str, ...] = ()
