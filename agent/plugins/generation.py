from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from agent.plugins.jobs import RegisteredPluginJob
    from agent.plugins.scope import PluginScope, ScopedEventBus
    from agent.plugins.specs import RegisteredProactiveSource
    from infra.channels.contract import Channel
    from agent.plugins.skill_host import PreparedSkillCatalog
    from agent.mcp.host import PreparedMcpCatalog
    from agent.plugins.activity_host import PreparedJobCatalog, PreparedProactiveCatalog
    from agent.plugins.snapshot import RuntimeSnapshot


GateStatus = Literal["passed", "failed"]


@dataclass(frozen=True)
class PluginSemanticCheck:
    check_id: str
    passed: bool
    evidence: object = ""


@dataclass(frozen=True)
class PluginReadinessContext:
    generation_id: str
    mcp_catalog: PreparedMcpCatalog
    job_catalog: PreparedJobCatalog
    proactive_catalog: PreparedProactiveCatalog


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
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    managed_services: dict[str, dict[str, Any]] = field(default_factory=dict)
    before_turn_modules: tuple[object, ...] = ()
    before_reasoning_modules: tuple[object, ...] = ()
    prompt_render_modules: tuple[object, ...] = ()
    before_step_modules: tuple[object, ...] = ()
    after_step_modules: tuple[object, ...] = ()
    after_reasoning_modules: tuple[object, ...] = ()
    after_turn_modules: tuple[object, ...] = ()
    proactive_modules: tuple[object, ...] = ()
    proactive_lifecycles: tuple[object, ...] = ()
    proactive_module_factories: tuple[object, ...] = ()
    proactive_runtime_factories: tuple[object, ...] = ()
    proactive_sources: tuple[RegisteredProactiveSource, ...] = ()
    jobs: tuple[RegisteredPluginJob, ...] = ()
    channels: tuple[Channel, ...] = ()
    dashboard_module: Path | None = None


@dataclass
class PluginGeneration:
    plugin_id: str
    generation_id: str
    module_path: str
    source_revision: str
    config_revision: str
    instance: object
    scope: PluginScope
    contributions: PluginContributions
    gate_result: GateResult
    skill_catalog: PreparedSkillCatalog | None = None
    mcp_catalog: PreparedMcpCatalog | None = None
    job_catalog: PreparedJobCatalog | None = None
    proactive_catalog: PreparedProactiveCatalog | None = None
    runtime_snapshot: RuntimeSnapshot | None = None
    staged_event_bus: ScopedEventBus | None = None
    initialization_started: bool = False
    minimum_resource_count: int = 0
    state: str = "active"
    lease_count: int = 0
