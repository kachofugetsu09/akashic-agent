from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from agent.config_models import ContextCompactionConfig

if TYPE_CHECKING:
    from agent.context import ContextBuilder
    from agent.core.passive_turn import Reasoner
    from agent.core.runner import CoreRunner
    from agent.core.runtime_support import ToolDiscoveryState
    from agent.provider import LLMProvider
    from agent.retrieval.protocol import MemoryRetrievalPipeline
    from agent.tools.registry import ToolRegistry
    from bus.event_bus import EventBus
    from bus.processing import ProcessingState
    from bus.queue import MessageBus
    from core.memory.engine import MemoryEngine
    from core.memory.runtime import MemoryRuntime
    from proactive_v2.presence import PresenceStore
    from session.compaction_runtime import SessionCompactionPort
    from session.manager import SessionManager


# ── Config dataclasses（参数，不含服务对象）───────────────────────────────────


@dataclass
class LLMConfig:
    model: str = "deepseek-chat"
    light_model: str = ""
    max_iterations: int = 10
    max_tokens: int = 0
    tool_search_enabled: bool = False
    multimodal: bool = True
    vl_available: bool = False


# ── 服务对象分组（仅放对象，不放配置参数）──────────────────────────────────────


@dataclass
class LLMServices:
    """集中持有主模型与轻量模型服务。"""

    provider: LLMProvider
    light_provider: LLMProvider
    fallback_provider: LLMProvider | None = None
    fallback_model: str = ""


@dataclass
class MemoryServices:
    engine: MemoryEngine | None = None


@dataclass
class SessionServices:
    session_manager: SessionManager
    presence: PresenceStore | None = None
    compaction_runtime: "SessionCompactionPort | None" = None


@dataclass
class AgentLoopDeps:
    bus: "MessageBus"
    provider: "LLMProvider"
    tools: "ToolRegistry"
    session_manager: "SessionManager"
    workspace: Path
    event_bus: "EventBus | None" = None
    presence: "PresenceStore | None" = None
    light_provider: "LLMProvider | None" = None
    processing_state: "ProcessingState | None" = None
    memory_runtime: "MemoryRuntime | None" = None
    retrieval_pipeline: "MemoryRetrievalPipeline | None" = None
    context: "ContextBuilder | None" = None
    llm_services: LLMServices | None = None
    memory_services: MemoryServices | None = None
    session_services: SessionServices | None = None
    tool_discovery: "ToolDiscoveryState | None" = None
    reasoner: "Reasoner | None" = None
    core_runner: "CoreRunner | None" = None

@dataclass
class AgentLoopConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    context_compaction: ContextCompactionConfig = field(
        default_factory=ContextCompactionConfig
    )
