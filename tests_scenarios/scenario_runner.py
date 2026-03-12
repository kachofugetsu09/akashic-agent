from __future__ import annotations

import copy
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent.config import load_config
from agent.config_models import Config
from agent.looping.core import AgentLoop
from agent.provider import LLMProvider, LLMResponse
from agent.tools.registry import ToolRegistry
from bootstrap.providers import build_providers
from bootstrap.tools import build_registered_tools
from bus.events import InboundMessage
from bus.processing import ProcessingState
from bus.queue import MessageBus
from core.memory.runtime import MemoryRuntime
from core.net.http import SharedHttpResources
from proactive.presence import PresenceStore
from session.manager import SessionManager
from tests_scenarios.fixtures import (
    ScenarioAssertions,
    ScenarioMemoryItem,
    ScenarioSpec,
)
from tests_scenarios.judge_runner import ScenarioJudgeRunner, ScenarioJudgeVerdict


class RecordingProvider:
    def __init__(self, provider: Any) -> None:
        self._provider = provider
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> LLMResponse:
        started = time.perf_counter()
        response = await self._provider.chat(**kwargs)
        self.calls.append(
            {
                "model": kwargs.get("model", ""),
                "max_tokens": kwargs.get("max_tokens", 0),
                "tool_names": [
                    tool.get("function", {}).get("name", "")
                    for tool in (kwargs.get("tools") or [])
                ],
                "messages_count": len(kwargs.get("messages") or []),
                "last_user_message": _last_user_message(kwargs.get("messages") or []),
                "response_content": response.content,
                "response_tool_calls": [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                    for tc in response.tool_calls
                ],
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )
        return response


class RecordingToolRegistry(ToolRegistry):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    async def execute(self, name: str, arguments: dict) -> str:
        started = time.perf_counter()
        result = await super().execute(name, arguments)
        self.calls.append(
            {
                "name": name,
                "arguments": arguments,
                "result": result,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )
        return result


@dataclass
class ScenarioRuntime:
    config: Config
    workspace: Path
    artifact_dir: Path
    loop: AgentLoop
    tools: RecordingToolRegistry
    session_manager: SessionManager
    memory_runtime: MemoryRuntime
    http_resources: SharedHttpResources
    llm_provider: RecordingProvider
    light_provider: LLMProvider | None


@dataclass
class ScenarioResult:
    spec_id: str
    artifact_dir: Path
    passed: bool
    final_content: str
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    memory_trace: dict[str, Any] = field(default_factory=dict)
    session_before: dict[str, Any] = field(default_factory=dict)
    session_after: dict[str, Any] = field(default_factory=dict)
    assertion_errors: list[str] = field(default_factory=list)
    judge_verdict: ScenarioJudgeVerdict | None = None
    error: str = ""

    def failure_message(self) -> str:
        parts = []
        if self.error:
            parts.append(self.error)
        parts.extend(self.assertion_errors)
        if self.judge_verdict and not self.judge_verdict.passed:
            parts.append(f"judge 未通过: {self.judge_verdict.reasons}")
        return "\n".join(parts)


class ScenarioRunner:
    def __init__(
        self,
        *,
        config_path: str | Path = "config.json",
        artifact_root: str | Path = ".pytest_artifacts/agent-loop-mvp",
    ) -> None:
        self._config_path = Path(config_path)
        self._artifact_root = Path(artifact_root)
        self._repo_root = Path(__file__).resolve().parent.parent

    async def run(self, spec: ScenarioSpec) -> ScenarioResult:
        session_key = self._resolve_session_key(spec)
        runtime = await self._create_runtime(spec)
        final_content = ""
        session_before: dict[str, Any] = {}
        session_after: dict[str, Any] = {}
        memory_trace: dict[str, Any] = {}
        assertion_errors: list[str] = []
        judge_verdict: ScenarioJudgeVerdict | None = None
        try:
            # 1. 先把 history、memory、memory2 固定种子写进隔离 runtime。
            await self._seed_runtime(runtime, spec, session_key=session_key)
            session_before = self._snapshot_session(runtime.session_manager, session_key)
            # 2. 再发送一条固定消息，真实跑一轮 AgentLoop 主路径。
            outbound = await runtime.loop._process(self._build_message(spec))
            final_content = outbound.content
            session_after = self._snapshot_session(runtime.session_manager, session_key)
            memory_trace = self._load_memory_trace(runtime.workspace)
            # 3. 最后分别执行硬断言和可选 judge，并把结果落盘。
            assertion_errors = self._check_assertions(
                spec.assertions,
                final_content=final_content,
                tool_calls=runtime.tools.calls,
                memory_trace=memory_trace,
            )
            if spec.judge:
                judge_verdict = await self._run_judge(
                    runtime,
                    spec,
                    final_content=final_content,
                    memory_trace=memory_trace,
                )
            passed = not assertion_errors
            if judge_verdict is not None:
                passed = passed and judge_verdict.passed
            result = ScenarioResult(
                spec_id=spec.id,
                artifact_dir=runtime.artifact_dir,
                passed=passed,
                final_content=final_content,
                llm_calls=runtime.llm_provider.calls,
                tool_calls=runtime.tools.calls,
                memory_trace=memory_trace,
                session_before=session_before,
                session_after=session_after,
                assertion_errors=assertion_errors,
                judge_verdict=judge_verdict,
            )
        except Exception as exc:
            if not session_before:
                session_before = self._safe_snapshot_session(
                    runtime.session_manager,
                    session_key,
                )
            session_after = self._safe_snapshot_session(
                runtime.session_manager,
                session_key,
            )
            memory_trace = self._safe_load_memory_trace(runtime.workspace)
            result = ScenarioResult(
                spec_id=spec.id,
                artifact_dir=runtime.artifact_dir,
                passed=False,
                final_content=final_content,
                llm_calls=runtime.llm_provider.calls,
                tool_calls=runtime.tools.calls,
                memory_trace=memory_trace,
                session_before=session_before,
                session_after=session_after,
                error=str(exc),
            )
        await self._write_artifacts(runtime, spec, result)
        await self._close_runtime(runtime)
        return result

    async def _create_runtime(self, spec: ScenarioSpec) -> ScenarioRuntime:
        artifact_dir = self._prepare_artifact_dir(spec.id)
        workspace = artifact_dir / "workspace"
        self._prepare_workspace(workspace)
        config = self._build_config(workspace)
        http_resources = SharedHttpResources()
        runtime_provider, light_provider = build_providers(config)
        tools = RecordingToolRegistry()
        bus = MessageBus()
        (
            tools,
            _push_tool,
            scheduler,
            _mcp_registry,
            memory_runtime,
        ) = build_registered_tools(
            config,
            workspace,
            http_resources,
            bus=bus,
            provider=runtime_provider,
            light_provider=light_provider,
            tools=tools,
        )
        loop_provider = RecordingProvider(runtime_provider)
        session_manager = SessionManager(workspace)
        loop = AgentLoop(
            bus=bus,
            provider=loop_provider,
            tools=tools,
            session_manager=session_manager,
            workspace=workspace,
            model=config.model,
            max_iterations=config.max_iterations,
            max_tokens=config.max_tokens,
            presence=PresenceStore(workspace / "presence.json"),
            light_model=config.light_model or config.model,
            light_provider=light_provider or runtime_provider,
            processing_state=ProcessingState(),
            memory_top_k_procedure=config.memory_v2.top_k_procedure,
            memory_top_k_history=config.memory_v2.top_k_history,
            memory_route_intention_enabled=config.memory_v2.route_intention_enabled,
            memory_sop_guard_enabled=config.memory_v2.sop_guard_enabled,
            memory_gate_llm_timeout_ms=config.memory_v2.gate_llm_timeout_ms,
            memory_gate_max_tokens=config.memory_v2.gate_max_tokens,
            memory_runtime=memory_runtime,
            tool_search_enabled=config.tool_search_enabled,
            memory_hyde_enabled=config.memory_v2.hyde_enabled,
            memory_hyde_timeout_ms=config.memory_v2.hyde_timeout_ms,
        )
        scheduler.agent_loop = loop
        return ScenarioRuntime(
            config=config,
            workspace=workspace,
            artifact_dir=artifact_dir,
            loop=loop,
            tools=tools,
            session_manager=session_manager,
            memory_runtime=memory_runtime,
            http_resources=http_resources,
            llm_provider=loop_provider,
            light_provider=light_provider,
        )

    def _prepare_artifact_dir(self, scenario_id: str) -> Path:
        latest_dir = self._artifact_root / scenario_id / "latest"
        if latest_dir.exists():
            shutil.rmtree(latest_dir)
        latest_dir.mkdir(parents=True, exist_ok=True)
        return latest_dir

    def _prepare_workspace(self, workspace: Path) -> None:
        workspace.mkdir(parents=True, exist_ok=True)
        skills_src = self._repo_root / "skills"
        if skills_src.exists():
            (workspace / "skills").symlink_to(skills_src)

    def _build_config(self, workspace: Path) -> Config:
        config = copy.deepcopy(load_config(self._config_path))
        config.channels.socket = str(workspace / "akasic.sock")
        config.memory_v2.db_path = str(workspace / "memory" / "memory2.db")
        return config

    async def _seed_runtime(
        self,
        runtime: ScenarioRuntime,
        spec: ScenarioSpec,
        *,
        session_key: str,
    ) -> None:
        # 1. 先写入 v1 memory 文件，保证 system prompt 能读到测试专用内容。
        runtime.memory_runtime.port.write_long_term(spec.memory.long_term)
        runtime.memory_runtime.port.write_self(spec.memory.self_profile)
        runtime.memory_runtime.port.write_now(spec.memory.now)
        # 2. 再注入固定 session history，保证本轮上下文完全可控。
        session = runtime.session_manager.get_or_create(session_key)
        session.messages = copy.deepcopy(spec.history)
        session.metadata = {}
        session.last_consolidated = 0
        runtime.session_manager.save(session)
        # 3. 最后写入 memory2 测试数据，走真实 embedding 与真实 upsert。
        for index, item in enumerate(spec.memory2_items):
            await self._save_memory_item(runtime, item, index)

    async def _save_memory_item(
        self,
        runtime: ScenarioRuntime,
        item: ScenarioMemoryItem,
        index: int,
    ) -> None:
        source_ref = item.source_ref or f"scenario:{index}"
        await runtime.memory_runtime.port.save_item(
            summary=item.summary,
            memory_type=item.memory_type,
            extra=item.extra,
            source_ref=source_ref,
            happened_at=item.happened_at or None,
        )

    def _build_message(self, spec: ScenarioSpec) -> InboundMessage:
        return InboundMessage(
            channel=spec.channel,
            sender="scenario",
            chat_id=spec.chat_id,
            content=spec.message,
            timestamp=spec.request_time,
        )

    def _resolve_session_key(self, spec: ScenarioSpec) -> str:
        # 1. 统一使用真实主链路的 channel:chat_id 规则推导 session_key。
        spec.validate_session_key()
        # 2. 保留显式字段仅用于兼容和输入校验，不再作为执行时主来源。
        return spec.derived_session_key

    def _snapshot_session(self, manager: SessionManager, session_key: str) -> dict[str, Any]:
        session = manager.get_or_create(session_key)
        return {
            "key": session.key,
            "metadata": copy.deepcopy(session.metadata),
            "messages": copy.deepcopy(session.messages),
            "history": copy.deepcopy(session.get_history()),
        }

    def _safe_snapshot_session(
        self,
        manager: SessionManager,
        session_key: str,
    ) -> dict[str, Any]:
        try:
            return self._snapshot_session(manager, session_key)
        except Exception:
            return {}

    def _load_memory_trace(self, workspace: Path) -> dict[str, Any]:
        trace_path = workspace / "memory" / "memory2_retrieve_trace.jsonl"
        if not trace_path.exists():
            return {}
        lines = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return lines[-1] if lines else {}

    def _safe_load_memory_trace(self, workspace: Path) -> dict[str, Any]:
        try:
            return self._load_memory_trace(workspace)
        except Exception:
            return {}

    def _check_assertions(
        self,
        assertions: ScenarioAssertions,
        *,
        final_content: str,
        tool_calls: list[dict[str, Any]],
        memory_trace: dict[str, Any],
    ) -> list[str]:
        errors: list[str] = []
        normalized_final = _normalize_assert_text(final_content)
        if assertions.route_decision:
            actual = str(memory_trace.get("route_decision", ""))
            if actual != assertions.route_decision:
                errors.append(f"route_decision 不匹配: expected={assertions.route_decision} actual={actual}")
        if assertions.min_history_hits is not None:
            actual = int(memory_trace.get("history_hits", 0) or 0)
            if actual < assertions.min_history_hits:
                errors.append(f"history_hits 过小: expected>={assertions.min_history_hits} actual={actual}")
        if assertions.max_history_hits is not None:
            actual = int(memory_trace.get("history_hits", 0) or 0)
            if actual > assertions.max_history_hits:
                errors.append(f"history_hits 过大: expected<={assertions.max_history_hits} actual={actual}")
        called_tools = [item["name"] for item in tool_calls]
        for tool_name in assertions.required_tools:
            if tool_name not in called_tools:
                errors.append(f"缺少预期工具调用: {tool_name}")
        for needle in assertions.final_contains:
            if _normalize_assert_text(needle) not in normalized_final:
                errors.append(f"最终回复缺少关键字: {needle}")
        for needle in assertions.final_not_contains:
            if _normalize_assert_text(needle) in normalized_final:
                errors.append(f"最终回复包含了不应出现的内容: {needle}")
        return errors

    async def _run_judge(
        self,
        runtime: ScenarioRuntime,
        spec: ScenarioSpec,
        *,
        final_content: str,
        memory_trace: dict[str, Any],
    ) -> ScenarioJudgeVerdict:
        provider = runtime.light_provider
        if provider is None:
            provider = build_providers(runtime.config)[0]
        runner = ScenarioJudgeRunner(
            provider=provider,
            model=runtime.config.light_model or runtime.config.model,
        )
        return await runner.run(
            spec,
            spec.judge,
            final_content=final_content,
            memory_trace=memory_trace,
            tool_calls=runtime.tools.calls,
        )

    async def _write_artifacts(
        self,
        runtime: ScenarioRuntime,
        spec: ScenarioSpec,
        result: ScenarioResult,
    ) -> None:
        self._write_json(runtime.artifact_dir / "config_snapshot.json", _config_snapshot(runtime.config, runtime.workspace))
        self._write_json(runtime.artifact_dir / "input.json", _scenario_snapshot(spec))
        self._write_json(runtime.artifact_dir / "summary.json", _result_snapshot(result))
        self._write_jsonl(runtime.artifact_dir / "llm_calls.jsonl", result.llm_calls)
        self._write_jsonl(runtime.artifact_dir / "tool_calls.jsonl", result.tool_calls)
        self._write_json(runtime.artifact_dir / "memory_trace.json", result.memory_trace)
        self._write_json(runtime.artifact_dir / "session_before.json", result.session_before)
        self._write_json(runtime.artifact_dir / "session_after.json", result.session_after)
        if result.judge_verdict is not None:
            self._write_json(runtime.artifact_dir / "judge.json", asdict(result.judge_verdict))

    async def _close_runtime(self, runtime: ScenarioRuntime) -> None:
        await runtime.memory_runtime.aclose()
        await runtime.http_resources.aclose()

    def _write_json(self, path: Path, payload: Any) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def _config_snapshot(config: Config, workspace: Path) -> dict[str, Any]:
    return {
        "model": config.model,
        "light_model": config.light_model,
        "tool_search_enabled": config.tool_search_enabled,
        "memory_v2_enabled": config.memory_v2.enabled,
        "memory_v2_db_path": config.memory_v2.db_path,
        "workspace": str(workspace),
    }


def _scenario_snapshot(spec: ScenarioSpec) -> dict[str, Any]:
    return {
        "id": spec.id,
        "message": spec.message,
        "channel": spec.channel,
        "chat_id": spec.chat_id,
        "session_key": spec.session_key or spec.derived_session_key,
        "derived_session_key": spec.derived_session_key,
        "request_time": spec.request_time.isoformat(),
        "history": spec.history,
        "memory": asdict(spec.memory),
        "memory2_items": [asdict(item) for item in spec.memory2_items],
        "assertions": asdict(spec.assertions),
        "judge": asdict(spec.judge) if spec.judge else None,
    }


def _result_snapshot(result: ScenarioResult) -> dict[str, Any]:
    return {
        "spec_id": result.spec_id,
        "passed": result.passed,
        "final_content": result.final_content,
        "error": result.error,
        "assertion_errors": result.assertion_errors,
        "judge_verdict": asdict(result.judge_verdict) if result.judge_verdict else None,
    }


def _last_user_message(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            return json.dumps(content, ensure_ascii=False)
        return str(content)
    return ""


def _normalize_assert_text(text: str) -> str:
    return (
        str(text)
        .replace(" ", "")
        .replace("\n", "")
        .replace("\t", "")
        .replace("\r", "")
        .replace("《", "")
        .replace("》", "")
    )
