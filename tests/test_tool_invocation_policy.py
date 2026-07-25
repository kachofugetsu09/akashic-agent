from __future__ import annotations

from pathlib import Path

import pytest

from agent.policies.tool_invocation_policy import (
    ToolInvocationContext,
    ToolInvocationDecision,
    ToolInvocationPolicyEngine,
)


def test_context_normalizes_capabilities_arguments_and_metadata() -> None:
    context = ToolInvocationContext(
        tool_name="read_file",
        arguments={"path": "README.md"},
        registry_risk="read-only",
        capabilities=frozenset({"filesystem.read"}),
        source="passive",
        metadata={"origin": "unit"},
    )

    assert context.tool_name == "read_file"
    assert dict(context.arguments) == {"path": "README.md"}
    assert context.capabilities == frozenset({"filesystem.read"})
    assert dict(context.metadata) == {"origin": "unit"}


def test_context_rejects_invalid_source() -> None:
    with pytest.raises(ValueError, match="unsupported tool invocation source"):
        ToolInvocationContext(tool_name="read_file", source="invalid")  # type: ignore[arg-type]


def test_context_rejects_invalid_task_execution_phase() -> None:
    with pytest.raises(ValueError, match="unsupported task execution phase"):
        ToolInvocationContext(
            tool_name="read_file",
            task_execution_phase="wrk",  # type: ignore[arg-type]
        )


def test_decision_allowed_helper_and_trace_metadata() -> None:
    decision = ToolInvocationDecision(
        action="allow",
        reason="read_only_allowed",
        risk="read-only",
        metadata={"tool_name": "read_file"},
    )

    assert decision.allowed is True
    assert decision.to_trace_metadata() == {
        "action": "allow",
        "reason": "read_only_allowed",
        "risk": "read-only",
        "policy_name": "ToolInvocationPolicyEngine",
        "metadata": {"tool_name": "read_file"},
    }


def test_decision_rejects_invalid_action() -> None:
    with pytest.raises(ValueError, match="unsupported tool invocation action"):
        ToolInvocationDecision(
            action="sandbox",  # type: ignore[arg-type]
            reason="not_in_p1",
            risk="unknown",
        )


def test_engine_exists() -> None:
    engine = ToolInvocationPolicyEngine()

    assert engine.policy_name == "ToolInvocationPolicyEngine"


def test_unregistered_tool_is_denied() -> None:
    decision = ToolInvocationPolicyEngine().evaluate(
        ToolInvocationContext(
            tool_name="missing_tool",
            registered=False,
            registry_risk="unknown",
        )
    )

    assert decision.action == "deny"
    assert decision.allowed is False
    assert decision.reason == "tool_invocation_unregistered_tool"
    assert decision.metadata["tool_name"] == "missing_tool"


def test_destructive_tool_is_denied() -> None:
    decision = ToolInvocationPolicyEngine().evaluate(
        ToolInvocationContext(
            tool_name="delete_everything",
            registered=True,
            registry_risk="destructive",
        )
    )

    assert decision.action == "deny"
    assert decision.reason == "tool_invocation_destructive_denied"
    assert decision.risk == "destructive"


def test_read_only_tool_is_allowed() -> None:
    decision = ToolInvocationPolicyEngine().evaluate(
        ToolInvocationContext(
            tool_name="read_file",
            registered=True,
            registry_risk="read-only",
        )
    )

    assert decision.action == "allow"
    assert decision.allowed is True
    assert decision.reason == "risk_strategy_read_only_allowed"
    assert decision.metadata["risk_strategy"]["reason"] == (
        "risk_strategy_read_only_allowed"
    )


def test_non_task_execution_write_external_and_unknown_require_approval() -> None:
    engine = ToolInvocationPolicyEngine()

    write = engine.evaluate(
        ToolInvocationContext(tool_name="write_file", registry_risk="write")
    )
    external = engine.evaluate(
        ToolInvocationContext(
            tool_name="message_push",
            registry_risk="external-side-effect",
        )
    )
    unknown = engine.evaluate(
        ToolInvocationContext(tool_name="custom_tool", registry_risk="unknown")
    )

    assert write.action == "defer"
    assert external.action == "defer"
    assert unknown.action == "defer"
    assert write.reason == "risk_strategy_write_requires_approval"
    assert external.reason == "risk_strategy_external_side_effect_requires_approval"
    assert unknown.reason == "risk_strategy_unknown_requires_approval"
    assert write.metadata["approval_scope"] == "tool_call"


def test_passive_read_only_shell_capability_requires_approval() -> None:
    decision = ToolInvocationPolicyEngine().evaluate(
        ToolInvocationContext(
            tool_name="shell",
            registered=True,
            registry_risk="read-only",
            capabilities=frozenset({"shell.execute"}),
        )
    )

    assert decision.action == "defer"
    assert decision.reason == "risk_strategy_shell_requires_approval"


def test_task_execution_work_allows_read_only() -> None:
    decision = ToolInvocationPolicyEngine().evaluate(
        ToolInvocationContext(
            tool_name="read_file",
            registry_risk="read-only",
            source="task_execution",
            task_execution_active=True,
            task_execution_phase="work",
        )
    )

    assert decision.action == "allow"
    assert decision.reason == "tool_invocation_task_execution_read_only_allowed"


def test_task_execution_work_allows_control_capability_even_when_write_risk() -> None:
    decision = ToolInvocationPolicyEngine().evaluate(
        ToolInvocationContext(
            tool_name="finish_task_step_execution",
            registered=True,
            registry_risk="write",
            capabilities=frozenset({"task_execution.finish"}),
            source="task_execution",
            task_execution_active=True,
            task_execution_phase="work",
        )
    )

    assert decision.action == "allow"
    assert decision.reason == "tool_invocation_task_execution_control_allowed"


def test_task_execution_work_defers_write_shell_external_and_unknown() -> None:
    engine = ToolInvocationPolicyEngine()

    write = engine.evaluate(
        ToolInvocationContext(
            tool_name="write_file",
            registry_risk="write",
            source="task_execution",
            task_execution_active=True,
            task_execution_phase="work",
        )
    )
    shell = engine.evaluate(
        ToolInvocationContext(
            tool_name="shell",
            registry_risk="read-only",
            source="task_execution",
            task_execution_active=True,
            task_execution_phase="work",
        )
    )
    external = engine.evaluate(
        ToolInvocationContext(
            tool_name="message_push",
            registry_risk="external-side-effect",
            source="task_execution",
            task_execution_active=True,
            task_execution_phase="work",
        )
    )
    unknown = engine.evaluate(
        ToolInvocationContext(
            tool_name="custom_tool",
            registry_risk="unknown",
            source="task_execution",
            task_execution_active=True,
            task_execution_phase="work",
        )
    )

    assert write.action == "defer"
    assert shell.action == "defer"
    assert external.action == "defer"
    assert unknown.action == "defer"
    assert write.reason == "tool_invocation_task_execution_authorization_required"
    assert write.metadata["durable_transition"] == "waiting_authorization"


def test_task_execution_unregistered_and_destructive_precedence() -> None:
    engine = ToolInvocationPolicyEngine()

    unregistered = engine.evaluate(
        ToolInvocationContext(
            tool_name="missing",
            registered=False,
            registry_risk="read-only",
            source="task_execution",
            task_execution_active=True,
            task_execution_phase="work",
        )
    )
    destructive = engine.evaluate(
        ToolInvocationContext(
            tool_name="delete_everything",
            registry_risk="destructive",
            source="task_execution",
            task_execution_active=True,
            task_execution_phase="work",
        )
    )

    assert unregistered.action == "deny"
    assert unregistered.reason == "tool_invocation_unregistered_tool"
    assert destructive.action == "deny"
    assert destructive.reason == "tool_invocation_destructive_denied"


def test_policy_types_are_exported_from_policies_package() -> None:
    from agent.policies import (
        DefaultToolRiskStrategy,
        RiskStrategyContext,
        RiskStrategyDecision,
        ToolInvocationContext as ExportedContext,
        ToolInvocationDecision as ExportedDecision,
        ToolInvocationPolicyEngine as ExportedEngine,
    )
    from agent.policies.tool_risk_strategy import (
        DefaultToolRiskStrategy as DirectStrategy,
        RiskStrategyContext as DirectStrategyContext,
        RiskStrategyDecision as DirectStrategyDecision,
    )

    assert ExportedContext is ToolInvocationContext
    assert ExportedDecision is ToolInvocationDecision
    assert ExportedEngine is ToolInvocationPolicyEngine
    assert DefaultToolRiskStrategy is DirectStrategy
    assert RiskStrategyContext is DirectStrategyContext
    assert RiskStrategyDecision is DirectStrategyDecision


def test_invocation_policy_denies_file_path_outside_resource_roots(
    tmp_path: Path,
) -> None:
    decision = ToolInvocationPolicyEngine().evaluate(
        ToolInvocationContext(
            tool_name="read_file",
            arguments={"path": "../secret.txt"},
            registered=True,
            registry_risk="read-only",
            metadata={"resource_roots": (str(tmp_path),)},
        )
    )

    assert decision.action == "deny"
    assert decision.reason == "resource_policy_file_path_outside_roots"
    assert decision.metadata["resource_policy"]["metadata"]["invoker_reached"] is False


def test_invocation_policy_allows_workspace_file_then_existing_read_only_rule(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text("ok", encoding="utf-8")

    decision = ToolInvocationPolicyEngine().evaluate(
        ToolInvocationContext(
            tool_name="read_file",
            arguments={"path": "README.md"},
            registered=True,
            registry_risk="read-only",
            metadata={"resource_roots": (str(tmp_path),)},
        )
    )

    assert decision.action == "allow"
    assert decision.reason == "risk_strategy_read_only_allowed"
    assert decision.metadata["resource_policy"]["reason"] == (
        "resource_policy_file_path_allowed"
    )


def test_invocation_policy_denies_protected_argument_with_resource_metadata(
    tmp_path: Path,
) -> None:
    decision = ToolInvocationPolicyEngine().evaluate(
        ToolInvocationContext(
            tool_name="tool_search",
            arguments={"query": "x", "_session_key": "forged"},
            registered=True,
            registry_risk="read-only",
            metadata={"resource_roots": (str(tmp_path),)},
        )
    )

    assert decision.action == "deny"
    assert decision.reason == "resource_policy_protected_argument_forged"
    assert (
        decision.metadata["resource_policy"]["metadata"]["argument"] == "_session_key"
    )


def test_resource_policy_path_deny_precedes_unknown_risk_default(tmp_path: Path) -> None:
    decision = ToolInvocationPolicyEngine().evaluate(
        ToolInvocationContext(
            tool_name="read_file",
            arguments={"path": "/etc/passwd"},
            registered=True,
            registry_risk="unknown",
            metadata={"resource_roots": (str(tmp_path),)},
        )
    )

    assert decision.action == "deny"
    assert decision.reason == "resource_policy_protected_system_path"
    assert decision.metadata["resource_policy"]["metadata"]["invoker_reached"] is False


def test_resource_policy_protected_argument_deny_precedes_write_risk() -> None:
    decision = ToolInvocationPolicyEngine().evaluate(
        ToolInvocationContext(
            tool_name="write_file",
            arguments={"path": "notes.md", "_session_key": "forged"},
            registered=True,
            registry_risk="write",
        )
    )

    assert decision.action == "deny"
    assert decision.reason == "resource_policy_protected_argument_forged"
    assert (
        decision.metadata["resource_policy"]["metadata"]["argument"] == "_session_key"
    )


def test_invocation_policy_records_shell_resource_allow_then_defers_task_execution(
    tmp_path: Path,
) -> None:
    decision = ToolInvocationPolicyEngine().evaluate(
        ToolInvocationContext(
            tool_name="shell",
            arguments={"command": "pwd"},
            registered=True,
            registry_risk="external-side-effect",
            source="task_execution",
            task_execution_active=True,
            task_execution_phase="work",
            metadata={"resource_roots": (str(tmp_path),)},
        )
    )

    assert decision.action == "defer"
    assert decision.reason == "tool_invocation_task_execution_authorization_required"
    assert (
        decision.metadata["resource_policy"]["reason"]
        == "resource_policy_shell_command_allowed"
    )


def test_invocation_policy_records_url_resource_allow_metadata(tmp_path: Path) -> None:
    decision = ToolInvocationPolicyEngine().evaluate(
        ToolInvocationContext(
            tool_name="web_fetch",
            arguments={"url": "https://example.com/page"},
            registered=True,
            registry_risk="read-only",
            metadata={"resource_roots": (str(tmp_path),)},
        )
    )

    assert decision.action == "allow"
    assert decision.reason == "risk_strategy_read_only_allowed"
    assert decision.metadata["resource_policy"]["reason"] == "resource_policy_url_allowed"
