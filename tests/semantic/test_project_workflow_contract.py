from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE_DOCUMENTS = (
    ROOT / "AGENTS.md",
    ROOT / "docs" / "INDEX.md",
    ROOT / "docs" / "WORKFLOW.md",
    ROOT / "docs" / "writing-rules.md",
    ROOT / "docs" / "templates" / "review-contract.md",
)
MARKDOWN_LINK = re.compile(r"\[[^]]+]\(([^)]+)\)")


def test_core_workbook_links_resolve() -> None:
    for document in CORE_DOCUMENTS:
        assert document.is_file(), f"工作手册文件缺失: {document.relative_to(ROOT)}"
        text = document.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.split("#", maxsplit=1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (document.parent / target).resolve()
            assert (
                resolved.exists()
            ), f"工作手册链接失效: {document.relative_to(ROOT)} -> {raw_target}"


def test_core_workbook_files_are_tracked() -> None:
    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            *[str(path.relative_to(ROOT)) for path in CORE_DOCUMENTS],
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert tracked.returncode == 0, f"核心工作手册未进入版本控制: {tracked.stderr}"


def test_new_session_and_change_workflow_have_fixed_entries() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    index = (ROOT / "docs" / "INDEX.md").read_text(encoding="utf-8")

    assert "docs/INDEX.md" in agents
    assert "docs/WORKFLOW.md" in agents
    assert "WORKFLOW.md" in index


def test_pull_request_template_carries_contract_and_gate_status() -> None:
    template = (ROOT / ".github" / "pull_request_template.md").read_text(
        encoding="utf-8"
    )

    for field in (
        "change_type",
        "semantic_delta",
        "capability_owner",
        "runtime_patch",
        "authoritative_state_owner",
        "protected_state",
        "sourceDigest",
        "planDigest",
        "private-contract-gate",
        "NOW.md",
    ):
        assert field in template, f"PR 模板缺少工作流字段: {field}"


def test_mobile_ownership_and_review_are_workflow_gates() -> None:
    projectneed = (ROOT / "docs" / "projectneed.md").read_text(encoding="utf-8")
    workflow = (ROOT / "docs" / "WORKFLOW.md").read_text(encoding="utf-8")

    assert "MOB-001" in projectneed
    for field in (
        "capability_owner",
        "consumer_scope",
        "runtime_patch_reason",
        "client_only_alternative",
    ):
        assert field in projectneed, f"MOB-001 缺少能力归属字段: {field}"
        assert field in workflow, f"工作流缺少能力归属门: {field}"

    assert "base..head" in workflow
    assert "最终 head" in workflow


def test_cross_repository_review_pins_lineage_and_evidence_layers() -> None:
    projectneed = (ROOT / "docs" / "projectneed.md").read_text(encoding="utf-8")
    workflow = (ROOT / "docs" / "WORKFLOW.md").read_text(encoding="utf-8")
    review = (ROOT / "docs" / "templates" / "review-contract.md").read_text(
        encoding="utf-8"
    )

    for invariant in ("MOB-002", "MOB-003", "MOB-004", "TST-007", "TST-008"):
        assert invariant in projectneed, f"缺少跨仓库不变量: {invariant}"

    for requirement in (
        "唯一 writer",
        "schema lineage",
        "runtime commit/tree",
        "scenario profile/hash",
        "requested_ref/resolved_sha/change_source_pr_head",
        "run-specific application ID",
        "pm list packages -u",
        "force-stop",
    ):
        assert requirement in workflow, f"工作流缺少跨仓库评审要求: {requirement}"

    for evidence in (
        "worktree_writers",
        "repository",
        "worktree",
        "branch",
        "owner",
        "base_head",
        "allowed_paths",
        "status",
        "handoff_head",
        "dirty_state",
        "schema lineage",
        "runtime commit/tree",
        "scenario profile/hash",
        "requested_ref",
        "resolved_sha",
        "change_source_pr_head",
        "candidate_application_id",
        "candidate_test_application_id",
        "app_apk_sha256",
        "test_apk_sha256",
        "pm list packages -u",
        "collision_result",
        "protected_packages_before",
        "protected_packages_after",
        "test_phases",
        "phase_boundary",
        "instrumentation_oracle",
        "cleanup_result",
    ):
        assert evidence in review, f"Review 合同缺少证据字段: {evidence}"

    for device_rule in (
        "run-specific application ID",
        "pm list packages -u",
        "collision",
        "base.apk",
        "app data",
        "0 test",
    ):
        assert device_rule in projectneed, f"TST-008 缺少设备保护规则: {device_rule}"


def test_mobile_device_gate_evidence_records_incident_and_current_run() -> None:
    design = (
        ROOT / "docs" / "design" / "mobile-cross-repository-semantic-gate.md"
    ).read_text(encoding="utf-8")

    for evidence in (
        "069a7df1dc9cd1d6cca6f0665d5804dd801d2e85",
        "50a05ba56e385628ebbb55e2d300226a90f212d1",
        "f37a42826d9ad5e0988d8b26eba5dd7a20fb29b8",
        "88365c13369b592290fd69918642b7166fc57c55",
        "com.akashic.mobile.review.rpr6069af37",
        "com.akashic.mobile.review.rpr6069af37.test",
        "0bbf9affde1b9ecfdc382d7674425eb8c74f052bb67a889430756a5088132828",
        "578cdf00c85373160b219487a9650981792f86b59a327030ca9a20ca925e23af",
        "gate_result=passed",
        "OK (0 tests)",
        "pairSendAndReceiveFixedMedia",
        "processRestartResumesWithoutHistoryDuplicates",
        "allowBackup=false",
        "ceDataInode=2589746",
        "正式 v0.8.0/code21",
        "无法恢复",
    ):
        assert evidence in design, f"设备 Gate 设计缺少固定证据: {evidence}"


def test_plugin_gate_evidence_separates_ref_resolution_and_change_source() -> None:
    design = (
        ROOT / "docs" / "design" / "mobile-cross-repository-semantic-gate.md"
    ).read_text(encoding="utf-8")

    for field in ("requested_ref", "resolved_sha", "change_source_pr_head"):
        assert field in design, f"插件 Gate 证据缺少身份字段: {field}"

    for revision in (
        "b434fa74b370fafcd0c64129fe1f641f73f0dbcf",
        "b7f9d4ecee877d22b5452651d9abf699b2d30b7b",
        "cee5bef98e6271c9eb069a6498b4ca072e85c878",
        "5c1d4009bee04af271627819fd5731e1978b5dfe",
        "d5227249f5ad195ab7693ae8c72690ee7db32e28",
        "520ba10032089b1e056a9eecc5f2c1f459c75e5c",
        "334276c4e972f1d80b0a353605d068abc5135b18",
    ):
        assert revision in design, f"插件 Gate 设计缺少固定 revision: {revision}"
