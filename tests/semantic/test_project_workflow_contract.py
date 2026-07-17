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
