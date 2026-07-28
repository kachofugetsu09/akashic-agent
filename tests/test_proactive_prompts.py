from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from prompts.proactive import build_compose_prompt_messages


def test_build_compose_prompt_messages_forbids_fabricated_links(tmp_path: Path):
    veda = tmp_path / "memory/VEDA.md"
    veda.parent.mkdir(parents=True)
    veda.write_text("test veda", encoding="utf-8")
    ctx = SimpleNamespace(
        now_str="2026-03-18 12:00:00 CST",
        feed_text="（暂无订阅内容）",
        chat_text="用户: 5070ti能跑27b吗",
    )

    system_msg, user_msg = build_compose_prompt_messages(
        workspace=tmp_path,
        prompt_context=ctx,
    )

    assert "test veda" in system_msg
    assert "禁止输出 example.com 这类占位链接" in system_msg
    assert "仅当上面的「今天的新内容」里明确带有真实「原文链接:」字段时" in user_msg
