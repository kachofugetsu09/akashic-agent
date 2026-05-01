from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent.core.response_parser import ResponseMetadata
from agent.lifecycle.types import AfterReasoningCtx, PromptRenderCtx


def _load_citation_plugin_module() -> Any:
    path = Path(__file__).parents[1] / "plugins" / "01_citation" / "plugin.py"
    spec = importlib.util.spec_from_file_location("test_01_citation_plugin", path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_citation_module = _load_citation_plugin_module()
CitationAfterReasoningModule = _citation_module.CitationAfterReasoningModule
CitationPlugin = _citation_module.CitationPlugin
ProtocolTagCleanupModule = _citation_module.ProtocolTagCleanupModule
extract_cited_ids = _citation_module.extract_cited_ids
extract_cited_ids_from_tool_chain = _citation_module.extract_cited_ids_from_tool_chain
strip_trailing_protocol_tags = _citation_module.strip_trailing_protocol_tags


def test_citation_extracts_ascii_marker_only_at_end() -> None:
    clean, ids = extract_cited_ids("答复正文\n§cited:[mem_1,mem-2]§")

    assert clean == "答复正文"
    assert ids == ["mem_1", "mem-2"]


def test_citation_extracts_marker_with_spaces_after_commas() -> None:
    clean, ids = extract_cited_ids("答复正文\n§cited:[mem_1, mem-2]§")

    assert clean == "答复正文"
    assert ids == ["mem_1", "mem-2"]


def test_citation_strips_empty_marker() -> None:
    clean, ids = extract_cited_ids("答复正文\n§cited:[]§")

    assert clean == "答复正文"
    assert ids == []


def test_citation_keeps_body_text_when_marker_not_at_end() -> None:
    text = "正文里提到 §cited:[mem_1]§ 这串文本，但不是协议行。\n后面还有内容"

    clean, ids = extract_cited_ids(text)

    assert clean == text
    assert ids == []


def test_citation_extracts_before_trailing_protocol_tag() -> None:
    clean, ids = extract_cited_ids("答复正文\n§cited:[mem_1]§ <meme:shy>")

    assert clean == "答复正文 <meme:shy>"
    assert ids == ["mem_1"]


def test_citation_keeps_multiple_trailing_protocol_tags() -> None:
    clean, ids = extract_cited_ids(
        "答复正文\n§cited:[mem_1]§ <meme:shy> <foo:bar>"
    )

    assert clean == "答复正文 <meme:shy> <foo:bar>"
    assert ids == ["mem_1"]


def test_citation_rejects_marker_with_trailing_body_text() -> None:
    text = "答复正文\n§cited:[mem_1]§ 其他文字"

    clean, ids = extract_cited_ids(text)

    assert clean == text
    assert ids == []


def test_citation_rejects_malformed_trailing_protocol_tag() -> None:
    text = "答复正文\n§cited:[mem_1]§ <bad tag>"

    clean, ids = extract_cited_ids(text)

    assert clean == text
    assert ids == []


def test_citation_strips_leftover_trailing_protocol_tags() -> None:
    clean = strip_trailing_protocol_tags("答复正文 <memem:clever> <foo:bar>")

    assert clean == "答复正文"


def test_citation_keeps_body_when_leftover_tag_is_not_trailing() -> None:
    text = "答复正文 <memem:clever> 后面还有内容"

    assert strip_trailing_protocol_tags(text) == text


def test_citation_tool_chain_fallback_uses_recall_memory_cited_item_ids() -> None:
    tool_chain = [
        {
            "text": "thinking",
            "calls": [
                {
                    "name": "recall_memory",
                    "result": json.dumps(
                        {"count": 2, "cited_item_ids": ["mem_1", "mem_2"]}
                    ),
                }
            ],
        }
    ]

    assert extract_cited_ids_from_tool_chain(tool_chain) == ["mem_1", "mem_2"]


def test_citation_tool_chain_fallback_uses_item_ids() -> None:
    tool_chain = [
        {
            "text": "thinking",
            "calls": [
                {
                    "name": "recall_memory",
                    "result": json.dumps(
                        {"count": 2, "items": [{"id": "mem_1"}, {"id": "mem_2"}]}
                    ),
                }
            ],
        }
    ]

    assert extract_cited_ids_from_tool_chain(tool_chain) == ["mem_1", "mem_2"]


@pytest.mark.asyncio
async def test_citation_prompt_module_injects_prompt_section() -> None:
    plugin = CitationPlugin()
    module = plugin.prompt_render_modules_bottom()[0]
    ctx = PromptRenderCtx(
        session_key="telegram:1",
        channel="telegram",
        chat_id="1",
        content="你好",
        media=None,
        timestamp=datetime.now(timezone.utc),
        history=[],
        skill_names=[],
        retrieved_memory_block="",
        disabled_sections=set(),
        turn_injection_prompt="",
    )
    frame = SimpleNamespace(slots={"prompt:ctx": ctx})

    await module.run(frame)

    assert ctx.system_sections_bottom[0].name == "citation_protocol"
    assert "§cited:[id1,id2,id3]§" in ctx.system_sections_bottom[0].content


@pytest.mark.asyncio
async def test_citation_after_reasoning_writes_persist_slot() -> None:
    module = CitationAfterReasoningModule()
    ctx = AfterReasoningCtx(
        session_key="telegram:1",
        channel="telegram",
        chat_id="1",
        tools_used=(),
        thinking=None,
        response_metadata=ResponseMetadata(raw_text="答复正文\n§cited:[mem_1]§"),
        streamed=False,
        tool_chain=(),
        context_retry={},
        reply="答复正文\n§cited:[mem_1]§",
    )
    frame = SimpleNamespace(slots={"reasoning:ctx": ctx})

    await module.run(frame)

    assert ctx.reply == "答复正文"
    assert frame.slots["persist:assistant:cited_memory_ids"] == ["mem_1"]


@pytest.mark.asyncio
async def test_citation_cleanup_module_strips_leftover_protocol_tags() -> None:
    module = ProtocolTagCleanupModule()
    ctx = AfterReasoningCtx(
        session_key="telegram:1",
        channel="telegram",
        chat_id="1",
        tools_used=(),
        thinking=None,
        response_metadata=ResponseMetadata(raw_text="答复正文 <memem:clever>"),
        streamed=False,
        tool_chain=(),
        context_retry={},
        reply="答复正文 <memem:clever>",
    )
    frame = SimpleNamespace(slots={"reasoning:ctx": ctx})

    await module.run(frame)

    assert ctx.reply == "答复正文"
