from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ResponseMetadata:
    raw_text: str


@dataclass
class ParsedResponse:
    clean_text: str
    metadata: ResponseMetadata


def parse_response(
    raw_text: str,
    *,
    tool_chain: list[dict[str, object]],
) -> ParsedResponse:
    return ParsedResponse(
        clean_text=raw_text,
        metadata=ResponseMetadata(raw_text=raw_text),
    )
