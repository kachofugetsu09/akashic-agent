from __future__ import annotations

from dataclasses import dataclass


# 插件兼容接口：AfterReasoningCtx 会向插件暴露 ResponseMetadata。
# 新核心不要增加依赖；迁移现有插件及其测试前保留这里的结构。
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
