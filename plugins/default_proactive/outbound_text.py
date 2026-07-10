from __future__ import annotations

import re

from ftfy.fixes import decode_escapes, fix_line_breaks


_ESCAPED_LINE_BREAK_RE = re.compile(r"(?<!\\)\\[nr]")


def normalize_outbound_text(text: str) -> str:
    normalized = fix_line_breaks(text)
    if _should_decode_escaped_line_breaks(normalized):
        normalized = fix_line_breaks(decode_escapes(normalized))
    return normalized


def _should_decode_escaped_line_breaks(text: str) -> bool:
    escaped_count = len(_ESCAPED_LINE_BREAK_RE.findall(text))
    if escaped_count == 0:
        return False
    if "\\n\\n" in text or "\\r\\n" in text:
        return True
    return escaped_count >= 2 and escaped_count > text.count("\n")
