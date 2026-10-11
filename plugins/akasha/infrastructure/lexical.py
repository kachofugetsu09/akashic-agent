"""在线查询与构建共用的中英分词器。"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jieba

TOKEN_CHUNKS = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u9fff]+")


# jieba 导入与词典初始化只在首次切分中文时发生，不进入插件启动路径。
@cache
def _tokenizer() -> "jieba.Tokenizer":
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pkg_resources is deprecated as an API.*",
            category=UserWarning,
        )
        # jieba 0.42.1 在 Python 3.12 下产生 invalid escape sequence SyntaxWarning，
        # 测试与 Gate 的 -W error 下必须显式忽略，否则导入即失败。
        warnings.filterwarnings("ignore", category=SyntaxWarning)
        import jieba
    return jieba.Tokenizer()


def tokenize(text: str) -> Counter[str]:
    """把中文与 ASCII 文本切成词频。"""

    terms: list[str] = []
    for chunk in TOKEN_CHUNKS.findall(text.lower()):
        pieces = (
            _tokenizer().lcut_for_search(chunk)
            if any("\u3400" <= c <= "\u9fff" for c in chunk)
            else [chunk]
        )
        terms.extend(piece.strip() for piece in pieces if len(piece.strip()) >= 2)
    return Counter(terms)
