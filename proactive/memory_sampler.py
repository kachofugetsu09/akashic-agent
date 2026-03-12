"""
proactive/memory_sampler.py — 随机记忆片段抽取。

把 MEMORY.md 文本切成语义块（section / 段落 / 要点），
随机抽取 N 条注入 LLM prompt，制造"偶然想起"的自然感。
纯函数，无 I/O，无 LLM。
"""

from __future__ import annotations

import random as _random
import re


def split_memory_chunks(text: str, max_chunk_chars: int = 400) -> list[str]:
    """将记忆文本切成语义块。

    切分策略（优先级递减）：
    1. ## 开头的 section 标题 + 其紧跟的内容作为一块
    2. 空行分隔的段落
    3. 单个 bullet 点（- / * 开头的行）
    4. 超过 max_chunk_chars 的块再按句子细分
    """
    text = text.strip()
    if not text:
        return []

    # 按 ## 标题切分 section
    sections = re.split(r"(?=^##\s)", text, flags=re.MULTILINE)
    chunks: list[str] = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        if len(section) <= max_chunk_chars:
            chunks.append(section)
            continue

        # section 过长：拆成段落或 bullet
        paragraphs = re.split(r"\n{2,}", section)
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) <= max_chunk_chars:
                chunks.append(para)
                continue
            # 段落还是太长：按 bullet 行拆
            lines = para.splitlines()
            buf: list[str] = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                is_bullet = bool(re.match(r"^[-*]\s", line))
                if is_bullet and buf:
                    chunks.append("\n".join(buf))
                    buf = []
                buf.append(line)
            if buf:
                chunks.append("\n".join(buf))

    return [c for c in chunks if c.strip()]


def sample_memory_chunks(
    memory_text: str,
    n: int,
    rng: _random.Random | None = None,
) -> list[str]:
    """随机抽取 n 个记忆块。

    n=0 或文本为空返回 []。
    可用的块不足 n 个时返回全部。
    """
    if n <= 0 or not memory_text.strip():
        return []
    chunks = split_memory_chunks(memory_text)
    if not chunks:
        return []
    r = rng or _random.Random()
    k = min(n, len(chunks))
    return r.sample(chunks, k)
