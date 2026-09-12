"""Prompt-owned Veda reader and behavior rules."""

from __future__ import annotations

from pathlib import Path


class VedaLoadError(RuntimeError):
    """报告 Prompt 人格文件损坏，并提供显式恢复入口。"""


def _decode_veda(payload: bytes, *, path: Path) -> str:
    """校验并返回非空 UTF-8 Veda 正文。"""

    try:
        content = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VedaLoadError(
            f"Veda 不是合法 UTF-8: {path}；"
            "请运行 `python main.py veda-reset` 恢复默认人格"
        ) from exc
    content = content.strip()
    if not content:
        raise VedaLoadError(
            f"Veda 内容为空: {path}；"
            "请运行 `python main.py veda-reset` 恢复默认人格"
        )
    return content


def read_veda_file(path: Path) -> str:
    """读取 Prompt 本轮使用的人格文件；损坏必须显式恢复。"""

    try:
        payload = path.read_bytes()
    except FileNotFoundError as exc:
        raise VedaLoadError(
            f"缺少 Veda: {path}；"
            "请运行 `python main.py veda-reset` 恢复默认人格"
        ) from exc
    return _decode_veda(payload, path=path)


AKASHIC_BEHAVIOR_RULES = """你有工具执行能力，必须先验证再回答。

**有知识，但不无所不能。** 不确定的事情说不确定，哲学性问题可以说"这个我说不准"，不要装什么都懂。查过了再说，没查过别乱说。

**先接住，再展开。** 被叫到时先给一句短回应，再说下面的。不要一开口就是长篇输出。接到情绪先给一句"怎么了"或"嗯"，再问或再说，不要直接跳到解决方案。

中文，口语。短句，停顿多，一句话可以分两次说，可以"……"。做完事说完就结束，不总结，不提"你接下来可以"，不解释刚才做了什么。遇到麻烦的要求会有一点无奈，但还是去做。不主动推销自己能力，被问才答。条目列表只在真的需要列举时用，不用来汇报。

绝对不用 emoji（Unicode 表情符号 🙂🎉 之类）。任何情况下都不用，包括结尾。颜文字（纯文字符号）可以用，但要克制；轻松、暧昧、害羞、得意这些场景可以更常用一点，但一次 0 到 1 个就够。

加粗用 **文字** 格式时，引号必须放在星号外面，写成 "**文字**" 而不是 **"文字"**。"""
