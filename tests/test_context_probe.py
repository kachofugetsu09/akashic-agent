from __future__ import annotations

import pytest

from docker.debug.context_probe import _ensure_successful_reply


def test_context_probe_rejects_runtime_failure_reply() -> None:
    with pytest.raises(RuntimeError, match="turn 3 返回运行时失败回复"):
        _ensure_successful_reply("处理消息时出错，请稍后再试。", 3)


def test_context_probe_accepts_normal_reply() -> None:
    _ensure_successful_reply("记住了，你喝茶不加糖。", 1)
