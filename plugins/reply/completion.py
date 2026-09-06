from contextlib import AbstractAsyncContextManager, AbstractContextManager
from typing import Protocol

from agent.plugin_composition import ServiceKey
from session.log import MessageReader


class Completion(Protocol):
    def activity(self, reader: MessageReader, source: str) -> AbstractContextManager[None]: ...

    def __call__(self, reader: MessageReader, source: str) -> AbstractAsyncContextManager[None]: ...


# 可选策略覆盖一次回复的完成阶段；Reply 自身不取得发送能力。
REPLY_COMPLETION = ServiceKey[Completion]("reply.completion.v1")
