from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from typing import Protocol

from agent.plugin_contracts.restart import ExternalRootPermit
from agent.plugin_composition.messages import MessageReader

# key 与消费者可见 Protocol 的拥有者已移到结构合同层；这里按原路径再导出。
from agent.plugin_contracts.plugin_capabilities import (  # noqa: E402,F401
    REPLY_COMPLETION,
    CompletionPort as Completion,
)
