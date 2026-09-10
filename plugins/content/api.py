from __future__ import annotations

import re
import hashlib
import json
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from agent.plugin_contracts.turn_effects import PostCommitEffect, post_commit_effect
from agent.plugin_contracts.artifacts import check_artifact_id
from agent.plugin_contracts import ContentPart, ContentReferences, Control, Input, Message














ContentCheck = Callable[[ContentPart], ContentReferences]





# 内容 schema 与校验函数的拥有者已移到结构合同层；这里保留再导出，对象身份不变。
from agent.plugin_contracts.content import (  # noqa: E402,F401
    ContentSchema,
    Reference,
    Span,
    TextProtocol,
    TextSource,
    check_artifact,
    check_turn_input,
    is_user_input,
    legacy_post_commit_effect,
)

TextDecoder = Callable[
    [TextSource, tuple[Reference, ...]],
    Awaitable[tuple[Sequence[Span], Mapping[str, object]]],
]
