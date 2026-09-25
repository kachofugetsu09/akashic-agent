from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any

from agent.plugin_composition.models import (
    StreamCallback,
)
from agent.plugin_contracts import ContentPart
from agent.plugin_contracts.content import (
    CONTENT as CONTENT,
    Content as Content,
    ContentView as ContentView,
)
from agent.plugin_contracts.context import (
    CONTEXT as CONTEXT,
    MATERIALS as MATERIALS,
    ContextBuilder as ContextBuilder,
    ContextMaterials as ContextMaterials,
    MaterialView as MaterialView,
)
from agent.plugin_contracts.models import (
    MODEL_CALLS as MODEL_CALLS,
    MODEL_CHECKS as MODEL_CHECKS,
    MODEL_CONTENT as MODEL_CONTENT,
    MODEL_PROJECTION as MODEL_PROJECTION,
    MODEL_SELECTION as MODEL_SELECTION,
    ContextModel as ContextModel,
    ModelChecks as ModelChecks,
    ModelContent as ModelContent,
    ModelProjections as ModelProjections,
    ModelSelection as ModelSelection,
)
from agent.plugin_contracts.react import (
    REACT as REACT,
)
from agent.plugin_contracts.sources import (
    SOURCE_CHECK as SOURCE_CHECK,
)
from agent.plugin_contracts.tools import (
    TOOL_CLEANUP as TOOL_CLEANUP,
    TOOL_PROGRAM as TOOL_PROGRAM,
    TOOLS as TOOLS,
    ToolCatalog as ToolCatalog,
    ToolCleanup as ToolCleanup,
    ToolMenu as ToolMenu,
    ToolPresentation as ToolPresentation,
    ToolProgram as ToolProgram,
    ToolView as ToolView,
)
from agent.plugin_contracts.turns import (
    TURN_PROJECTION as TURN_PROJECTION,
    TurnProjection as TurnProjection,
)

Materials = Mapping[str, object]
Summary = Mapping[str, object]
Reminder = Mapping[str, object]
Authorize = Callable[[str, Mapping[str, object]], Awaitable[Mapping[str, object] | str]]
Preview = Callable[[str], AbstractContextManager[StreamCallback]]


ContentRenderer = Callable[[ContentPart], Sequence[Mapping[str, Any]]]
CallReader = Callable[[str], Mapping[str, Any]]
