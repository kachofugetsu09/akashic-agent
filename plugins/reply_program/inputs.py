from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any

from agent.plugin_composition.models import (
    StreamCallback,
)
from agent.plugin_contracts import ContentPart
from plugins.content.contract import (
    CONTENT as CONTENT,
    Content as Content,
    ContentView as ContentView,
)
from plugins.context.contract import (
    CONTEXT as CONTEXT,
    MATERIALS_V4 as MATERIALS,
    ContextBuilder as ContextBuilder,
    ContextMaterialsV4 as ContextMaterials,
    MaterialView as MaterialView,
)
from plugins.models.contract import (
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
from plugins.react.contract import (
    REACT_ORDERED_V2 as REACT,
)
from agent.plugin_contracts.tools import (
    TOOL_CLEANUP as TOOL_CLEANUP,
    TOOLS as TOOLS,
    ToolCatalog as ToolCatalog,
    ToolCleanup as ToolCleanup,
    ToolView as ToolView,
)
from plugins.tools.contract import (
    TOOL_PROGRAM_V2 as TOOL_PROGRAM,
    ToolMenu as ToolMenu,
    ToolPresentation as ToolPresentation,
    OrderedToolProgram as ToolProgram,
)
from plugins.turn_projection.contract import (
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
