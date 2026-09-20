"""Compatibility module preserving the old shell runtime module identity."""

import sys

from agent.plugin_composition import shell_runtime as _runtime

sys.modules[__name__] = _runtime
