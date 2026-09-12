"""Compatibility module preserving the old archive module identity."""

import sys

from agent.plugin_composition import archive as _archive

sys.modules[__name__] = _archive
