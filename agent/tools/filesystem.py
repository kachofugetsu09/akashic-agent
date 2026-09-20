"""Compatibility module preserving the old Host Bridge file module identity."""

import sys

from agent.host_bridge import filesystem as _filesystem

sys.modules[__name__] = _filesystem
