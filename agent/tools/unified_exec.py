"""Compatibility module preserving the old process runtime module identity."""

import sys

from agent import process_runtime as _runtime

sys.modules[__name__] = _runtime
