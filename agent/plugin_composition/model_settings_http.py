"""Shared unavailable error for Core model adapters.

The Models plugin owns payloads, HTTP routes, RPC method schemas, and provider
error mapping. Core adapters only need this typed boundary error.
"""

from __future__ import annotations


class ModelControlUnavailable(RuntimeError):
    """The bound plugin snapshot does not provide model control services."""


__all__ = ["ModelControlUnavailable"]
