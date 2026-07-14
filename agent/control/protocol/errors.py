from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JsonRpcError(Exception):
    code: int
    message: str
    data: dict[str, Any] | None = None

    def envelope(self, request_id: str | int | None) -> dict[str, object]:
        error: dict[str, object] = {"code": self.code, "message": self.message}
        if self.data is not None:
            error["data"] = self.data
        return {"jsonrpc": "2.0", "id": request_id, "error": error}


PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
SERVER_OVERLOADED = -32001
NOT_INITIALIZED = -32002
INCOMPATIBLE_VERSION = -32003
UNAUTHORIZED = -32004
THREAD_NOT_FOUND = -32010
THREAD_BUSY = -32011
TURN_NOT_FOUND = -32012
NOT_SUPPORTED = -32013
