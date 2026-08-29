#!/usr/bin/env python3
"""Configure ordinary model plugins through the public settings HTTP API."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def _request(method: str, url: str, payload: object | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    parsed = urlsplit(url)
    request = Request(
        url,
        data=body,
        method=method,
        headers={
            "Content-Type": "application/json",
            "Origin": f"{parsed.scheme}://{parsed.netloc}",
            "X-Akasic-CSRF": "1",
        },
    )
    try:
        with urlopen(request, timeout=10) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise HTTPError(
            error.url,
            error.code,
            f"{error.reason}: {detail}",
            error.headers,
            None,
        ) from error
    if not isinstance(result, dict):
        raise ValueError(f"model settings returned non-object: {url}")
    return result


def wait_for_settings(settings_url: str, timeout: float = 30.0) -> dict:
    """Wait until the public model catalog is ready."""

    deadline = time.monotonic() + timeout
    while True:
        try:
            return _request("GET", f"{settings_url}/catalog")
        except (OSError, URLError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def add_openai_models(
    settings_url: str,
    *,
    connection_id: str,
    endpoint: str,
    api_key: str,
    chat_model: str | None = None,
    context_window: int = 64_000,
    reasoning_effort: str | None = None,
    embedding_model: str | None = None,
    embedding_dimensions: int | None = None,
    allow_unverified_manual: bool = False,
) -> None:
    """Add and select one connection using only the public model contract."""

    catalog = wait_for_settings(settings_url)
    models = catalog.get("models")
    if not isinstance(models, list):
        raise ValueError("model catalog is missing models")
    requested = {
        f"{connection_id}:chat": chat_model,
        f"{connection_id}:embedding": embedding_model,
    }
    existing = {
        str(item.get("id")): str(item.get("model"))
        for item in models
        if isinstance(item, Mapping)
    }
    if any(model_id in existing for model_id in requested):
        if any(
            value is not None and existing.get(model_id) != value
            for model_id, value in requested.items()
        ):
            raise ValueError(f"fixture connection conflicts: {connection_id}")
        return
    revision = int(catalog["revision"])
    receipt = _request(
        "POST",
        f"{settings_url}/command",
        {
            "type": "add_connection",
            "expected_revision": revision,
            "connection_id": connection_id,
            "name": connection_id,
            "driver_id": "openai-compatible",
            "endpoint": endpoint,
            "auth_identity": connection_id,
            "credential": {"driver": "api_key", "access_token": api_key},
            "driver_config": {
                "allow_unverified_manual": allow_unverified_manual,
            },
        },
    )
    revision = int(receipt["revision"])
    if chat_model is not None:
        model_id = f"{connection_id}:chat"
        receipt = _request(
            "POST",
            f"{settings_url}/command",
            {
                "type": "add_model",
                "expected_revision": revision,
                "model_id": model_id,
                "connection_id": connection_id,
                "kind": "chat",
                "model": chat_model,
                "default_reasoning_effort": reasoning_effort,
                "capabilities": {
                    "context_window": context_window,
                    "input_modalities": ["text"],
                    "supports_tool_calls": True,
                    "supported_reasoning_efforts": (
                        [reasoning_effort] if reasoning_effort else []
                    ),
                },
                "capability_sources": {"context_window": "fixture"},
                "driver_config": {},
            },
        )
        revision = int(receipt["revision"])
        for role in ("default", "fast", "agent"):
            receipt = _request(
                "POST",
                f"{settings_url}/command",
                {
                    "type": "set_default",
                    "expected_revision": revision,
                    "role": role,
                    "model_id": model_id,
                },
            )
            revision = int(receipt["revision"])
    if embedding_model is not None:
        model_id = f"{connection_id}:embedding"
        receipt = _request(
            "POST",
            f"{settings_url}/command",
            {
                "type": "add_model",
                "expected_revision": revision,
                "model_id": model_id,
                "connection_id": connection_id,
                "kind": "embedding",
                "model": embedding_model,
                "capabilities": {
                    "embedding_dimensions": embedding_dimensions,
                    "embedding_normalization": "none",
                },
                "capability_sources": {
                    "embedding_dimensions": "fixture",
                    "embedding_normalization": "fixture",
                },
                "driver_config": {},
            },
        )
        revision = int(receipt["revision"])
        _request(
            "POST",
            f"{settings_url}/command",
            {
                "type": "set_default",
                "expected_revision": revision,
                "role": None,
                "model_id": model_id,
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings-url", required=True)
    parser.add_argument("--connection", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--chat-model")
    parser.add_argument("--context-window", type=int, default=64_000)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-dimensions", type=int)
    arguments = parser.parse_args()
    api_key = os.environ.get(arguments.api_key_env, "").strip()
    if not api_key:
        raise ValueError(f"missing credential environment: {arguments.api_key_env}")
    add_openai_models(
        arguments.settings_url,
        connection_id=arguments.connection,
        endpoint=arguments.endpoint,
        api_key=api_key,
        chat_model=arguments.chat_model,
        context_window=arguments.context_window,
        reasoning_effort=arguments.reasoning_effort,
        embedding_model=arguments.embedding_model,
        embedding_dimensions=arguments.embedding_dimensions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
