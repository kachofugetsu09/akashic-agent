from __future__ import annotations

import asyncio
import fcntl
import hashlib
import importlib.metadata
import json
import logging
import os
import tempfile
import zlib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import httpx

from agent.model_runtime.catalog.litellm_registry import resolve_catalog_capabilities
from agent.plugin_composition import DiscoveredModel

logger = logging.getLogger(__name__)

_REMOTE_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
_CACHE_SCHEMA_VERSION = 1
_MAX_BODY_BYTES = 8 * 1024 * 1024
_MAX_ENTRIES = 20_000


class CatalogError(ValueError):
    """A public capability catalog failed validation."""


class LiteLlmCapabilityCatalog:
    """Refresh LiteLLM capability facts during an explicit model sync."""

    def __init__(
        self,
        cache_path: Path,
        *,
        writable: bool,
        transport: httpx.AsyncBaseTransport | None = None,
        bundled_models: Mapping[str, Mapping[str, Any]] | None = None,
        minimum_entries: int | None = None,
    ) -> None:
        self._cache_path = cache_path
        self._writable = writable
        self._transport = transport
        self._bundled_models = bundled_models
        self._minimum_entries = minimum_entries
        self._lock = asyncio.Lock()

    async def enrich(
        self,
        discovered: tuple[DiscoveredModel, ...],
        *,
        provider_id: str,
    ) -> tuple[DiscoveredModel, ...]:
        """Fill unknown image-input facts without changing provider inventory."""

        async with self._lock:
            models, source = await self._load_for_sync()
        return tuple(
            _enrich_model(item, provider_id=provider_id, models=models, source=source)
            for item in discovered
        )

    async def _load_for_sync(self) -> tuple[dict[str, dict[str, Any]], str]:
        if not self._writable:
            return await self._load_for_sync_unlocked()
        try:
            lock_fd = await _acquire_file_lock(self._cache_path)
        except OSError as exc:
            logger.warning("LiteLLM 能力目录锁不可用，跳过缓存写入: %s", exc)
            writable = self._writable
            self._writable = False
            try:
                return await self._load_for_sync_unlocked()
            finally:
                self._writable = writable
        try:
            return await self._load_for_sync_unlocked()
        finally:
            await asyncio.to_thread(_release_file_lock, lock_fd)

    async def _load_for_sync_unlocked(
        self,
    ) -> tuple[dict[str, dict[str, Any]], str]:
        bundled = self._load_bundled()
        minimum_entries = (
            self._minimum_entries
            if self._minimum_entries is not None
            else max(1, len(bundled) * 3 // 4)
        )
        cached = self._load_cache(minimum_entries=minimum_entries)
        etag = cached[2] if cached is not None else ""
        try:
            remote = await self._fetch_remote(
                etag=etag,
                minimum_entries=minimum_entries,
            )
        except (httpx.HTTPError, CatalogError, UnicodeDecodeError) as exc:
            logger.warning("LiteLLM 能力目录刷新失败，使用本地快照: %s", exc)
        else:
            if remote is None:
                if cached is not None:
                    return cached[0], _source("remote", cached[1])
            else:
                models, digest, response_etag = remote
                if self._writable:
                    try:
                        self._save_cache(models, digest=digest, etag=response_etag)
                    except OSError as exc:
                        logger.warning("LiteLLM 能力目录缓存写入失败: %s", exc)
                return models, _source("remote", digest)

        if cached is not None:
            return cached[0], _source("remote", cached[1])
        return bundled, f"litellm-wheel@{importlib.metadata.version('litellm')}"

    async def _fetch_remote(
        self,
        *,
        etag: str,
        minimum_entries: int,
    ) -> tuple[dict[str, dict[str, Any]], str, str] | None:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip, identity",
        }
        if etag:
            headers["If-None-Match"] = etag
        timeout = httpx.Timeout(5.0, connect=3.0)
        async with httpx.AsyncClient(
            transport=self._transport,
            timeout=timeout,
            follow_redirects=False,
        ) as client:
            async with client.stream("GET", _REMOTE_URL, headers=headers) as response:
                if response.status_code == 304:
                    return None
                response.raise_for_status()
                body = await _read_limited_body(response)
        try:
            raw = json.loads(body.decode("utf-8"))
            models = _validate_models(raw, minimum_entries=minimum_entries)
            digest = hashlib.sha256(_models_bytes(models)).hexdigest()
        except (json.JSONDecodeError, RecursionError) as exc:
            raise CatalogError("响应不是有效 JSON") from exc
        return models, digest, response.headers.get("etag", "")

    def _load_bundled(self) -> dict[str, dict[str, Any]]:
        if self._bundled_models is not None:
            return _validate_models(self._bundled_models, minimum_entries=1)
        distribution = importlib.metadata.distribution("litellm")
        path = Path(
            str(
                distribution.locate_file(
                    "litellm/model_prices_and_context_window_backup.json"
                )
            )
        )
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, RecursionError) as exc:
            raise CatalogError("随包 LiteLLM 目录损坏") from exc
        return _validate_models(raw, minimum_entries=1)

    def _load_cache(
        self,
        *,
        minimum_entries: int,
    ) -> tuple[dict[str, dict[str, Any]], str, str] | None:
        try:
            raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or raw.get("schema_version") != 1:
                raise CatalogError("缓存 schema 不受支持")
            models = _validate_models(
                raw.get("models"),
                minimum_entries=minimum_entries,
            )
            digest = hashlib.sha256(_models_bytes(models)).hexdigest()
            if raw.get("sha256") != digest:
                raise CatalogError("缓存摘要不匹配")
            etag = raw.get("etag")
            if not isinstance(etag, str):
                raise CatalogError("缓存 ETag 无效")
            return models, digest, etag
        except FileNotFoundError:
            return None
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            RecursionError,
            CatalogError,
        ) as exc:
            logger.warning("忽略不可用的 LiteLLM 能力目录缓存: %s", exc)
            return None

    def _save_cache(
        self,
        models: Mapping[str, Mapping[str, Any]],
        *,
        digest: str,
        etag: str,
    ) -> None:
        """Publish one complete cache envelope with an atomic rename."""

        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "schema_version": _CACHE_SCHEMA_VERSION,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "etag": etag,
                "sha256": digest,
                "models": models,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        fd, temp_name = tempfile.mkstemp(
            dir=self._cache_path.parent,
            prefix=f".{self._cache_path.name}.",
        )
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self._cache_path)
            dir_fd = os.open(self._cache_path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise


async def _read_limited_body(response: httpx.Response) -> bytes:
    """Read raw HTTP bytes and cap both compressed and decoded content."""

    encoding = response.headers.get("content-encoding", "").strip().lower()
    if encoding in {"", "identity"}:
        decoder: Any | None = None
    elif encoding == "gzip":
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    else:
        raise CatalogError(f"不支持的 Content-Encoding: {encoding}")
    raw_length = response.headers.get("content-length")
    if raw_length is not None:
        try:
            content_length = int(raw_length)
        except ValueError as error:
            raise CatalogError("响应 Content-Length 无效") from error
        if content_length < 0 or content_length > _MAX_BODY_BYTES:
            raise CatalogError("压缩响应超过 8 MiB")

    body = bytearray()
    raw_bytes = 0

    async def raw_chunks():
        if response.is_stream_consumed:
            yield response.content
            return
        async for raw_chunk in response.aiter_raw(chunk_size=64 * 1024):
            yield raw_chunk

    try:
        async for chunk in raw_chunks():
            raw_bytes += len(chunk)
            if raw_bytes > _MAX_BODY_BYTES:
                raise CatalogError("压缩响应超过 8 MiB")
            if decoder is None:
                decoded = chunk
            else:
                remaining = _MAX_BODY_BYTES - len(body)
                decoded = decoder.decompress(chunk, remaining + 1)
                if decoder.unconsumed_tail:
                    raise CatalogError("解压响应超过 8 MiB")
            if len(body) + len(decoded) > _MAX_BODY_BYTES:
                raise CatalogError("解压响应超过 8 MiB")
            body.extend(decoded)
        if decoder is not None:
            remaining = _MAX_BODY_BYTES - len(body)
            decoded = decoder.flush(remaining + 1)
            if len(body) + len(decoded) > _MAX_BODY_BYTES:
                raise CatalogError("解压响应超过 8 MiB")
            body.extend(decoded)
            if not decoder.eof or decoder.unused_data:
                raise CatalogError("gzip 响应无效")
    except zlib.error as error:
        raise CatalogError("gzip 响应无效") from error
    return bytes(body)


async def _acquire_file_lock(cache_path: Path) -> int:
    """Acquire the cache transaction lock with cancellation-safe polling."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(f"{cache_path}.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return lock_fd
            except BlockingIOError:
                await asyncio.sleep(0.05)
    except BaseException:
        os.close(lock_fd)
        raise


def _release_file_lock(lock_fd: int) -> None:
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(lock_fd)


def _validate_models(
    raw: object,
    *,
    minimum_entries: int,
) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, Mapping):
        raise CatalogError("目录根节点必须是对象")
    if not minimum_entries <= len(raw) <= _MAX_ENTRIES:
        raise CatalogError(f"目录条目数异常: {len(raw)}")
    result: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key or not isinstance(value, Mapping):
            raise CatalogError("目录包含无效条目")
        result[key] = dict(value)
    return result


def _enrich_model(
    model: DiscoveredModel,
    *,
    provider_id: str,
    models: Mapping[str, Mapping[str, Any]],
    source: str,
) -> DiscoveredModel:
    resolved = resolve_catalog_capabilities(
        provider_id,
        model.model,
        models=models,
    )
    if resolved is None:
        return model
    capabilities = model.capabilities
    sources = model.capability_sources
    if sources.context_window == "unknown" and resolved.context_window > 0:
        capabilities = replace(
            capabilities,
            context_window=resolved.context_window,
        )
        sources = replace(sources, context_window=source)
    if sources.max_output_tokens == "unknown" and resolved.max_output_tokens > 0:
        capabilities = replace(
            capabilities,
            max_output_tokens=resolved.max_output_tokens,
        )
        sources = replace(sources, max_output_tokens=source)
    if sources.input_modalities == "unknown" and resolved.input_modalities_known:
        capabilities = replace(
            capabilities,
            input_modalities=resolved.input_modalities,
        )
        sources = replace(sources, input_modalities=source)
    if (
        sources.reasoning_efforts == "unknown"
        and resolved.supported_reasoning_efforts
    ):
        capabilities = replace(
            capabilities,
            supported_reasoning_efforts=resolved.supported_reasoning_efforts,
        )
        sources = replace(sources, reasoning_efforts=source)
    return replace(model, capabilities=capabilities, capability_sources=sources)


def _models_bytes(models: Mapping[str, Mapping[str, Any]]) -> bytes:
    return json.dumps(
        models,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _source(kind: str, digest: str) -> str:
    return f"litellm-{kind}@sha256:{digest[:16]}"
