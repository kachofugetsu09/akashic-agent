from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import socket
import tempfile
import threading
import tomllib
from collections.abc import MutableMapping
from pathlib import Path
from typing import Callable, Literal
from uuid import uuid4

import tomlkit
import uvicorn
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.config import Config
from agent.model_runtime.auth.codex import CodexAuthDriver
from agent.model_runtime.auth.store import Credential, CredentialStore
from agent.model_runtime.catalog.codex import CodexModelCatalog
from agent.model_runtime.catalog.opencode_go import OpenCodeGoModelCatalog
from agent.model_runtime.catalog.litellm_registry import (
    CatalogCapabilities,
    resolve_catalog_capabilities,
    resolve_catalog_provider_id,
)
from agent.model_runtime.errors import AuthenticationError, ModelRuntimeError, TransportError
from agent.model_runtime.store import ModelRegistryStore
from agent.provider import LLMProvider
from bootstrap.setup_main import patch_main_model_config
from bootstrap.setup_wizard import WizardAnswers


class SettingsServer(uvicorn.Server):
    """在线程边界发布一次确定的监听启动结果。"""

    def __init__(self, config: uvicorn.Config) -> None:
        super().__init__(config)
        self.startup_event = threading.Event()

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        try:
            await super().startup(sockets=sockets)
        finally:
            self.startup_event.set()


class ModelQuery(BaseModel):
    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(default="", max_length=200)
    api_key: str = ""
    credential_id: str = ""
    use_local_opencode: bool = False
    base_url: str = ""


class ApplyPayload(BaseModel):
    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=200)
    source_id: str = Field(default="", max_length=96)
    source_name: str = Field(default="", max_length=80)
    api_key: str = ""
    credential_id: str = ""
    use_local_opencode: bool = False
    base_url: str = Field(default="", max_length=2048)
    context_window: int = Field(default=0, ge=0)
    max_output_tokens: int = Field(default=0, ge=0)
    input_modalities: list[Literal["text", "image"]] | None = None
    reasoning_effort: str = Field(default="", max_length=32)
    expected_config_revision: str = Field(default="", max_length=64)


class RoleBindingPayload(BaseModel):
    role: Literal["default", "fast", "agent", "vision"]
    model_id: str = Field(min_length=1, max_length=128)
    reasoning_effort: str = Field(default="", max_length=32)
    expected_revision: int | None = Field(default=None, ge=0)


class CodexLoginSession:
    """保存一次 device login 的非敏感浏览器可见状态。"""

    def __init__(self, login_id: str, driver: CodexAuthDriver) -> None:
        self.login_id = login_id
        self.driver = driver
        self.code = driver.begin_device_login()
        self.status = "waiting"
        self.error = ""


def create_settings_app(
    config_path: Path,
    workspace: Path,
    *,
    credential_store: CredentialStore | None = None,
    on_applied: Callable[[], None] | None = None,
) -> FastAPI:
    """创建只监听本机的设置 API，并提供脱敏状态与原子配置应用。"""
    store = credential_store or CredentialStore.for_workspace(workspace)
    apply_lock = threading.Lock()
    login_lock = threading.Lock()
    logins: dict[str, CodexLoginSession] = {}
    project_root = Path(__file__).resolve().parent.parent
    static_dir = project_root / "static" / "chat"
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(Exception)
    async def internal_error(_request: Request, exc: Exception):
        logging.getLogger(__name__).exception("settings request failed", exc_info=exc)
        return _error_response(500, "internal_error", "设置操作失败，请查看服务端日志")

    @app.middleware("http")
    async def protect_settings_boundary(request: Request, call_next):
        # 1. 修改请求必须来自同源页面并携带显式 CSRF header。
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin", "")
            expected = f"http://{request.url.netloc}"
            if origin != expected or request.headers.get("x-akasic-csrf") != "1":
                return _error_response(403, "csrf_rejected", "请求来源无效")

        # 2. 所有响应禁止缓存和跨页面泄露引用信息。
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; "
            "frame-ancestors 'self' "
            "http://127.0.0.1:5173 http://localhost:5173"
        )
        return response

    @app.get("/api/settings/state")
    async def state() -> dict[str, object]:
        return _read_settings_state(config_path, workspace, store)

    @app.post("/api/settings/models")
    async def models(payload: ModelQuery) -> dict[str, object]:
        try:
            if payload.provider == "codex":
                auth = CodexAuthDriver(store, payload.credential_id or "codex_default")
                entries = await CodexModelCatalog(auth).list_models()
                return {
                    "models": [
                        {
                            "id": entry.slug,
                            "contextWindow": entry.capabilities.context_window,
                            "maxOutputTokens": entry.capabilities.max_output_tokens,
                            "inputModalities": list(entry.capabilities.input_modalities),
                            "supportedReasoningEfforts": list(
                                entry.capabilities.supported_reasoning_efforts
                            ),
                            "defaultReasoningEffort": (
                                entry.capabilities.default_reasoning_effort
                            ),
                        }
                        for entry in entries
                    ]
                }
            if payload.provider == "opencode-go":
                key = _candidate_api_key(
                    payload.api_key,
                    config_path,
                    workspace,
                    store,
                    payload.provider,
                    use_local_opencode=payload.use_local_opencode,
                    credential_id=payload.credential_id.strip(),
                )
                entries = await OpenCodeGoModelCatalog(
                    key,
                    base_url=payload.base_url or "https://opencode.ai/zen/go/v1",
                ).list_models()
                return {
                    "models": [
                        {
                            "id": entry.slug,
                            "supportedReasoningEfforts": list(
                                entry.supported_reasoning_efforts
                            ),
                        }
                        for entry in entries
                    ]
                }
            capabilities = resolve_catalog_capabilities(
                payload.provider,
                payload.model.strip(),
                base_url=payload.base_url.strip(),
            )
            if capabilities is not None:
                return {
                    "models": [
                        {
                            "id": payload.model.strip(),
                            "contextWindow": capabilities.context_window,
                            "maxOutputTokens": capabilities.max_output_tokens,
                            "inputModalities": list(capabilities.input_modalities),
                            "supportedReasoningEfforts": list(
                                capabilities.supported_reasoning_efforts
                            ),
                        }
                    ]
                }
            if not payload.model.strip():
                key = _candidate_api_key(
                    payload.api_key,
                    config_path,
                    workspace,
                    store,
                    payload.provider,
                    use_local_opencode=payload.use_local_opencode,
                    credential_id=payload.credential_id.strip(),
                )
                base_url = payload.base_url.strip().rstrip("/")
                if not base_url.startswith(("https://", "http://")):
                    raise TransportError("Base URL 必须是 http(s) 地址")
                async with httpx.AsyncClient(timeout=15.0) as client:
                    response = await client.get(
                        f"{base_url}/models",
                        headers={"Authorization": f"Bearer {key}"},
                    )
                    response.raise_for_status()
                body = response.json()
                rows = body.get("data") if isinstance(body, dict) else None
                if not isinstance(rows, list):
                    raise TransportError("模型目录响应缺少 data 数组")
                model_ids = sorted(
                    {
                        str(row.get("id") or "").strip()
                        for row in rows
                        if isinstance(row, dict) and str(row.get("id") or "").strip()
                    }
                )
                models: list[dict[str, object]] = []
                for model_id in model_ids:
                    item: dict[str, object] = {"id": model_id}
                    model_capabilities = resolve_catalog_capabilities(
                        payload.provider,
                        model_id,
                        base_url=payload.base_url.strip(),
                    )
                    if model_capabilities is not None:
                        item.update(
                            {
                                "contextWindow": model_capabilities.context_window,
                                "maxOutputTokens": model_capabilities.max_output_tokens,
                                "inputModalities": list(
                                    model_capabilities.input_modalities
                                ),
                                "supportedReasoningEfforts": list(
                                    model_capabilities.supported_reasoning_efforts
                                ),
                            }
                        )
                    models.append(item)
                return {"models": models}
            return {"models": []}
        except (AuthenticationError, TransportError, httpx.HTTPError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/settings/apply")
    async def apply(payload: ApplyPayload) -> dict[str, object]:
        if (
            payload.max_output_tokens > 0
            and payload.context_window > 0
            and payload.max_output_tokens >= payload.context_window
        ):
            raise HTTPException(status_code=422, detail="最大输出必须小于上下文窗口")
        if payload.input_modalities is not None and "text" not in payload.input_modalities:
            raise HTTPException(status_code=422, detail="输入模态必须包含 text")
        if not apply_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="已有设置操作正在执行")
        try:
            current_revision = _settings_revision(config_path, workspace)
            if (
                payload.expected_config_revision
                and payload.expected_config_revision != current_revision
            ):
                raise HTTPException(status_code=409, detail="配置已经变化，请刷新后重试")
            operation_id = f"settings-{uuid4().hex}"
            provider_capabilities = await _provider_catalog_capabilities(
                payload,
                store,
            )
            answers = _answers(
                payload,
                config_path,
                workspace,
                store,
                provider_capabilities=provider_capabilities,
            )
            await _validate_live_candidate(answers, store)
            _apply_candidate(
                config_path,
                workspace,
                answers,
                operation_id,
                credential_store=store,
                on_applied=on_applied,
            )
            return {"operationId": operation_id, "status": "applied"}
        finally:
            apply_lock.release()

    @app.post("/api/settings/roles")
    async def set_role(payload: RoleBindingPayload) -> dict[str, object]:
        if not apply_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="已有设置操作正在执行")
        try:
            registry = ModelRegistryStore.for_workspace(workspace)
            try:
                revision = registry.set_role(
                    payload.role,
                    payload.model_id.strip(),
                    reasoning_effort=payload.reasoning_effort,
                    expected_revision=payload.expected_revision,
                )
            except RuntimeError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return {"status": "applied", "revision": revision}
        finally:
            apply_lock.release()

    @app.post("/api/settings/codex-login")
    async def begin_codex_login() -> dict[str, object]:
        login_id = f"codex-{uuid4().hex}"
        try:
            store.provision_connection(
                "codex_default",
                name="Codex",
                provider="codex",
                base_url="https://chatgpt.com/backend-api/codex",
            )
            session = await asyncio.to_thread(
                CodexLoginSession,
                login_id,
                CodexAuthDriver(store, "codex_default"),
            )
        except ModelRuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        with login_lock:
            logins[login_id] = session
        threading.Thread(
            target=_complete_codex_login,
            args=(session, login_lock),
            name=f"settings-{login_id}",
            daemon=True,
        ).start()
        return _codex_login_state(session)

    @app.get("/api/settings/codex-login/{login_id}")
    async def codex_login_status(login_id: str) -> dict[str, object]:
        with login_lock:
            session = logins.get(login_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Codex 登录会话不存在")
            return _codex_login_state(session)

    @app.get("/chat")
    @app.get("/chat/")
    @app.get("/settings")
    @app.get("/settings/")
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    app.mount("/assets", StaticFiles(directory=static_dir, check_dir=False), name="settings-assets")
    return app


def _error_response(status_code: int, code: str, message: str):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status_code, content={"code": code, "message": message})


def _complete_codex_login(
    session: CodexLoginSession,
    lock: threading.Lock,
) -> None:
    """在后台完成阻塞轮询，只发布状态而不发布 token。"""
    try:
        session.driver.complete_device_login(session.code)
    except ModelRuntimeError as exc:
        with lock:
            session.status = "failed"
            session.error = str(exc)
        return
    with lock:
        session.status = "completed"


def _codex_login_state(session: CodexLoginSession) -> dict[str, object]:
    return {
        "loginId": session.login_id,
        "status": session.status,
        "userCode": session.code.user_code,
        "verificationUri": session.code.verification_uri,
        "interval": session.code.interval,
        "error": session.error,
    }


def _read_settings_state(
    config_path: Path,
    workspace: Path,
    store: CredentialStore,
) -> dict[str, object]:
    """读取配置结构和凭据元数据，不解析或返回任何 secret。"""
    local_opencode = _local_opencode_key(required=False) is not None
    model_store = ModelRegistryStore.for_workspace(workspace)
    model_snapshot = model_store.read_snapshot()
    credential_meta = (
        CredentialStore.for_workspace(workspace).metadata()
        if model_store.exists()
        else store.metadata()
    )
    if not config_path.exists() and model_snapshot is None:
        return {
            "mode": "needs_setup",
            "workspace": str(workspace),
            "activeRuntime": None,
            "runtimes": [],
            "roleBindings": {},
            "modelRevision": 0,
            "codexConfigured": "codex_default" in credential_meta,
            "localOpenCodeConfigured": local_opencode,
            "configRevision": "",
        }
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {
            "mode": "needs_repair",
            "workspace": str(workspace),
            "error": "config.toml 无法解析，请先修复或移走该文件",
            "activeRuntime": None,
            "runtimes": [],
            "roleBindings": {},
            "modelRevision": 0,
            "codexConfigured": "codex_default" in credential_meta,
            "localOpenCodeConfigured": local_opencode,
            "configRevision": "",
        }
    llm = raw.get("llm") if isinstance(raw.get("llm"), dict) else {}
    active = (
        model_snapshot.roles["default"].runtime_id
        if model_snapshot is not None
        else llm.get("main")
    )
    role_bindings: dict[str, dict[str, str]] = {}
    try:
        if model_snapshot is not None:
            runtimes = [
                _runtime_summary(
                    runtime_id,
                    runtime.as_config_table(),
                    credential_meta,
                )
                for runtime_id, runtime in model_snapshot.runtimes.items()
            ]
            role_bindings = {
                role: {
                    "modelId": binding.runtime_id,
                    "reasoningEffort": binding.reasoning_effort,
                }
                for role, binding in model_snapshot.roles.items()
            }
            mode = "ready"
        elif isinstance(active, str):
            runtimes_raw = llm.get("runtimes") if isinstance(llm.get("runtimes"), dict) else {}
            agent = raw.get("agent") if isinstance(raw.get("agent"), dict) else {}
            legacy_max_output_tokens = agent.get(
                "max_tokens",
                raw.get("max_tokens", 0),
            )
            runtimes = [
                _runtime_summary(
                    runtime_id,
                    value,
                    credential_meta,
                    missing_max_output_tokens=(
                        legacy_max_output_tokens if runtime_id == active else 0
                    ),
                )
                for runtime_id, value in runtimes_raw.items()
                if isinstance(runtime_id, str) and isinstance(value, dict)
            ]
            mode = "ready"
        else:
            runtimes = [_runtime_summary("legacy_main", active, credential_meta)] if isinstance(active, dict) else []
            active = "legacy_main" if runtimes else None
            mode = "needs_repair" if runtimes else "needs_setup"
    except ValueError as exc:
        return {
            "mode": "needs_repair",
            "workspace": str(workspace),
            "error": str(exc),
            "activeRuntime": None,
            "runtimes": [],
            "roleBindings": {},
            "modelRevision": model_snapshot.revision if model_snapshot else 0,
            "codexConfigured": "codex_default" in credential_meta,
            "localOpenCodeConfigured": local_opencode,
            "configRevision": "",
        }
    return {
        "mode": mode,
        "workspace": str(workspace),
        "activeRuntime": active,
        "runtimes": runtimes,
        "roleBindings": role_bindings,
        "modelRevision": model_snapshot.revision if model_snapshot else 0,
        "codexConfigured": "codex_default" in credential_meta,
        "localOpenCodeConfigured": local_opencode,
        "configRevision": _settings_revision(config_path, workspace),
    }


def _runtime_summary(
    runtime_id: str,
    raw: dict[str, object],
    credential_meta: dict[str, dict[str, str]],
    *,
    missing_max_output_tokens: object = 0,
) -> dict[str, object]:
    context_window = raw.get("context_window", 0)
    max_output_tokens = raw.get(
        "max_output_tokens",
        missing_max_output_tokens,
    )
    input_modalities = raw.get("input_modalities", ["text"])
    if isinstance(context_window, bool) or not isinstance(context_window, int):
        raise ValueError(f"runtime {runtime_id} 的 context_window 必须是整数")
    if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int):
        raise ValueError(f"runtime {runtime_id} 的 max_output_tokens 必须是整数")
    if not isinstance(input_modalities, list) or not all(
        isinstance(item, str) for item in input_modalities
    ):
        raise ValueError(f"runtime {runtime_id} 的 input_modalities 必须是字符串数组")
    auth = str(raw.get("auth") or "")
    inline = str(raw.get("api_key") or "")
    source = "credential_store" if auth else ("environment" if inline.startswith("${") else "inline" if inline else "none")
    return {
        "id": runtime_id,
        "provider": str(raw.get("provider") or ""),
        "model": str(raw.get("model") or ""),
        "sourceId": str(raw.get("source_id") or ""),
        "sourceName": str(raw.get("source_name") or raw.get("provider") or ""),
        "catalogProvider": str(
            raw.get("catalog_provider_id") or raw.get("provider") or ""
        ),
        "baseUrl": str(raw.get("base_url") or ""),
        "contextWindow": context_window,
        "maxOutputTokens": max_output_tokens,
        "inputModalities": input_modalities,
        "capabilitySource": str(raw.get("capability_source") or "unknown"),
        "capabilitySources": {
            "contextWindow": str(raw.get("context_window_source") or raw.get("capability_source") or "unknown"),
            "maxOutputTokens": str(raw.get("max_output_tokens_source") or raw.get("capability_source") or "unknown"),
            "inputModalities": str(raw.get("input_modalities_source") or raw.get("capability_source") or "unknown"),
        },
        "reasoningEffort": str(raw.get("reasoning_effort") or ""),
        "supportedReasoningEfforts": list(
            raw.get("supported_reasoning_efforts")
            if isinstance(raw.get("supported_reasoning_efforts"), list)
            else []
        ),
        "credential": {
            "id": auth,
            "configured": bool(auth and auth in credential_meta) or bool(inline),
            "source": source,
        },
    }


def _candidate_api_key(
    value: str,
    config_path: Path,
    workspace: Path,
    store: CredentialStore,
    provider: str,
    *,
    use_local_opencode: bool,
    credential_id: str = "",
) -> str:
    if value:
        return value
    if use_local_opencode:
        key = _local_opencode_key(required=True)
        assert key is not None
        return key
    if credential_id:
        try:
            return store.api_key(credential_id)
        except AuthenticationError:
            pass
    saved = _saved_api_key(config_path, workspace, store, provider)
    if saved:
        return saved
    raise AuthenticationError("API Key 不能为空")


def _answers(
    payload: ApplyPayload,
    config_path: Path,
    workspace: Path,
    store: CredentialStore,
    *,
    provider_capabilities: CatalogCapabilities | None = None,
) -> WizardAnswers:
    requested_provider = payload.provider.strip().lower()
    catalog_provider_id = resolve_catalog_provider_id(
        requested_provider,
        model=payload.model.strip(),
        base_url=payload.base_url.strip(),
    )
    provider = (
        catalog_provider_id
        if requested_provider not in {"codex", "opencode-go"}
        and catalog_provider_id
        else requested_provider
    )
    requested_source_id = payload.source_id.strip()
    if requested_source_id and (
        not requested_source_id.replace("-", "").replace("_", "").isalnum()
        or requested_source_id.startswith("__")
    ):
        raise ValueError("source_id 只能包含字母、数字、连字符和下划线")
    legacy_runtime_id = f"{provider.strip().lower().replace('-', '_')}_main"
    runtime_id = (
        f"{requested_source_id}__{hashlib.sha256(payload.model.strip().encode('utf-8')).hexdigest()[:10]}"
        if requested_source_id
        else legacy_runtime_id
    )
    source_id = requested_source_id or f"source:{legacy_runtime_id}"
    source_name = payload.source_name.strip() or provider
    auth_id = payload.credential_id.strip()
    if provider == "codex":
        auth_id = auth_id or "codex_default"
        _ = store.get(auth_id)
        source_id = auth_id
        if requested_source_id:
            runtime_id = (
                f"{auth_id}__"
                f"{hashlib.sha256(payload.model.strip().encode('utf-8')).hexdigest()[:10]}"
            )
        api_key = ""
    else:
        credential_suffix = requested_source_id or legacy_runtime_id
        auth_id = auth_id or f"model_{credential_suffix}"
        api_key = _candidate_api_key(
            payload.api_key,
            config_path,
            workspace,
            store,
            provider,
            use_local_opencode=payload.use_local_opencode,
            credential_id=auth_id,
        )
    capabilities = provider_capabilities or resolve_catalog_capabilities(
        provider, payload.model.strip(), base_url=payload.base_url.strip()
    )
    context_window = payload.context_window or (
        capabilities.context_window if capabilities is not None else 0
    )
    max_output_tokens = (
        payload.max_output_tokens
        if "max_output_tokens" in payload.model_fields_set
        else capabilities.max_output_tokens
        if capabilities is not None
        else 0
    )
    input_modalities = payload.input_modalities or (
        list(capabilities.input_modalities) if capabilities is not None else ["text"]
    )
    catalog_source = (
        "provider_catalog"
        if provider_capabilities is not None
        else "litellm"
        if capabilities is not None
        else "unknown"
    )
    context_source = (
        "explicit"
        if payload.context_window > 0
        else catalog_source
        if context_window > 0
        else "unknown"
    )
    output_source = (
        "explicit"
        if "max_output_tokens" in payload.model_fields_set
        else catalog_source
        if max_output_tokens > 0
        else "unknown"
    )
    modalities_source = (
        "explicit"
        if payload.input_modalities is not None
        else catalog_source
        if capabilities is not None and capabilities.input_modalities_known
        else "unknown"
    )
    sources = {context_source, output_source, modalities_source}
    capability_source = next(iter(sources)) if len(sources) == 1 else "mixed"
    return WizardAnswers(
        provider=provider,
        runtime_id=runtime_id,
        source_id=source_id,
        source_name=source_name,
        auth_id=auth_id,
        api_key=api_key,
        model=payload.model.strip(),
        catalog_provider_id=catalog_provider_id,
        base_url=payload.base_url.strip(),
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        multimodal="image" in input_modalities,
        capability_source=capability_source,
        context_window_source=context_source,
        max_output_tokens_source=output_source,
        input_modalities_source=modalities_source,
        reasoning_effort=payload.reasoning_effort.strip(),
        supported_reasoning_efforts=(
            tuple(capabilities.supported_reasoning_efforts)
            if capabilities is not None
            else ()
        ),
        supports_parallel_tool_calls=(
            capabilities.supports_parallel_tool_calls
            if capabilities is not None
            else True
        ),
    )


async def _provider_catalog_capabilities(
    payload: ApplyPayload,
    store: CredentialStore,
) -> CatalogCapabilities | None:
    """Resolve capabilities from a provider-owned catalog when available."""

    if payload.provider.strip().lower() != "codex":
        return None

    # Codex exposes authoritative limits with the authenticated model list.
    auth_id = payload.credential_id.strip() or "codex_default"
    entries = await CodexModelCatalog(CodexAuthDriver(store, auth_id)).list_models()
    model = payload.model.strip()
    entry = next((candidate for candidate in entries if candidate.slug == model), None)
    if entry is None:
        raise TransportError(f"Codex 模型目录不存在: {model}")
    caps = entry.capabilities
    return CatalogCapabilities(
        context_window=caps.context_window,
        max_output_tokens=caps.max_output_tokens,
        input_modalities=tuple(caps.input_modalities),
        input_modalities_known=entry.input_modalities_known,
        reasoning=bool(caps.supported_reasoning_efforts),
        tool_call=True,
        supported_reasoning_efforts=tuple(caps.supported_reasoning_efforts),
        supports_parallel_tool_calls=caps.supports_parallel_tool_calls,
        source="provider_catalog",
    )


def _settings_revision(config_path: Path, workspace: Path) -> str:
    """Return the canonical model revision, falling back before migration."""

    registry_revision = ModelRegistryStore.for_workspace(workspace).revision()
    if registry_revision:
        return f"models:{registry_revision}"

    if not config_path.exists():
        return ""
    stat = config_path.stat()
    identity = f"{stat.st_dev}:{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(identity.encode("ascii")).hexdigest()


async def _validate_live_candidate(
    answers: WizardAnswers,
    store: CredentialStore,
) -> None:
    """通过真实 Provider 请求验证候选认证、模型和 wire protocol。"""
    if answers.provider == "codex":
        models = await CodexModelCatalog(
            CodexAuthDriver(store, answers.auth_id)
        ).list_models()
        if answers.model not in {model.slug for model in models}:
            raise TransportError(f"Codex 模型目录中不存在 {answers.model}")
        return
    provider = LLMProvider(
        api_key=answers.api_key,
        base_url=answers.base_url,
        provider_name=answers.provider,
        max_retries=0,
    )
    await provider.chat(
        [{"role": "user", "content": "Reply with OK."}],
        [],
        answers.model,
        8,
    )


def _apply_candidate(
    config_path: Path,
    workspace: Path,
    answers: WizardAnswers,
    operation_id: str,
    *,
    credential_store: CredentialStore,
    on_applied: Callable[[], None] | None,
) -> None:
    """Back up and atomically publish one database-owned model candidate."""

    model_store = ModelRegistryStore.for_workspace(workspace)
    original = (
        config_path.read_text(encoding="utf-8")
        if config_path.exists()
        else _new_config(workspace)
    )
    candidate_source = _candidate_source_document(original, model_store)
    candidate = patch_main_model_config(candidate_source, answers)
    candidate_document = tomlkit.parse(candidate)
    candidate_llm = candidate_document.get("llm")
    if not isinstance(candidate_llm, MutableMapping):
        raise ValueError("候选模型配置缺少 llm table")
    runtimes = candidate_llm.get("runtimes")
    runtime_id = answers.runtime_id or (
        f"{answers.provider.strip().lower().replace('-', '_')}_main"
    )
    runtime = (
        runtimes.get(runtime_id)
        if isinstance(runtimes, MutableMapping)
        else None
    )
    if not isinstance(runtime, MutableMapping):
        raise ValueError(f"候选模型 runtime 不存在: {runtime_id}")
    if answers.provider != "codex":
        runtime.pop("api_key", None)
        runtime["auth"] = answers.auth_id

    config_snapshot = config_path.read_bytes() if config_path.exists() else None
    registry_existed = model_store.path.is_file()
    backup_dir = workspace / "backups" / "model-settings" / operation_id
    backup_dir.mkdir(parents=True, exist_ok=False)
    if config_snapshot is not None:
        target = backup_dir / "config.before"
        target.write_bytes(config_snapshot)
        os.chmod(target, 0o600)
    registry_backup = backup_dir / "model-registry.before.sqlite3"
    if registry_existed:
        model_store.backup_to(registry_backup)
    (backup_dir / "manifest.json").write_text(
        json.dumps(
            {
                "operation_id": operation_id,
                "config": config_snapshot is not None,
                "model_registry": registry_existed,
                "credentials": "model-registry.sqlite3",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    backup_path = config_path.with_name(f"{config_path.name}.{operation_id}.bak")
    if config_snapshot is not None:
        backup_path.write_bytes(config_snapshot)
        os.chmod(backup_path, 0o600)

    restart_attempted = False
    try:
        credentials = (
            {answers.auth_id: credential_store.get(answers.auth_id)}
            if answers.provider == "codex"
            else {
                answers.auth_id: Credential(
                    driver="api_key",
                    access_token=answers.api_key,
                )
            }
        )
        _ = model_store.replace_from_llm_config(
            candidate_llm,
            credentials=credentials,
        )
        _atomic_write(config_path, _strip_model_config(original), 0o600)
        _ = Config.load(
            config_path,
            workspace=workspace,
        )
        if config_snapshot is None:
            from bootstrap.init_workspace import init_workspace

            _ = init_workspace(config_path=config_path, workspace=workspace)
        if on_applied is not None:
            restart_attempted = True
            on_applied()
    except BaseException:
        _restore_optional_file(config_path, config_snapshot, 0o600)
        if registry_existed:
            model_store.restore_from(registry_backup)
        else:
            _remove_sqlite_database(model_store.path)
        if (
            on_applied is not None
            and restart_attempted
            and config_snapshot is not None
        ):
            on_applied()
        raise


def _candidate_source_document(original: str, store: ModelRegistryStore) -> str:
    """Combine static TOML with the current database projection for patching."""

    document = tomlkit.parse(original)
    snapshot = store.read_snapshot()
    if snapshot is not None:
        document["llm"] = snapshot.as_config_llm()
    return tomlkit.dumps(document)


def _strip_model_config(original: str) -> str:
    """Keep static process configuration and replace model TOML with a marker."""

    document = tomlkit.parse(original)
    llm = document.get("llm")
    if not isinstance(llm, MutableMapping):
        llm = tomlkit.table()
        document["llm"] = llm
    for key in ("main", "fast", "agent", "vl", "runtimes"):
        llm.pop(key, None)
    llm["registry"] = "workspace"
    return tomlkit.dumps(document)


def _saved_api_key(
    config_path: Path,
    workspace: Path,
    store: CredentialStore,
    provider: str,
) -> str:
    """Read a saved key through the credential owner without exposing it."""

    snapshot = ModelRegistryStore.for_workspace(workspace).read_snapshot()
    if snapshot is not None:
        runtime_id = f"{provider.strip().lower().replace('-', '_')}_main"
        runtime = snapshot.runtimes.get(runtime_id)
        if runtime is not None and runtime.auth_id:
            return CredentialStore.for_workspace(workspace).api_key(runtime.auth_id)
    if not config_path.exists():
        return ""
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    llm = raw.get("llm")
    if not isinstance(llm, dict):
        return ""
    runtimes = llm.get("runtimes")
    if not isinstance(runtimes, dict):
        return ""
    runtime_id = f"{provider.strip().lower().replace('-', '_')}_main"
    runtime = runtimes.get(runtime_id)
    return str(runtime.get("api_key") or "") if isinstance(runtime, dict) else ""


def _restore_optional_file(path: Path, payload: bytes | None, mode: int) -> None:
    if payload is None:
        if path.exists():
            path.unlink()
        return
    _atomic_write_bytes(path, payload, mode)


def _remove_sqlite_database(path: Path) -> None:
    for candidate in (
        path,
        path.with_name(f"{path.name}-wal"),
        path.with_name(f"{path.name}-shm"),
    ):
        if candidate.exists():
            candidate.unlink()


def _local_opencode_key(*, required: bool) -> str | None:
    """只读导入 OpenCode Go 本机登录，不修改其 canonical auth store。"""
    path = Path.home() / ".local" / "share" / "opencode" / "auth.json"
    if not path.exists():
        if required:
            raise AuthenticationError("未找到本机 OpenCode Go 登录")
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AuthenticationError("OpenCode auth.json 无法读取") from exc
    entry = document.get("opencode-go") if isinstance(document, dict) else None
    key = str(entry.get("key") or "") if isinstance(entry, dict) else ""
    if not key:
        if required:
            raise AuthenticationError("OpenCode Go 登录缺少 API key")
        return None
    return key


def _new_config(workspace: Path) -> str:
    """创建首次启动所需的最小主模型、控制面与 Web Chat 配置。"""
    document = tomlkit.document()
    runtime = tomlkit.table()
    runtime["workspace"] = str(workspace)
    document["runtime"] = runtime
    document["llm"] = tomlkit.table()
    document["agent"] = tomlkit.table()
    channels = tomlkit.table()
    chat = tomlkit.table()
    chat["enabled"] = True
    chat["channel_name"] = "web"
    channels["chat"] = chat
    document["channels"] = channels
    app_server = tomlkit.table()
    app_server["enabled"] = True
    document["app_server"] = app_server
    return tomlkit.dumps(document)


def _atomic_write(path: Path, content: str, mode: int) -> None:
    _atomic_write_bytes(path, content.encode("utf-8"), mode)


def _atomic_write_bytes(path: Path, content: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.settings-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
