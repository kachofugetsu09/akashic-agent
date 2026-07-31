from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
import tomllib
from pathlib import Path
from typing import Callable, Literal
from uuid import uuid4

import tomlkit
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.config import Config
from agent.model_runtime.auth.codex import CodexAuthDriver
from agent.model_runtime.auth.store import CredentialStore
from agent.model_runtime.catalog.codex import CodexModelCatalog
from agent.model_runtime.catalog.opencode_go import OpenCodeGoModelCatalog
from agent.model_runtime.errors import AuthenticationError, ModelRuntimeError, TransportError
from agent.provider import LLMProvider
from bootstrap.setup_main import patch_main_model_config
from bootstrap.setup_wizard import WizardAnswers


class ModelQuery(BaseModel):
    provider: str = Field(min_length=1, max_length=64)
    api_key: str = ""
    credential_id: str = ""
    use_local_opencode: bool = False
    base_url: str = ""


class ApplyPayload(BaseModel):
    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=200)
    api_key: str = ""
    credential_id: str = ""
    use_local_opencode: bool = False
    base_url: str = Field(default="", max_length=2048)
    context_window: int = Field(gt=0)
    max_output_tokens: int = Field(ge=0)
    input_modalities: list[Literal["text", "image"]] = Field(
        default_factory=lambda: ["text"]
    )
    reasoning_effort: str = Field(default="", max_length=32)


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
    store = credential_store or CredentialStore()
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
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
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
                        }
                        for entry in entries
                    ]
                }
            if payload.provider == "opencode-go":
                key = _candidate_api_key(
                    payload.api_key,
                    config_path,
                    payload.provider,
                    use_local_opencode=payload.use_local_opencode,
                )
                entries = await OpenCodeGoModelCatalog(
                    key,
                    base_url=payload.base_url or "https://opencode.ai/zen/go/v1",
                ).list_models()
                return {"models": [{"id": entry.slug} for entry in entries]}
            return {"models": []}
        except (AuthenticationError, TransportError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/settings/apply")
    async def apply(payload: ApplyPayload) -> dict[str, object]:
        if (
            payload.max_output_tokens > 0
            and payload.max_output_tokens >= payload.context_window
        ):
            raise HTTPException(status_code=422, detail="最大输出必须小于上下文窗口")
        if "text" not in payload.input_modalities:
            raise HTTPException(status_code=422, detail="输入模态必须包含 text")
        if not apply_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="已有设置操作正在执行")
        try:
            operation_id = f"settings-{uuid4().hex}"
            answers = _answers(payload, config_path, store)
            await _validate_live_candidate(answers, store)
            _apply_candidate(
                config_path,
                workspace,
                answers,
                operation_id,
                on_applied=on_applied,
            )
            return {"operationId": operation_id, "status": "applied"}
        finally:
            apply_lock.release()

    @app.post("/api/settings/codex-login")
    async def begin_codex_login() -> dict[str, object]:
        login_id = f"codex-{uuid4().hex}"
        try:
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

    @app.get("/")
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
    credential_meta = store.metadata()
    local_opencode = _local_opencode_key(required=False) is not None
    if not config_path.exists():
        return {
            "mode": "needs_setup",
            "workspace": str(workspace),
            "activeRuntime": None,
            "runtimes": [],
            "codexConfigured": "codex_default" in credential_meta,
            "localOpenCodeConfigured": local_opencode,
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
            "codexConfigured": "codex_default" in credential_meta,
            "localOpenCodeConfigured": local_opencode,
        }
    llm = raw.get("llm") if isinstance(raw.get("llm"), dict) else {}
    active = llm.get("main")
    try:
        if isinstance(active, str):
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
            "codexConfigured": "codex_default" in credential_meta,
            "localOpenCodeConfigured": local_opencode,
        }
    return {
        "mode": mode,
        "workspace": str(workspace),
        "activeRuntime": active,
        "runtimes": runtimes,
        "codexConfigured": "codex_default" in credential_meta,
        "localOpenCodeConfigured": local_opencode,
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
        "baseUrl": str(raw.get("base_url") or ""),
        "contextWindow": context_window,
        "maxOutputTokens": max_output_tokens,
        "inputModalities": input_modalities,
        "credential": {
            "id": auth,
            "configured": bool(auth and auth in credential_meta) or bool(inline),
            "source": source,
        },
    }


def _candidate_api_key(
    value: str,
    config_path: Path,
    provider: str,
    *,
    use_local_opencode: bool,
) -> str:
    if value:
        return value
    if use_local_opencode:
        key = _local_opencode_key(required=True)
        assert key is not None
        return key
    saved = _saved_api_key(config_path, provider)
    if saved:
        return saved
    raise AuthenticationError("API Key 不能为空")


def _answers(
    payload: ApplyPayload,
    config_path: Path,
    store: CredentialStore,
) -> WizardAnswers:
    provider = payload.provider.strip().lower()
    auth_id = payload.credential_id.strip()
    if provider == "codex":
        auth_id = auth_id or "codex_default"
        _ = store.get(auth_id)
        api_key = ""
    else:
        auth_id = ""
        api_key = _candidate_api_key(
            payload.api_key,
            config_path,
            provider,
            use_local_opencode=payload.use_local_opencode,
        )
    return WizardAnswers(
        provider=provider,
        auth_id=auth_id,
        api_key=api_key,
        model=payload.model.strip(),
        base_url=payload.base_url.strip(),
        context_window=payload.context_window,
        max_output_tokens=payload.max_output_tokens,
        multimodal="image" in payload.input_modalities,
        reasoning_effort=payload.reasoning_effort.strip(),
    )


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
    on_applied: Callable[[], None] | None,
) -> None:
    """备份配置并原子应用候选，完整加载失败时恢复原文件。"""
    original = (
        config_path.read_text(encoding="utf-8")
        if config_path.exists()
        else _new_config(workspace)
    )
    candidate = patch_main_model_config(original, answers)
    config_snapshot = config_path.read_bytes() if config_path.exists() else None
    backup_path = config_path.with_name(f"{config_path.name}.{operation_id}.bak")
    if config_snapshot is not None:
        backup_path.write_bytes(config_snapshot)
        os.chmod(backup_path, 0o600)

    restart_attempted = False
    try:
        _atomic_write(config_path, candidate, 0o600)
        _ = Config.load(config_path, workspace=workspace)
        if on_applied is not None:
            restart_attempted = True
            on_applied()
    except BaseException:
        if config_snapshot is None:
            if config_path.exists():
                config_path.unlink()
        else:
            _atomic_write_bytes(config_path, config_snapshot, 0o600)
        if on_applied is not None and restart_attempted:
            on_applied()
        raise


def _saved_api_key(config_path: Path, provider: str) -> str:
    """从新 schema 的目标 runtime 读取已保存 key，不向 API 响应暴露。"""
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
    chat["host"] = os.environ.get("AKASHIC_CHAT_HOST", "127.0.0.1")
    chat["port"] = 6322
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


def create_settings_server(
    config_path: Path,
    workspace: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 6321,
    on_applied: Callable[[], None] | None = None,
) -> uvicorn.Server:
    config = uvicorn.Config(
        create_settings_app(config_path, workspace, on_applied=on_applied),
        host=host,
        port=port,
        log_level="info",
        access_log=False,
        ws="none",
    )
    return uvicorn.Server(config)
