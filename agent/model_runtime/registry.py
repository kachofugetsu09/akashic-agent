from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent.model_runtime.auth import CodexAuthDriver, CredentialStore
from agent.model_runtime.runtime import ChatRuntimeAdapter, ModelRuntime, ResponsesRuntime
from agent.model_runtime.transports import CodexResponsesTransport


@dataclass(frozen=True)
class RuntimeAssembly:
    auth_driver: str
    catalog: str
    transport: str
    profile: str


_BUILTIN_ASSEMBLIES = {
    "codex": RuntimeAssembly("codex", "codex", "responses", "codex"),
    "openai": RuntimeAssembly("api_key", "manual", "chat_completions", "openai_compatible"),
    "deepseek": RuntimeAssembly("api_key", "manual", "chat_completions", "deepseek"),
    "qwen": RuntimeAssembly("api_key", "manual", "chat_completions", "dashscope"),
}
SUPPORTED_PROVIDER_PRESETS = frozenset(_BUILTIN_ASSEMBLIES)


class ModelRuntimeRegistry:
    """按 provider preset 组装认证、目录、profile 与 transport。"""

    def __init__(self, credential_store: CredentialStore | None = None) -> None:
        self.credential_store = credential_store or CredentialStore()

    def assembly(self, provider: str) -> RuntimeAssembly:
        if not provider:
            return RuntimeAssembly(
                "api_key", "manual", "chat_completions", "legacy_auto"
            )
        return _BUILTIN_ASSEMBLIES.get(
            provider.lower(),
            _BUILTIN_ASSEMBLIES["openai"],
        )

    def build_codex_transport(
        self,
        *,
        auth_id: str,
        runtime_id: str,
        base_url: str,
        connect_timeout_s: float,
        read_timeout_s: float,
        write_timeout_s: float,
        pool_timeout_s: float,
    ) -> CodexResponsesTransport:
        assembly = self.assembly("codex")
        if assembly.transport != "responses":
            raise RuntimeError("Codex preset transport 不变量被破坏")
        auth = CodexAuthDriver(self.credential_store, auth_id)
        return CodexResponsesTransport(
            auth,
            runtime_id=runtime_id,
            base_url=base_url,
            connect_timeout_s=connect_timeout_s,
            read_timeout_s=read_timeout_s,
            write_timeout_s=write_timeout_s,
            pool_timeout_s=pool_timeout_s,
        )

    def build_runtime(
        self,
        *,
        provider: str,
        auth_id: str,
        runtime_id: str,
        base_url: str,
        connect_timeout_s: float,
        read_timeout_s: float,
        write_timeout_s: float,
        pool_timeout_s: float,
        chat_factory: Callable[[str], Any],
    ) -> ModelRuntime:
        """按 preset 选择 transport/profile，并返回统一 runtime。"""
        assembly = self.assembly(provider)
        if assembly.transport == "responses":
            transport = self.build_codex_transport(
                auth_id=auth_id,
                runtime_id=runtime_id,
                base_url=base_url or "https://chatgpt.com/backend-api/codex",
                connect_timeout_s=connect_timeout_s,
                read_timeout_s=read_timeout_s,
                write_timeout_s=write_timeout_s,
                pool_timeout_s=pool_timeout_s,
            )
            return ResponsesRuntime(transport)
        if assembly.transport == "chat_completions":
            return ChatRuntimeAdapter(chat_factory(assembly.profile))
        raise ValueError(f"不支持的模型 transport: {assembly.transport}")


__all__ = ["ModelRuntimeRegistry", "RuntimeAssembly", "SUPPORTED_PROVIDER_PRESETS"]
