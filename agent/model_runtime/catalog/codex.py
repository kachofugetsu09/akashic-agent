from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from agent.model_runtime.auth.codex import (
    CODEX_API_BASE,
    CODEX_CLIENT_VERSION,
    CodexAuthDriver,
)
from agent.model_runtime.errors import AuthenticationError, TransportError
from agent.model_runtime.context_policy import recommended_context_settings
from agent.model_runtime.types import ModelCapabilities


@dataclass(frozen=True)
class CodexModel:
    slug: str
    capabilities: ModelCapabilities
    input_modalities_known: bool


class CodexModelCatalog:
    """从 Codex `/models` 边界加载并校验模型元数据。"""

    def __init__(
        self,
        auth: CodexAuthDriver,
        *,
        base_url: str = CODEX_API_BASE,
        client_version: str = CODEX_CLIENT_VERSION,
    ) -> None:
        self.auth = auth
        self.base_url = base_url.rstrip("/")
        self.client_version = client_version

    async def list_models(self) -> list[CodexModel]:
        headers = self.auth.headers()
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.base_url}/models",
                params={"client_version": self.client_version},
                headers=headers,
            )
        if response.status_code == 401:
            headers = self.auth.headers(force_refresh=True)
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    params={"client_version": self.client_version},
                    headers=headers,
                )
        if response.status_code in {401, 403}:
            raise AuthenticationError("Codex 模型目录认证失败，请重新登录")
        if response.status_code >= 400:
            raise TransportError(f"Codex 模型目录请求失败 (HTTP {response.status_code})")
        payload = response.json()
        raw_models = payload.get("models")
        if not isinstance(raw_models, list):
            raise TransportError("Codex 模型目录响应缺少 models 数组")
        parsed_models = [self._parse_model(item) for item in raw_models]
        return [
            model
            for raw, model in zip(raw_models, parsed_models, strict=True)
            if raw.get("visibility", "list") == "list"
            and raw.get("supported_in_api", True) is not False
        ]

    @staticmethod
    def _parse_model(raw: Any) -> CodexModel:
        if not isinstance(raw, dict) or not isinstance(raw.get("slug"), str):
            raise TransportError("Codex 模型目录包含无效模型项")
        context_window = raw.get("context_window") or raw.get("max_context_window")
        if not isinstance(context_window, int) or context_window <= 0:
            raise TransportError(f"模型 {raw['slug']} 缺少有效 context_window")
        max_context_window = raw.get("max_context_window") or context_window
        if (
            not isinstance(max_context_window, int)
            or max_context_window < context_window
        ):
            raise TransportError(f"模型 {raw['slug']} 的 max_context_window 无效")
        modalities = raw.get("input_modalities")
        modalities_known = isinstance(modalities, list)
        parsed_modalities = (
            tuple(str(item) for item in modalities)
            if isinstance(modalities, list)
            else ("text",)
        )
        efforts = raw.get("supported_reasoning_levels") or []
        effort_names = tuple(
            str(item.get("effort"))
            for item in efforts
            if isinstance(item, dict) and item.get("effort")
        )
        percent = int(raw.get("effective_context_window_percent", 90)) / 100
        capabilities = ModelCapabilities(
            context_window=context_window,
            max_output_tokens=recommended_context_settings(
                max_context_window, percent
            ).output_reserve,
            max_context_window=max_context_window,
            effective_context_percent=percent,
            supported_reasoning_efforts=effort_names,
            default_reasoning_effort=raw.get("default_reasoning_level"),
            input_modalities=parsed_modalities,
            supports_parallel_tool_calls=bool(raw.get("supports_parallel_tool_calls")),
            supports_reasoning_summaries=bool(
                raw.get(
                    "supports_reasoning_summary_parameter",
                    raw.get("supports_reasoning_summaries", False),
                )
            ),
            use_responses_lite=bool(raw.get("use_responses_lite")),
        )
        return CodexModel(
            slug=raw["slug"],
            capabilities=capabilities,
            input_modalities_known=modalities_known,
        )
