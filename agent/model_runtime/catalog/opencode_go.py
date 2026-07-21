from __future__ import annotations

import json
from dataclasses import dataclass

import httpx

from agent.model_runtime.errors import AuthenticationError, TransportError
from agent.model_runtime.provider_profiles import (
    OPENCODE_GO_BASE_URL,
    OPENCODE_GO_PROFILE,
)


@dataclass(frozen=True)
class OpenCodeGoModel:
    slug: str


class OpenCodeGoModelCatalog:
    """从 OpenCode Go `/models` 加载可走 Chat Completions 的模型。"""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = OPENCODE_GO_BASE_URL,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    async def list_models(self) -> list[OpenCodeGoModel]:
        # 1. 在外部 HTTP 边界读取目录，网络错误统一转换为 transport error。
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
        except httpx.HTTPError as exc:
            raise TransportError(f"OpenCode Go 模型目录请求失败：{exc}") from exc

        if response.status_code in {401, 403}:
            raise AuthenticationError("OpenCode Go 模型目录认证失败，请检查 API key")
        if response.status_code >= 400:
            raise TransportError(
                f"OpenCode Go 模型目录请求失败 (HTTP {response.status_code})"
            )

        # 2. 校验 OpenAI 模型目录结构，并只暴露已知 Chat 模型家族。
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise TransportError("OpenCode Go 模型目录返回了无效 JSON") from exc
        raw_models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(raw_models, list):
            raise TransportError("OpenCode Go 模型目录响应缺少 data 数组")

        models: list[OpenCodeGoModel] = []
        for raw in raw_models:
            model_id = raw.get("id") if isinstance(raw, dict) else None
            if not isinstance(model_id, str) or not model_id.strip():
                raise TransportError("OpenCode Go 模型目录包含无效模型项")
            if OPENCODE_GO_PROFILE.classify_model(model_id) == "chat_completions":
                models.append(OpenCodeGoModel(slug=model_id))
        return models
