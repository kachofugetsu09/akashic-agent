from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import cast

import httpx

from agent.model_runtime.errors import AuthenticationError, TransportError
from agent.model_runtime.catalog.litellm_registry import resolve_catalog_capabilities
from agent.model_runtime.provider_profiles import (
    OPENCODE_GO_BASE_URL,
    OPENCODE_GO_PROFILE,
)


@dataclass(frozen=True)
class OpenCodeGoModel:
    slug: str
    supported_reasoning_efforts: tuple[str, ...] = ()


_OPENCODE_PROVIDER_PREFIX = "opencode-go/"


def _parse_opencode_go_reasoning_efforts(
    output: str,
) -> dict[str, tuple[str, ...]]:
    """从 OpenCode verbose 模型目录提取每个模型的真实 variant 名称。"""
    decoder = json.JSONDecoder()
    cursor = 0
    efforts: dict[str, tuple[str, ...]] = {}

    # 1. 每个记录由 provider/model 标题行和紧随其后的 JSON 对象组成。
    while cursor < len(output):
        while cursor < len(output) and output[cursor].isspace():
            cursor += 1
        if cursor >= len(output):
            break
        line_end = output.find("\n", cursor)
        if line_end == -1:
            raise TransportError("OpenCode 模型目录包含不完整的标题行")
        header = output[cursor:line_end].strip()
        if not header.startswith(_OPENCODE_PROVIDER_PREFIX):
            raise TransportError("OpenCode 模型目录包含未知记录")
        model_id = header.removeprefix(_OPENCODE_PROVIDER_PREFIX).strip()
        if not model_id:
            raise TransportError("OpenCode 模型目录包含空模型 ID")

        # 2. JSON 由 OpenCode 自己生成；结构异常必须暴露，不能回退到模型名猜测。
        json_start = line_end + 1
        while json_start < len(output) and output[json_start].isspace():
            json_start += 1
        try:
            decoded, cursor = decoder.raw_decode(output, json_start)
        except json.JSONDecodeError as exc:
            raise TransportError("OpenCode 模型目录返回了无效 JSON") from exc
        if not isinstance(decoded, dict):
            raise TransportError(f"OpenCode 模型 {model_id} 的元数据无效")
        metadata = cast(dict[str, object], decoded)
        variants = metadata.get("variants")
        if not isinstance(variants, dict):
            raise TransportError(f"OpenCode 模型 {model_id} 的 variants 无效")
        variant_map = cast(dict[object, object], variants)
        variant_names: list[str] = []
        for name in variant_map:
            if not isinstance(name, str) or not name:
                raise TransportError(f"OpenCode 模型 {model_id} 的 variants 无效")
            variant_names.append(name)
        efforts[model_id] = tuple(variant_names)
    return efforts


async def _load_opencode_go_reasoning_efforts(
    executable: str,
) -> dict[str, tuple[str, ...]]:
    """调用 OpenCode 的公开模型命令并返回 Go 模型的 variant 目录。"""
    # 1. OpenCode 是 variant 规则的 owner；直接读取其解析后的目录。
    try:
        process = await asyncio.create_subprocess_exec(
            executable,
            "models",
            "opencode-go",
            "--verbose",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise TransportError(f"无法执行 OpenCode 模型探测：{exc}") from exc

    # 2. 限制探测时间，并保留非零退出的明确失败语义。
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
    except TimeoutError as exc:
        _ = process.kill()
        _ = await process.wait()
        raise TransportError("OpenCode 模型探测超时") from exc
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        suffix = f"：{detail[:500]}" if detail else ""
        raise TransportError(f"OpenCode 模型探测失败{suffix}")
    try:
        output = stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise TransportError("OpenCode 模型目录不是有效 UTF-8") from exc
    return _parse_opencode_go_reasoning_efforts(output)


class OpenCodeGoModelCatalog:
    """从 OpenCode Go `/models` 加载可走 Chat Completions 的模型。"""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = OPENCODE_GO_BASE_URL,
        opencode_executable: str = "opencode",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.opencode_executable = opencode_executable

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
            decoded_payload: object = response.json()
        except json.JSONDecodeError as exc:
            raise TransportError("OpenCode Go 模型目录返回了无效 JSON") from exc
        if not isinstance(decoded_payload, dict):
            raise TransportError("OpenCode Go 模型目录响应不是对象")
        payload = cast(dict[str, object], decoded_payload)
        raw_models = payload.get("data")
        if not isinstance(raw_models, list):
            raise TransportError("OpenCode Go 模型目录响应缺少 data 数组")
        model_items = cast(list[object], raw_models)

        # 3. `/models` 决定当前账号可用模型，OpenCode 决定各模型真实 variant。
        try:
            reasoning_efforts = await _load_opencode_go_reasoning_efforts(
                self.opencode_executable
            )
        except TransportError as exc:
            logging.getLogger(__name__).warning(
                "OpenCode variant 目录不可用，思考强度改用本地模型注册表: %s",
                exc,
            )
            reasoning_efforts = {}
        models: list[OpenCodeGoModel] = []
        for raw in model_items:
            if not isinstance(raw, dict):
                raise TransportError("OpenCode Go 模型目录包含无效模型项")
            model_entry = cast(dict[str, object], raw)
            model_id = model_entry.get("id")
            if not isinstance(model_id, str) or not model_id.strip():
                raise TransportError("OpenCode Go 模型目录包含无效模型项")
            if OPENCODE_GO_PROFILE.classify_model(model_id) == "chat_completions":
                models.append(
                    OpenCodeGoModel(
                        slug=model_id,
                        supported_reasoning_efforts=(
                            reasoning_efforts[model_id]
                            if model_id in reasoning_efforts
                            else _registry_reasoning_efforts(
                                model_id,
                                base_url=self.base_url,
                            )
                        ),
                    )
                )
        return models


def _registry_reasoning_efforts(
    model_id: str,
    *,
    base_url: str,
) -> tuple[str, ...]:
    capabilities = resolve_catalog_capabilities(
        "opencode-go",
        model_id,
        base_url=base_url,
    )
    return capabilities.supported_reasoning_efforts if capabilities else ()
