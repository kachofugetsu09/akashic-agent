"""
Embedding 客户端，对接 DashScope text-embedding-v3（OpenAI 兼容接口）
"""

from __future__ import annotations

import asyncio
import logging
import math

from core.net.http import HttpRequester, RequestBudget, get_default_http_requester

logger = logging.getLogger(__name__)


def _parse_embedding_response(
    raw_response: object,
    *,
    expected_count: int,
    expected_dimension: int | None = None,
) -> list[list[float]]:
    """校验 embedding provider 返回的 indexed batch，并按 index 恢复输入顺序。"""
    if not isinstance(raw_response, dict):
        raise ValueError("embedding provider response 必须是 JSON object")
    raw_data = raw_response.get("data")
    if not isinstance(raw_data, list):
        raise ValueError("embedding provider response.data 必须是 JSON array")

    by_index: dict[int, list[float]] = {}
    response_dimension: int | None = None
    for position, raw_item in enumerate(raw_data):
        if not isinstance(raw_item, dict):
            raise ValueError(f"embedding provider data[{position}] 必须是 JSON object")
        raw_index = raw_item.get("index")
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise ValueError(f"embedding provider data[{position}].index 无效")
        if raw_index < 0 or raw_index >= expected_count or raw_index in by_index:
            raise ValueError(
                f"embedding provider data[{position}].index 超出范围或重复: {raw_index}"
            )

        raw_embedding = raw_item.get("embedding")
        if not isinstance(raw_embedding, list) or not raw_embedding:
            raise ValueError(
                f"embedding provider data[{position}].embedding 必须是非空 JSON array"
            )
        embedding: list[float] = []
        for value_position, value in enumerate(raw_embedding):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(
                    f"embedding provider data[{position}].embedding[{value_position}] 无效"
                )
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(
                    f"embedding provider data[{position}].embedding[{value_position}] 不是有限数字"
                )
            embedding.append(numeric)
        dimension = len(embedding)
        if expected_dimension is not None and dimension != expected_dimension:
            raise ValueError(
                f"embedding provider data[{position}].embedding 维度错误: "
                f"expected={expected_dimension} actual={dimension}"
            )
        if response_dimension is None:
            response_dimension = dimension
        elif dimension != response_dimension:
            raise ValueError(
                "embedding provider response.data 向量维度不一致"
            )
        by_index[raw_index] = embedding

    expected_indexes = set(range(expected_count))
    if set(by_index) != expected_indexes:
        raise ValueError(
            "embedding provider response.data 缺少结果或包含多余结果"
        )
    return [by_index[index] for index in range(expected_count)]


class Embedder:
    MAX_BATCH = 10  # DashScope 每批上限
    MAX_TEXT_LEN = 2000

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "text-embedding-v3",
        output_dimensionality: int | None = None,
        requester: HttpRequester | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/embeddings"
        self._key = api_key
        self._model = model
        self._output_dimensionality = output_dimensionality
        self._requester = requester or get_default_http_requester("external_default")

    @property
    def model_id(self) -> str:
        return self._model

    async def embed(self, text: str) -> list[float]:
        """单条 embed"""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """分批 embed，每批 ≤ MAX_BATCH，批间 sleep 0.3s"""
        results: list[list[float]] = []
        truncated = [t[: self.MAX_TEXT_LEN] for t in texts]

        for i in range(0, len(truncated), self.MAX_BATCH):
            batch = truncated[i : i + self.MAX_BATCH]
            payload: dict[str, object] = {"model": self._model, "input": batch}
            if self._output_dimensionality is not None:
                payload["dimensions"] = self._output_dimensionality
            resp = await self._requester.post(
                self._url,
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout_s=30.0,
                budget=RequestBudget(total_timeout_s=40.0),
            )
            resp.raise_for_status()
            results.extend(
                _parse_embedding_response(
                    resp.json(),
                    expected_count=len(batch),
                    expected_dimension=self._output_dimensionality,
                )
            )

            if i + self.MAX_BATCH < len(truncated):
                await asyncio.sleep(0.3)

        return results

    async def aclose(self) -> None:
        return None
