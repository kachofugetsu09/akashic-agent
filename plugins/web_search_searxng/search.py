"""Search a configured SearXNG instance through its JSON API."""
import httpx
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError, field_validator

from core.net.http import http_error_detail
from plugins.standard_web.contract import SearchRequest


class SearxngConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoint: HttpUrl | None = None
    timeout_seconds: float = Field(default=25.0, ge=1, le=120, allow_inf_nan=False)

    @field_validator("endpoint")
    @classmethod
    def check_endpoint(cls, value: HttpUrl | None) -> HttpUrl | None:
        if value is not None and (value.username or value.password or value.query or value.fragment):
            raise ValueError("服务地址不能包含账号、查询参数或片段")
        return value


class SearchResult(BaseModel):
    model_config = ConfigDict(strict=True)

    title: str
    url: str
    content: str = ""


class SearchResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    results: list[SearchResult]
    unresponsive_engines: list[list[str]] = Field(default_factory=list)


class SearxngSearch:
    def __init__(self, endpoint: HttpUrl, timeout_seconds: float):
        self._url = str(endpoint).rstrip("/") + "/search"
        self._timeout = timeout_seconds

    async def search(self, request: SearchRequest) -> dict[str, object]:
        """Keep partial engine failures visible and cap results at the requested count."""
        # 1. SearXNG accepts q and format; it has no result-count parameter.
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(self._url, params={"q": request.query, "format": "json"},
                                            headers={"accept": "application/json"})
                response.raise_for_status()
        except (httpx.RequestError, httpx.HTTPStatusError) as error:
            return {"query": request.query, "error": f"搜索失败：{http_error_detail(error)}"}

        # 2. HTML, malformed JSON, and missing results are failures, not empty searches.
        try:
            data = SearchResponse.model_validate_json(response.content)
        except ValidationError:
            return {"query": request.query, "error": "SearXNG 响应无效；请确认服务地址正确且已启用 JSON 格式"}
        result: dict[str, object] = {
            "query": request.query,
            "results": [item.model_dump() for item in data.results[:request.num_results]],
        }
        if data.unresponsive_engines:
            result["unresponsive_engines"] = data.unresponsive_engines
            if not data.results:
                result["error"] = "搜索引擎发生故障，未获得结果"
        return result
