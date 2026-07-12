"""AgentCard 解析器：从 peer agent 的 /.well-known/agent.json 获取元数据。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import httpx

from core.net.http import HttpRequester, RequestBudget


class AgentCardUnavailableError(RuntimeError):
    """AgentCard endpoint 因网络或 HTTP 状态不可访问。"""


class AgentCardSchemaError(ValueError):
    """AgentCard 响应不符合实际消费 schema。"""


@dataclass
class AgentSkill:
    id: str
    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


@dataclass
class AgentCard:
    name: str
    url: str
    description: str = ""
    skills: list[AgentSkill] = field(default_factory=list)

    @property
    def primary_skill(self) -> AgentSkill | None:
        return self.skills[0] if self.skills else None


def _object(value: Any, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AgentCardSchemaError(
            f"AgentCard {path} 必须是对象，实际为 {type(value).__name__}"
        )
    return cast(dict[str, object], value)


def _required_text(value: dict[str, object], key: str, path: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise AgentCardSchemaError(f"AgentCard {path}.{key} 必须是非空字符串")
    return raw


def _optional_text(value: dict[str, object], key: str, path: str) -> str:
    raw = value.get(key, "")
    if not isinstance(raw, str):
        raise AgentCardSchemaError(f"AgentCard {path}.{key} 必须是字符串")
    return raw


def _string_list(value: dict[str, object], key: str, path: str) -> list[str]:
    raw = value.get(key, [])
    if not isinstance(raw, list):
        raise AgentCardSchemaError(f"AgentCard {path}.{key} 必须是字符串数组")
    result: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            raise AgentCardSchemaError(
                f"AgentCard {path}.{key}[{index}] 必须是非空字符串"
            )
        result.append(item)
    return result


def _parse_skill(value: Any, index: int) -> AgentSkill:
    path = f"skills[{index}]"
    skill = _object(value, path)
    return AgentSkill(
        id=_required_text(skill, "id", path),
        name=_required_text(skill, "name", path),
        description=_required_text(skill, "description", path),
        tags=_string_list(skill, "tags", path),
        examples=_string_list(skill, "examples", path),
    )


def _parse_card(value: Any) -> AgentCard:
    """校验 AgentCard 外部边界并构造类型化 card。"""

    # 1. 校验根节点和工具实际使用的必需字段
    data = _object(value, "root")
    name = _required_text(data, "name", "root")
    url = _required_text(data, "url", "root")
    description = _optional_text(data, "description", "root")

    # 2. 校验技能数组，保留无技能但结构完整的合法 card
    raw_skills = data.get("skills", [])
    if not isinstance(raw_skills, list):
        raise AgentCardSchemaError("AgentCard root.skills 必须是数组")
    skills = [_parse_skill(skill, index) for index, skill in enumerate(raw_skills)]
    return AgentCard(name=name, url=url, description=description, skills=skills)


async def fetch_agent_card(base_url: str, requester: HttpRequester) -> AgentCard:
    """GET AgentCard，区分网络不可达与响应 schema 错误。"""

    url = base_url.rstrip("/") + "/.well-known/agent.json"

    # 1. 只把明确的网络、传输、超时和 HTTP 状态错误转成离线信号
    try:
        response = await requester.get(url, budget=RequestBudget(total_timeout_s=3.0))
        response.raise_for_status()
    except httpx.UnsupportedProtocol:
        raise
    except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
        raise AgentCardUnavailableError(
            f"无法访问 AgentCard endpoint {url}: {exc}"
        ) from exc

    # 2. JSON 解码和 schema 错误原样暴露给 Registry，不能伪装成离线
    return _parse_card(response.json())
