"""
AgentCard 解析器：从 peer agent 的 /.well-known/agent.json 获取元数据。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.net.http import HttpRequester, RequestBudget

logger = logging.getLogger(__name__)


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


async def fetch_agent_card(base_url: str, requester: HttpRequester) -> AgentCard:
    """GET {base_url}/.well-known/agent.json 并解析成 AgentCard。"""
    url = base_url.rstrip("/") + "/.well-known/agent.json"
    try:
        r = await requester.get(url, budget=RequestBudget(total_timeout_s=3.0))
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise RuntimeError(f"无法获取 AgentCard from {url}: {e}") from e

    skills = [
        AgentSkill(
            id=s.get("id", ""),
            name=s.get("name", ""),
            description=s.get("description", ""),
            tags=s.get("tags", []),
            examples=s.get("examples", []),
        )
        for s in data.get("skills", [])
    ]
    return AgentCard(
        name=data["name"],
        url=data.get("url", base_url),
        description=data.get("description", ""),
        skills=skills,
    )
