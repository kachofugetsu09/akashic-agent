from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

@dataclass
class ScenarioMemorySeed:
    long_term: str = ""
    self_profile: str = ""
    now: str = ""


@dataclass
class ScenarioMemoryItem:
    summary: str
    memory_type: str
    extra: dict = field(default_factory=dict)
    source_ref: str = ""
    happened_at: str = ""


@dataclass
class ScenarioAssertions:
    route_decision: str | None = None
    min_history_hits: int | None = None
    max_history_hits: int | None = None
    required_tools: list[str] = field(default_factory=list)
    final_contains: list[str] = field(default_factory=list)


@dataclass
class ScenarioJudgeSpec:
    goal: str
    rubric: list[str] = field(default_factory=list)


@dataclass
class ScenarioSpec:
    id: str
    message: str
    channel: str
    chat_id: str
    session_key: str
    request_time: datetime
    history: list[dict] = field(default_factory=list)
    memory: ScenarioMemorySeed = field(default_factory=ScenarioMemorySeed)
    memory2_items: list[ScenarioMemoryItem] = field(default_factory=list)
    assertions: ScenarioAssertions = field(default_factory=ScenarioAssertions)
    judge: ScenarioJudgeSpec | None = None


def build_tool_search_schedule_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        id="tool_search_schedule_real_tools",
        message="帮我十分钟后提醒喝水",
        channel="cli",
        chat_id="scenario-tool-search",
        session_key="cli:scenario-tool-search",
        request_time=datetime.fromisoformat("2026-03-12T10:00:00+08:00"),
        assertions=ScenarioAssertions(
            required_tools=["tool_search", "schedule"],
            final_contains=["提醒"],
        ),
    )


def build_smalltalk_no_retrieve_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        id="smalltalk_no_retrieve_real",
        message="今天天气不错，我刚泡了杯茶，感觉还行。",
        channel="cli",
        chat_id="scenario-smalltalk",
        session_key="cli:scenario-smalltalk",
        request_time=datetime.fromisoformat("2026-03-12T10:05:00+08:00"),
        history=[
            {
                "role": "user",
                "content": "我昨晚有点累，不过今天已经好多了。",
                "timestamp": "2026-03-01T12:00:00+08:00",
            },
            {
                "role": "assistant",
                "content": "那就好，今天可以轻松一点。",
                "timestamp": "2026-03-01T12:00:10+08:00",
            },
        ],
        memory=ScenarioMemorySeed(
            long_term="用户长期偏好：喜欢轻松聊天，不喜欢太正式的回复。",
        ),
        memory2_items=[
            ScenarioMemoryItem(
                summary="用户偏好轻松聊天风格，不喜欢太正式的回复。",
                memory_type="preference",
                extra={"scope_channel": "cli", "scope_chat_id": "scenario-smalltalk"},
                source_ref="scenario-smalltalk-pref",
                happened_at="2026-03-01T12:00:00+08:00",
            )
        ],
        assertions=ScenarioAssertions(
            route_decision="NO_RETRIEVE",
            max_history_hits=0,
        ),
    )


def build_sample_scenarios(root: Path | None = None) -> list[ScenarioSpec]:
    _ = root
    return [
        build_tool_search_schedule_scenario(),
        build_smalltalk_no_retrieve_scenario(),
    ]
