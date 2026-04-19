from dataclasses import dataclass, field


@dataclass
class MemoryItem:
    id: str
    memory_type: str  # procedure / preference / event / profile
    summary: str
    content_hash: str  # SHA256[:16]
    embedding: list[float] | None
    reinforcement: int
    extra_json: dict  # 类型专用字段
    source_ref: str | None
    happened_at: str | None
    created_at: str
    updated_at: str
    emotional_weight: int = 0
