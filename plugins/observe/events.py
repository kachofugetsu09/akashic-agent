from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RagHitLog:
    """一次检索中命中的单条记忆条目。"""

    item_id: str
    memory_type: str
    score: float
    summary: str              # 截断 120 字符
    injected: bool            # 是否最终注入到 context
    confidence_label: str = ""  # "有印象，不确定" 等，空串表示正常置信度
    forced: bool = False        # True = 因 tool_requirement 强制注入，非 score 过阈值


@dataclass
class RagQueryLog:
    """一次 memory 检索事件：query → hits → injected。"""

    caller: str                         # "passive" | "proactive" | "explicit"
    session_key: str
    query: str                          # 实际检索用的 query（rewrite 之后）
    orig_query: str | None              # 改写前原文，None = 未改写
    aux_queries: list[str]              # HyDE 生成的假想条目列表
    hits: list[RagHitLog]
    injected_count: int
    route_decision: str | None = None   # "RETRIEVE" | "NO_RETRIEVE"；None = 无 gate
    error: str | None = None


@dataclass
class TurnTrace:
    """一轮 agent 对话的完整记录。"""

    source: Literal["agent"]
    session_key: str
    user_msg: str | None            # 用户原文
    llm_output: str                 # LLM 最终输出完整文本
    raw_llm_output: str | None = None       # 装饰/清洗前的原始模型输出
    meme_tag: str | None = None             # 命中的 <meme:tag>
    meme_media_count: int | None = None     # 命中的媒体数量
    tool_calls: list[dict] = field(default_factory=list)
    # 每个 tool call: {name, args, result}（args/result 会截断）
    error: str | None = None
    tool_chain_json: str | None = None  # JSON: [{text, calls:[{name,args,result}]}] 每轮迭代完整记录
    history_window: int | None = None
    history_messages: int | None = None
    history_chars: int | None = None
    history_tokens: int | None = None
    prompt_tokens: int | None = None
    next_turn_baseline_tokens: int | None = None
    react_iteration_count: int | None = None
    react_input_sum_tokens: int | None = None
    react_input_peak_tokens: int | None = None
    react_final_input_tokens: int | None = None
    react_cache_prompt_tokens: int | None = None
    react_cache_hit_tokens: int | None = None


@dataclass
class MemoryWriteTrace:
    """PostResponseMemoryWorker 写入/supersede 的一条记忆记录。"""

    session_key: str
    source_ref: str
    action: str          # 'write' | 'supersede'
    memory_type: str | None = None   # write: 写入类型; supersede: None
    item_id: str | None = None       # write: 新条目 id (格式 'new:xxx' or 'reinforced:xxx')
    summary: str | None = None       # write: 写入的 summary
    superseded_ids: list[str] = field(default_factory=list)  # supersede: 被退休的 id 列表
    error: str | None = None
