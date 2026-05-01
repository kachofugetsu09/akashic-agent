from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RagHitLog:
    """一次检索中命中的单条记忆条目。"""

    item_id: str
    memory_type: str
    score: float
    summary: str        # 截断 120 字符
    injected: bool      # 是否最终注入到 context


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
    """一轮 agent 对话或 proactive tick 的完整记录。"""

    source: Literal["agent", "proactive"]
    session_key: str
    user_msg: str | None            # agent: 用户原文; proactive: None
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


@dataclass
class ProactiveDecisionTrace:
    """主动链路的结构化决策 trace，用于解释为什么发/没发。"""

    tick_id: str
    session_key: str
    stage: str
    reason_code: str | None = None
    should_send: bool | None = None
    action: str | None = None
    gate_reason: str | None = None
    pre_score: float | None = None
    base_score: float | None = None
    draw_score: float | None = None
    decision_score: float | None = None
    send_threshold: float | None = None
    interruptibility: float | None = None
    candidate_count: int | None = None
    candidate_item_ids: list[str] = field(default_factory=list)
    sleep_state: str | None = None
    sleep_prob: float | None = None
    sleep_available: bool | None = None
    sleep_data_lag_min: int | None = None
    user_replied_after_last_proactive: bool | None = None
    proactive_sent_24h: int | None = None
    fresh_items_24h: int | None = None
    delivery_key: str | None = None
    is_delivery_duplicate: bool | None = None
    is_message_duplicate: bool | None = None
    delivery_attempted: bool | None = None
    delivery_result: str | None = None
    reasoning_preview: str | None = None
    reasoning: str | None = None        # LLM 完整推理过程
    evidence_item_ids: list[str] = field(default_factory=list)  # LLM 实际引用的条目 IDs
    source_refs_json: str | None = None  # 引用条目元数据 JSON: [{source_name, title, url}]
    fetched_urls: list[str] = field(default_factory=list)  # reflect 阶段 web_fetch 调用的 URL
    stage_result_json: str | None = None
    decision_signals_json: str | None = None
    error: str | None = None
    sent_message: str | None = None     # act 阶段：实际发送的消息正文
    candidates_json: str | None = None  # JSON: [{kind, source_type, source_name, title, content, url, severity?}]
    # Evidence-First Research 字段
    research_status: str | None = None  # success | insufficient | error
    research_rounds_used: int | None = None
    research_tools_called: list[str] = field(default_factory=list)
    research_evidence_count: int | None = None
    research_reason: str | None = None
    fact_claims_count: int | None = None
