from proactive.loop import ProactiveLoop
from proactive.loop_helpers import (
    _build_tfidf_vectors,
    _cosine_sparse,
    _decision_with_randomized_score,
    _format_items,
    _format_recent,
    _item_id,
    _semantic_text,
    _source_key,
)


class ProactiveLoopRuntimeMixin:
    _init_runtime_state = ProactiveLoop._init_runtime_state
    _build_state_store = ProactiveLoop._build_state_store
    _build_fitbit_provider = ProactiveLoop._build_fitbit_provider
    _build_sender = ProactiveLoop._build_sender
    _build_judge = ProactiveLoop._build_judge
    _build_anyaction_gate = ProactiveLoop._build_anyaction_gate
    _build_sense = ProactiveLoop._build_sense
    _build_decide = ProactiveLoop._build_decide
    _build_memory_retriever = ProactiveLoop._build_memory_retriever
    _build_message_deduper = ProactiveLoop._build_message_deduper
    _build_tick = ProactiveLoop._build_tick
    _init_runtime_components = ProactiveLoop._init_runtime_components


__all__ = [
    "ProactiveLoopRuntimeMixin",
    "_build_tfidf_vectors",
    "_cosine_sparse",
    "_decision_with_randomized_score",
    "_format_items",
    "_format_recent",
    "_item_id",
    "_semantic_text",
    "_source_key",
]
