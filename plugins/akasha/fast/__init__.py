"""akasha.fast —— 离线重建加速后端（纯加性，靠 install() 注入，不改 core/replay/store/engine）。

  * mem_store.MemoryStore / CapturingMemoryStore —— 纯内存图 + 增量 in_strength/edges_by_src/fan 视图
  * graph_fast.install(store)  —— monkeypatch graph_expand(O(E)→增量) + edges_by_src/fan 视图 + has_user_turn 缓存
  * fast_dense.install()       —— 向量化 dense_message_candidates
  * dump.dump_to_db(path, store) —— 末尾一次性批量落库（复用 embedding_cache）
"""
from plugins.akasha.fast import mem_store, graph_fast, fast_dense, dump  # noqa: F401
