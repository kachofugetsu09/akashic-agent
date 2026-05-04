from memory2.query_builder import build_procedure_queries


def test_original_message_always_in_queries():
    """原始消息必须出现在 query 列表中。"""
    msg = "帮我搜一下代码里的关键字"
    queries = build_procedure_queries(msg)
    assert msg in queries


def test_no_hardcoded_caozuoguifan_suffix():
    """不能出现硬编码的 '操作规范' 后缀拼接。

    当前 handlers.py:104 的做法：f"{msg} 操作规范"
    这会让 embedding 向量向"规范/制度"语义偏移。
    """
    msg = "帮我搜一下代码里的关键字"
    queries = build_procedure_queries(msg)
    assert f"{msg} 操作规范" not in queries


def test_returns_at_least_two_queries():
    """保留原始 query，并追加 LLM 产出的改写 query。"""
    queries = build_procedure_queries("把这个B站视频下载下来", "B站视频 下载 SOP")
    assert queries == ["把这个B站视频下载下来", "B站视频 下载 SOP"]


def test_all_queries_are_non_empty_strings():
    queries = build_procedure_queries("给我生成一个RSS")
    assert all(isinstance(q, str) and q.strip() for q in queries)


def test_no_duplicate_queries():
    """返回值里不能有重复的 query。"""
    queries = build_procedure_queries("hello")
    assert len(queries) == len(set(queries))


def test_short_message_still_works():
    """极短消息不应崩溃，且仍然返回至少 1 个有效 query。"""
    queries = build_procedure_queries("下载")
    assert len(queries) >= 1
    assert queries[0].strip()


def test_generic_message_with_no_keywords_returns_original_only():
    """对于没有命中领域语境的普通消息，只返回原始消息本身。"""
    queries = build_procedure_queries("帮我创建一个新技能")
    assert queries == ["帮我创建一个新技能"]


def test_no_rewritten_query_returns_original_only():
    """没有 LLM 改写时，不额外扩展 query。"""
    msg = "把这个视频发给我"
    queries = build_procedure_queries(msg)
    assert queries == [msg]
