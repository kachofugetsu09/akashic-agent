from agent.policies.delegation import DelegationPolicy


def test_delegation_policy_marks_background_long_running_tasks():
    decision = DelegationPolicy().decide(
        task="请在后台遍历整个仓库并整理所有 RSS feed 的更新情况"
    )

    assert decision.should_spawn is True
    assert decision.meta.reason_code == "tool_chain_heavy"
    assert decision.meta.confidence == "high"
    assert decision.meta.source == "llm"


def test_delegation_policy_marks_inline_small_tasks():
    decision = DelegationPolicy().decide(task="帮我看下这个函数名是不是合适")

    assert decision.should_spawn is True
    assert decision.meta.reason_code == "tool_chain_heavy"
    assert decision.meta.source == "llm"


def test_delegation_policy_marks_tool_chain_heavy_tasks():
    decision = DelegationPolicy().decide(
        task="先 web search 搜索资料，再 fetch 网页并读取文件整理结论"
    )

    assert decision.should_spawn is True
    assert decision.meta.reason_code == "tool_chain_heavy"
