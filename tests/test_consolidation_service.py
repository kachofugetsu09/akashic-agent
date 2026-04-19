import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from agent.looping.consolidation import (
    ConsolidationService,
    _select_recent_history_entries,
    _replace_recent_turns_block,
)


class _Resp:
    def __init__(self, content: str) -> None:
        self.content = content


def _message_text(kwargs) -> str:
    messages = kwargs.get("messages", [])
    return "\n".join(str(msg.get("content") or "") for msg in messages)


def test_consolidation_service_archive_all_and_profile_extract():
    """consolidation 触发两次并行 LLM 调用：一次提取 event，一次提取 profile/preference/procedure。"""
    memory = SimpleNamespace(
        read_long_term=MagicMock(return_value="MEM"),
        read_history=MagicMock(
            return_value=(
                "[2026-03-15 09:00] 用户确认 Zigbee 需求\n\n"
                "[2026-03-15 09:30] 用户对本地控制方案感兴趣\n\n"
                "[2026-03-15 09:45] 用户准备下单 Zigbee 网关"
            )
        ),
        read_recent_context=MagicMock(return_value=""),
        write_recent_context=MagicMock(),
        append_history_once=MagicMock(return_value=True),
        append_pending_once=MagicMock(return_value=True),
        save_from_consolidation=AsyncMock(),
        save_item=AsyncMock(return_value="new:profile-1"),
        save_item_with_supersede=AsyncMock(return_value="new:profile-1"),
    )
    async def _chat_side_effect(**kwargs):
        text = _message_text(kwargs)
        if "近期语境压缩代理" in text:
            return _Resp(
                '{"active_topics":["用户最近关注 Zigbee 方案"],"user_preferences":[],"follow_ups":["可以继续聊本地控制方案"],"avoidances":[]}'
            )
        if "长期记忆提取专家" in text:
            return _Resp(
                '{"profile":[{"summary":"用户买了 Zigbee 网关","category":"purchase","happened_at":"2026-03-15","emotional_weight":4}],"preference":[],"procedure":[]}'
            )
        return _Resp(
            '{"history_entries":[{"summary":"[2026-03-15 10:00] 用户聊了 Zigbee 方案","emotional_weight":6}],"pending_items":[]}'
        )

    provider = SimpleNamespace(chat=AsyncMock(side_effect=_chat_side_effect))

    service = ConsolidationService(
        memory_port=memory,
        provider=provider,
        model="m",
        keep_count=20,
    )
    session = SimpleNamespace(
        key="cli:1",
        messages=[
            {"role": "user", "content": "我买了 Zigbee 网关", "timestamp": "2026-03-15T10:00:00"},
            {"role": "assistant", "content": "记住了", "timestamp": "2026-03-15T10:01:00"},
        ],
        last_consolidated=0,
        _channel="cli",
        _chat_id="1",
    )

    asyncio.run(service.consolidate(session, archive_all=True))

    memory.append_history_once.assert_called_once()
    memory.save_from_consolidation.assert_awaited_once()
    # 当前 tail 太短，没有 compression 源段，因此只会调用 event + 长期记忆
    assert provider.chat.await_count == 2
    event_prompt = next(
        call.kwargs["messages"][1]["content"]
        for call in provider.chat.await_args_list
        if "记忆提取代理" in call.kwargs["messages"][0]["content"]
    )
    implicit_prompt = next(
        call.kwargs["messages"][0]["content"]
        for call in provider.chat.await_args_list
        if "长期记忆提取专家" in call.kwargs["messages"][0]["content"]
    )
    assert "## 最近三次 consolidation event" in event_prompt
    assert "用户准备下单 Zigbee 网关" in event_prompt
    assert "不能作为人物身份、说话人归属、关系判断或具体事实归属的直接证据" in event_prompt
    assert "emotional_weight" in event_prompt
    assert "emotional_weight" in implicit_prompt
    # 长期记忆 call 保存了 profile（走 save_item_with_supersede）
    memory.save_item_with_supersede.assert_awaited_once()
    assert memory.save_from_consolidation.await_args.kwargs["emotional_weight"] == 6
    assert memory.save_item_with_supersede.await_args.kwargs["emotional_weight"] == 4
    memory.write_recent_context.assert_called_once()
    assert session.last_consolidated == 0


def test_consolidation_service_uses_profile_maint_for_file_side_io():
    memory_port = SimpleNamespace(
        save_from_consolidation=AsyncMock(),
        save_item=AsyncMock(return_value="new:profile-1"),
    )
    profile_maint = SimpleNamespace(
        read_long_term=MagicMock(return_value="MEM"),
        read_history=MagicMock(return_value=""),
        read_recent_context=MagicMock(return_value=""),
        write_recent_context=MagicMock(),
        append_history_once=MagicMock(return_value=True),
        append_pending_once=MagicMock(return_value=True),
    )
    async def _chat_side_effect(**kwargs):
        text = _message_text(kwargs)
        if "近期语境压缩代理" in text:
            return _Resp(
                '{"active_topics":["用户最近关注 Zigbee 方案"],"user_preferences":[],"follow_ups":[],"avoidances":[]}'
            )
        if "长期记忆提取专家" in text:
            return _Resp('{"profile":[],"preference":[],"procedure":[]}')
        return _Resp(
            '{"history_entries":["[2026-03-15 10:00] 用户聊了 Zigbee 方案"],"pending_items":[]}'
        )

    provider = SimpleNamespace(chat=AsyncMock(side_effect=_chat_side_effect))
    service = ConsolidationService(
        memory_port=memory_port,
        profile_maint=profile_maint,
        provider=provider,
        model="m",
        keep_count=20,
        profile_extractor=None,
    )
    session = SimpleNamespace(
        key="cli:1",
        messages=[
            {"role": "user", "content": "我买了 Zigbee 网关", "timestamp": "2026-03-15T10:00:00"},
            {"role": "assistant", "content": "记住了", "timestamp": "2026-03-15T10:01:00"},
        ],
        last_consolidated=0,
        _channel="cli",
        _chat_id="1",
    )

    asyncio.run(service.consolidate(session, archive_all=True))

    profile_maint.read_long_term.assert_called_once()
    profile_maint.append_history_once.assert_called_once()
    profile_maint.write_recent_context.assert_called_once()
    memory_port.save_from_consolidation.assert_awaited_once()


def test_consolidation_event_failure_does_not_write_implicit_long_term():
    """当 event 提取返回空响应时，隐式长期记忆（profile/preference/procedure）不应写库。
    验证 Issue-1 修复：即使 implicit_task 已先于 event 完成，也不会在 event 失败时写入，
    从而保留同窗口重跑的幂等语义。
    """
    memory = SimpleNamespace(
        read_long_term=MagicMock(return_value="MEM"),
        read_history=MagicMock(return_value=""),
        read_recent_context=MagicMock(return_value=""),
        write_recent_context=MagicMock(),
        append_history_once=MagicMock(return_value=True),
        append_pending_once=MagicMock(return_value=True),
        save_from_consolidation=AsyncMock(),
        save_item=AsyncMock(return_value="new:profile-1"),
        save_item_with_supersede=AsyncMock(return_value="new:profile-1"),
    )

    # implicit LLM 调用返回合法 profile，event LLM 调用返回空（失败路径）。
    # 用 asyncio.Event 强制让 implicit 调用比 event 调用更早完成，模拟竞态。
    implicit_done = asyncio.Event()

    async def _chat_side_effect(**kwargs):
        messages = kwargs.get("messages", [])
        content = (messages[0].get("content") or "") if messages else ""
        if "procedure" in content or "preference" in content or "profile" in content:
            # implicit 调用：立即返回合法结果
            implicit_done.set()
            return _Resp('{"profile":[{"summary":"用户买了 Zigbee 网关","category":"purchase","happened_at":"2026-03-15"}],"preference":[],"procedure":[]}')
        else:
            # event 调用：等 implicit 完成后再返回空响应，确保竞态场景
            await implicit_done.wait()
            return _Resp("")

    provider = SimpleNamespace(chat=AsyncMock(side_effect=_chat_side_effect))

    service = ConsolidationService(
        memory_port=memory,
        provider=provider,
        model="m",
        keep_count=20,
    )
    session = SimpleNamespace(
        key="cli:1",
        messages=[
            {"role": "user", "content": "我买了 Zigbee 网关", "timestamp": "2026-03-15T10:00:00"},
            {"role": "assistant", "content": "记住了", "timestamp": "2026-03-15T10:01:00"},
        ],
        last_consolidated=0,
        _channel="cli",
        _chat_id="1",
    )

    asyncio.run(service.consolidate(session, archive_all=True))

    # event 失败 → last_consolidated 不推进，隐式结果不写库
    assert session.last_consolidated == 0
    memory.save_item_with_supersede.assert_not_awaited()
    memory.append_history_once.assert_not_called()
    memory.write_recent_context.assert_not_called()


def test_consolidation_recent_context_formats_user_full_and_assistant_preview():
    memory = SimpleNamespace(
        read_long_term=MagicMock(return_value="MEM"),
        read_history=MagicMock(return_value=""),
        read_recent_context=MagicMock(return_value=""),
        write_recent_context=MagicMock(),
        append_history_once=MagicMock(return_value=True),
        append_pending_once=MagicMock(return_value=True),
        save_from_consolidation=AsyncMock(),
        save_item=AsyncMock(return_value="new:profile-1"),
        save_item_with_supersede=AsyncMock(return_value="new:profile-1"),
    )
    async def _chat_side_effect(**kwargs):
        text = _message_text(kwargs)
        if "近期语境压缩代理" in text:
            return _Resp(
                '{"active_topics":["用户最近在讨论 recent context 设计"],"user_preferences":["偏好轻量共享上下文"],"follow_ups":[],"avoidances":[],"ongoing_threads":["用户正在推进 recent context 设计"]}'
            )
        if "长期记忆提取专家" in text:
            return _Resp('{"profile":[],"preference":[],"procedure":[]}')
        return _Resp(
            '{"history_entries":["[2026-03-15 10:00] 用户在讨论近期记忆设计"],"pending_items":[]}'
        )

    provider = SimpleNamespace(chat=AsyncMock(side_effect=_chat_side_effect))
    service = ConsolidationService(
        memory_port=memory,
        provider=provider,
        model="m",
        keep_count=4,
    )
    session = SimpleNamespace(
        key="cli:1",
        messages=[
            {"role": "user", "content": "第一轮", "timestamp": "2026-03-15T09:58:00"},
            {"role": "assistant", "content": "第一轮回复", "timestamp": "2026-03-15T09:59:00"},
            {
                "role": "user",
                "content": "我更想要一个轻量 recent context 文件，不要太重。",
                "timestamp": "2026-03-15T10:00:00",
            },
            {
                "role": "assistant",
                "content": "可以把 compression 和 recent turns 合并在一个短文件里，默认直接全量读取，不需要二次拆分读取。",
                "timestamp": "2026-03-15T10:01:00",
            },
        ],
        last_consolidated=0,
        _channel="cli",
        _chat_id="1",
    )

    asyncio.run(service.consolidate(session, archive_all=True))

    written = memory.write_recent_context.call_args.args[0]
    assert "# Recent Context" in written
    assert "until: 2026-03-15T09:59:00" in written
    assert "用户最近在讨论 recent context 设计" in written
    assert "## Ongoing Threads" in written
    assert "用户正在推进 recent context 设计" in written
    assert "<!-- a-preview = assistant reply preview only -->" in written
    assert "[user] 我更想要一个轻量 recent context 文件，不要太重。" in written
    assert "[a-preview] 可以把 compression 和 recent turns 合并在一个短文件里" in written


def test_consolidation_recent_context_compresses_archived_window_not_kept_gap():
    memory = SimpleNamespace(
        read_long_term=MagicMock(return_value="MEM"),
        read_history=MagicMock(return_value=""),
        read_recent_context=MagicMock(
            return_value=(
                "# Recent Context\n\n"
                "## Compression\n"
                "until: 2026-03-15T10:03:00\n"
                "- 最近持续关注：旧话题\n\n"
                "## Ongoing Threads\n"
                "- none\n\n"
                "## Recent Turns\n"
                "<!-- a-preview = assistant reply preview only -->\n"
                "[user] 第七条\n"
                "[a-preview] 第八条\n"
            )
        ),
        write_recent_context=MagicMock(),
        append_history_once=MagicMock(return_value=True),
        append_pending_once=MagicMock(return_value=True),
        save_from_consolidation=AsyncMock(),
        save_item=AsyncMock(return_value="new:profile-1"),
        save_item_with_supersede=AsyncMock(return_value="new:profile-1"),
    )

    captured_prompt = {"text": ""}

    async def _chat_side_effect(**kwargs):
        text = _message_text(kwargs)
        if "近期语境压缩代理" in text:
            captured_prompt["text"] = text
            return _Resp(
                '{"active_topics":["用户最近在继续旧对话"],"user_preferences":[],"follow_ups":[],"avoidances":[],"ongoing_threads":[]}'
            )
        if "长期记忆提取专家" in text:
            return _Resp('{"profile":[],"preference":[],"procedure":[]}')
        return _Resp(
            '{"history_entries":["[2026-03-15 10:07] 用户继续对话"],"pending_items":[]}'
        )

    provider = SimpleNamespace(chat=AsyncMock(side_effect=_chat_side_effect))
    service = ConsolidationService(
        memory_port=memory,
        provider=provider,
        model="m",
        keep_count=4,
    )
    session = SimpleNamespace(
        key="cli:1",
        messages=[
            {"role": "user", "content": "第一条", "timestamp": "2026-03-15T10:00:00"},
            {"role": "assistant", "content": "第二条", "timestamp": "2026-03-15T10:01:00"},
            {"role": "user", "content": "第三条", "timestamp": "2026-03-15T10:02:00"},
            {"role": "assistant", "content": "第四条", "timestamp": "2026-03-15T10:03:00"},
            {"role": "user", "content": "第五条", "timestamp": "2026-03-15T10:04:00"},
            {"role": "assistant", "content": "第六条", "timestamp": "2026-03-15T10:05:00"},
            {"role": "user", "content": "第七条", "timestamp": "2026-03-15T10:06:00"},
            {"role": "assistant", "content": "第八条", "timestamp": "2026-03-15T10:07:00"},
            {"role": "user", "content": "第九条", "timestamp": "2026-03-15T10:08:00"},
            {"role": "assistant", "content": "第十条", "timestamp": "2026-03-15T10:09:00"},
            {"role": "user", "content": "第十一条", "timestamp": "2026-03-15T10:10:00"},
            {"role": "assistant", "content": "第十二条", "timestamp": "2026-03-15T10:11:00"},
            {"role": "user", "content": "第十三条", "timestamp": "2026-03-15T10:12:00"},
        ],
        last_consolidated=4,
        _channel="cli",
        _chat_id="1",
    )

    asyncio.run(service.consolidate(session))

    assert "【较早窗口（本次待压缩）】\nUSER: 第五条\nASSISTANT: 第六条\nUSER: 第七条\nASSISTANT: 第八条\nUSER: 第九条" in captured_prompt["text"]
    written = memory.write_recent_context.call_args.args[0]
    assert "until: 2026-03-15T10:08:00" in written
    assert "[user] 第十三条" in written


def test_replace_recent_turns_block_preserves_existing_compression():
    existing = (
        "# Recent Context\n\n"
        "## Compression\n"
        "until: 2026-03-15T10:03:00\n"
        "- 最近持续关注：旧话题\n\n"
        "## Ongoing Threads\n"
        "- none\n\n"
        "## Recent Turns\n"
        "<!-- a-preview = assistant reply preview only -->\n"
        "[user] 旧 recent\n"
    )

    updated = _replace_recent_turns_block(existing, "[user] 新 recent")

    assert "until: 2026-03-15T10:03:00" in updated
    assert "旧话题" in updated
    assert "[user] 新 recent" in updated
    assert "[user] 旧 recent" not in updated


def test_consolidation_archive_all_compresses_full_history_before_recent_turns():
    memory = SimpleNamespace(
        read_long_term=MagicMock(return_value="MEM"),
        read_history=MagicMock(return_value=""),
        read_recent_context=MagicMock(return_value=""),
        write_recent_context=MagicMock(),
        append_history_once=MagicMock(return_value=True),
        append_pending_once=MagicMock(return_value=True),
        save_from_consolidation=AsyncMock(),
        save_item=AsyncMock(return_value="new:profile-1"),
        save_item_with_supersede=AsyncMock(return_value="new:profile-1"),
    )
    captured_prompt = {"text": ""}

    async def _chat_side_effect(**kwargs):
        text = _message_text(kwargs)
        if "近期语境压缩代理" in text:
            captured_prompt["text"] = text
            return _Resp(
                '{"active_topics":["用户最近在继续旧对话"],"user_preferences":[],"follow_ups":[],"avoidances":[],"ongoing_threads":[]}'
            )
        if "长期记忆提取专家" in text:
            return _Resp('{"profile":[],"preference":[],"procedure":[]}')
        return _Resp(
            '{"history_entries":["[2026-03-15 10:12] 用户继续对话"],"pending_items":[]}'
        )

    provider = SimpleNamespace(chat=AsyncMock(side_effect=_chat_side_effect))
    service = ConsolidationService(
        memory_port=memory,
        provider=provider,
        model="m",
        keep_count=4,
    )
    session = SimpleNamespace(
        key="cli:1",
        messages=[
            {"role": "user", "content": "第一条", "timestamp": "2026-03-15T10:00:00"},
            {"role": "assistant", "content": "第二条", "timestamp": "2026-03-15T10:01:00"},
            {"role": "user", "content": "第三条", "timestamp": "2026-03-15T10:02:00"},
            {"role": "assistant", "content": "第四条", "timestamp": "2026-03-15T10:03:00"},
            {"role": "user", "content": "第五条", "timestamp": "2026-03-15T10:04:00"},
            {"role": "assistant", "content": "第六条", "timestamp": "2026-03-15T10:05:00"},
            {"role": "user", "content": "第七条", "timestamp": "2026-03-15T10:06:00"},
            {"role": "assistant", "content": "第八条", "timestamp": "2026-03-15T10:07:00"},
            {"role": "user", "content": "第九条", "timestamp": "2026-03-15T10:08:00"},
            {"role": "assistant", "content": "第十条", "timestamp": "2026-03-15T10:09:00"},
            {"role": "user", "content": "第十一条", "timestamp": "2026-03-15T10:10:00"},
            {"role": "assistant", "content": "第十二条", "timestamp": "2026-03-15T10:11:00"},
            {"role": "user", "content": "第十三条", "timestamp": "2026-03-15T10:12:00"},
        ],
        last_consolidated=0,
        _channel="cli",
        _chat_id="1",
    )

    asyncio.run(service.consolidate(session, archive_all=True))

    prompt = next(
        call.kwargs["messages"][1]["content"]
        for call in provider.chat.await_args_list
        if "近期语境压缩代理" in call.kwargs["messages"][0]["content"]
    )
    prompt_before_recent = prompt.split("【最新 recent turns", 1)[0]
    assert "USER: 第一条" in prompt_before_recent
    assert "ASSISTANT: 第十条" in prompt_before_recent
    assert "USER: 第十一条" in prompt_before_recent
    assert "ASSISTANT: 第十二条" not in prompt_before_recent
    written = memory.write_recent_context.call_args.args[0]
    assert "until: 2026-03-15T10:10:00" in written
    assert "[user] 第十三条" in written


def test_consolidation_recent_context_invalid_json_keeps_old_compression():
    old_recent_context = (
        "# Recent Context\n\n"
        "## Compression\n"
        "until: 2026-03-15T10:03:00\n"
        "- 最近持续关注：旧话题\n\n"
        "## Ongoing Threads\n"
        "- 重要线程\n\n"
        "## Recent Turns\n"
        "<!-- a-preview = assistant reply preview only -->\n"
        "[user] 旧 recent\n"
    )
    memory = SimpleNamespace(
        read_long_term=MagicMock(return_value="MEM"),
        read_history=MagicMock(return_value=""),
        read_recent_context=MagicMock(return_value=old_recent_context),
        write_recent_context=MagicMock(),
        append_history_once=MagicMock(return_value=True),
        append_pending_once=MagicMock(return_value=True),
        save_from_consolidation=AsyncMock(),
        save_item=AsyncMock(return_value="new:profile-1"),
        save_item_with_supersede=AsyncMock(return_value="new:profile-1"),
    )

    async def _chat_side_effect(**kwargs):
        text = _message_text(kwargs)
        if "近期语境压缩代理" in text:
            return _Resp("{bad json")
        if "长期记忆提取专家" in text:
            return _Resp('{"profile":[],"preference":[],"procedure":[]}')
        return _Resp(
            '{"history_entries":["[2026-03-15 10:07] 用户继续对话"],"pending_items":[]}'
        )

    provider = SimpleNamespace(chat=AsyncMock(side_effect=_chat_side_effect))
    service = ConsolidationService(
        memory_port=memory,
        provider=provider,
        model="m",
        keep_count=4,
    )
    session = SimpleNamespace(
        key="cli:1",
        messages=[
            {"role": "user", "content": "第一条", "timestamp": "2026-03-15T10:00:00"},
            {"role": "assistant", "content": "第二条", "timestamp": "2026-03-15T10:01:00"},
            {"role": "user", "content": "第三条", "timestamp": "2026-03-15T10:02:00"},
            {"role": "assistant", "content": "第四条", "timestamp": "2026-03-15T10:03:00"},
            {"role": "user", "content": "第五条", "timestamp": "2026-03-15T10:04:00"},
            {"role": "assistant", "content": "第六条", "timestamp": "2026-03-15T10:05:00"},
            {"role": "user", "content": "第七条", "timestamp": "2026-03-15T10:06:00"},
            {"role": "assistant", "content": "第八条", "timestamp": "2026-03-15T10:07:00"},
            {"role": "user", "content": "第九条", "timestamp": "2026-03-15T10:08:00"},
            {"role": "assistant", "content": "第十条", "timestamp": "2026-03-15T10:09:00"},
            {"role": "user", "content": "第十一条", "timestamp": "2026-03-15T10:10:00"},
            {"role": "assistant", "content": "第十二条", "timestamp": "2026-03-15T10:11:00"},
            {"role": "user", "content": "第十三条", "timestamp": "2026-03-15T10:12:00"},
        ],
        last_consolidated=4,
        _channel="cli",
        _chat_id="1",
    )

    asyncio.run(service.consolidate(session))

    written = memory.write_recent_context.call_args.args[0]
    assert "until: 2026-03-15T10:08:00" in written
    assert "旧话题" in written
    assert "重要线程" in written
    assert "[user] 第十三条" in written


def test_select_recent_history_entries_returns_last_three_chunks():
    history = (
        "[2026-03-15 09:00] A\n\n"
        "[2026-03-15 09:10] B\n\n"
        "[2026-03-15 09:20] C\n\n"
        "[2026-03-15 09:30] D"
    )
    assert _select_recent_history_entries(history, limit=3) == [
        "[2026-03-15 09:10] B",
        "[2026-03-15 09:20] C",
        "[2026-03-15 09:30] D",
    ]
