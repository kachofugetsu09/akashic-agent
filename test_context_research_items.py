#!/usr/bin/env python3
"""验证 context-only 场景下 research items 的构造"""

from proactive.tick import ProactiveTick, FetchSnapshot


def test_steam_context_research_items():
    """测试 Steam context 结构能正确构造 research items"""

    # 模拟 Steam context 数据结构
    steam_context = [
        {
            "_source": "steam",
            "available": True,
            "realtime": {
                "currently_playing": "Elden Ring"
            },
            "games": [
                {"name": "Elden Ring", "recent_2w_hours": 15.5},
                {"name": "Baldur's Gate 3", "recent_2w_hours": 8.2},
                {"name": "Cyberpunk 2077", "recent_2w_hours": 3.1},
            ]
        }
    ]

    # 创建 FetchSnapshot
    fetch = FetchSnapshot()
    fetch.background_context = steam_context

    # 创建 ProactiveTick 实例（只需要 _build_context_research_items 方法）
    tick = ProactiveTick(
        cfg=type('obj', (object,), {})(),
        state=None,
        presence=None,
        rng=None,
    )

    # 调用方法
    items = tick._build_context_research_items(fetch)

    # 验证结果
    print(f"构造的 items 数量: {len(items)}")
    assert len(items) > 0, "Steam context 应该能构造出至少 1 个 research item"

    # 验证第一个 item 的内容
    first_item = items[0]
    print(f"第一个 item:")
    print(f"  source_name: {first_item.source_name}")
    print(f"  title: {first_item.title}")
    print(f"  content: {first_item.content}")

    assert first_item.source_name == "steam"
    assert "Elden Ring" in first_item.title or "Elden Ring" in first_item.content

    print("✅ Steam context research items 构造测试通过")


def test_steam_context_active_games():
    """测试 Steam context 只有活跃游戏（无 currently_playing）"""

    steam_context = [
        {
            "_source": "steam",
            "available": True,
            "realtime": {},
            "games": [
                {"name": "Baldur's Gate 3", "recent_2w_hours": 12.5},
                {"name": "Cyberpunk 2077", "recent_2w_hours": 8.0},
                {"name": "The Witcher 3", "recent_2w_hours": 2.0},  # < 5 hours, 不应包含
            ]
        }
    ]

    fetch = FetchSnapshot()
    fetch.background_context = steam_context

    tick = ProactiveTick(
        cfg=type('obj', (object,), {})(),
        state=None,
        presence=None,
        rng=None,
    )

    items = tick._build_context_research_items(fetch)

    print(f"\n构造的 items 数量: {len(items)}")
    assert len(items) > 0, "Steam 活跃游戏应该能构造出 research item"

    first_item = items[0]
    print(f"第一个 item:")
    print(f"  title: {first_item.title}")
    print(f"  content: {first_item.content}")

    assert "Baldur's Gate 3" in first_item.content
    assert "Cyberpunk 2077" in first_item.content
    assert "The Witcher 3" not in first_item.content  # < 5 hours 不应包含

    print("✅ Steam 活跃游戏 research items 构造测试通过")


def test_generic_context():
    """测试通用 topic/summary 结构"""

    generic_context = [
        {
            "_source": "generic",
            "topic": "用户最近关注的技术话题",
            "summary": "用户最近在讨论 Rust 编程语言和 WebAssembly 技术"
        }
    ]

    fetch = FetchSnapshot()
    fetch.background_context = generic_context

    tick = ProactiveTick(
        cfg=type('obj', (object,), {})(),
        state=None,
        presence=None,
        rng=None,
    )

    items = tick._build_context_research_items(fetch)

    print(f"\n构造的 items 数量: {len(items)}")
    assert len(items) > 0, "通用 context 应该能构造出 research item"

    first_item = items[0]
    print(f"第一个 item:")
    print(f"  title: {first_item.title}")
    print(f"  content: {first_item.content}")

    assert "技术话题" in first_item.title or "Rust" in first_item.content

    print("✅ 通用 context research items 构造测试通过")


if __name__ == "__main__":
    test_steam_context_research_items()
    test_steam_context_active_games()
    test_generic_context()
    print("\n所有测试通过 ✅")
