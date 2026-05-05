from core.memory.markdown import _format_pending_items


def test_format_pending_items_keeps_allowed_tags_only():
    text = _format_pending_items(
        [
            {"tag": "identity", "content": "长期维护一个个人项目，主要使用 Python。"},
            {"tag": "preference", "content": "回复保持简洁。"},
            {"tag": "unknown", "content": "should be dropped"},
            {"tag": "", "content": "empty tag should be dropped"},
            {"tag": "requested_memory", "content": ""},
        ]
    )

    assert "- [identity] 长期维护一个个人项目，主要使用 Python。" in text
    assert "- [preference] 回复保持简洁。" in text
    assert "should be dropped" not in text


def test_format_pending_items_deduplicates_and_normalizes_tags():
    text = _format_pending_items(
        [
            {"tag": "Preference", "content": "不要在非游戏话题强行套游戏比喻。"},
            {"tag": "preference", "content": "不要在非游戏话题强行套游戏比喻。"},
            {"tag": "health_long_term", "content": "有长期健康管理需求。"},
        ]
    )

    assert text.count("- [preference] 不要在非游戏话题强行套游戏比喻。") == 1
    assert "- [health_long_term] 有长期健康管理需求。" in text
