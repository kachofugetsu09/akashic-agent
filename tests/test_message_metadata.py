"""消息附加信息的提交、恢复与同步合同。"""
from contextlib import aclosing, closing
from typing import Any, cast
import pytest
from infra.channels.message_view import follow_messages, message_rows
from session.log import MessageLog
from session.message import ContentPart, Output
from plugins.content.plugin import check_text

def writer(log, namespaces=frozenset({"citation", "meme"})):
    return log.writer("s", author="assistant", source="conversation", body_types=(Output,),
                      content={"text": check_text}, message_metadata_keys=namespaces)

@pytest.mark.asyncio
async def test_unknown_metadata_survives_restart_history_and_follow_without_plugins(tmp_path):
    path = tmp_path / "sessions.db"
    extra = {"future-plugin": {"version": 27, "values": [None, True, 1, "原文", {"x": [2]}]}}
    with closing(MessageLog(path)) as log:
        saved = writer(log, frozenset(extra)).append(
            "reply", Output((ContentPart("text", "still readable"),), "complete"), metadata=extra)
    # 当前没有 Content 注册、插件模块或 schema；读取只依赖已提交的公共 Message。
    with closing(MessageLog(path)) as log:
        reader = log.reader("s")
        assert reader.get("reply") == saved
        assert log.catalog().sessions().items[0].first_message == saved
        page = reader.read_page()
        row = message_rows(page)[0]
        assert row["metadata"] == extra
        assert cast(Any, row["body"])["parts"] == [{"kind": "text", "value": "still readable"}]
        async with aclosing(follow_messages(reader, after_seq=-1)) as follower:
            assert (await anext(follower))["items"] == [row]
