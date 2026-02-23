from datetime import datetime, timezone
from unittest.mock import AsyncMock
import json

import pytest

from feeds.base import FeedItem
from feeds.store import FeedStore
from feeds.tools import FeedManageTool, FeedQueryTool


@pytest.mark.asyncio
async def test_feed_manage_subscribe_by_url(tmp_path):
    store = FeedStore(tmp_path / "feeds.json")
    tool = FeedManageTool(store)

    result = await tool.execute(
        action="subscribe",
        name="Example",
        url="https://example.com/feed.xml",
    )

    assert "已订阅" in result
    subs = store.load()
    assert len(subs) == 1
    assert subs[0].url == "https://example.com/feed.xml"


@pytest.mark.asyncio
async def test_feed_manage_discover_returns_preview(tmp_path):
    store = FeedStore(tmp_path / "feeds.json")
    tool = FeedManageTool(store)
    tool._gen_tool._discover_best_page = AsyncMock(return_value="https://example.com/archive.html")
    tool._gen_tool._discover_entries_from_site = AsyncMock(return_value=[
        {"title": "A", "url": "https://example.com/a.html"},
        {"title": "B", "url": "https://example.com/b.html"},
    ])

    result = await tool.execute(action="discover", page_url="https://example.com/")
    assert "候选最优: https://example.com/archive.html" in result
    assert "结构化发现条目数: 2" in result


@pytest.mark.asyncio
async def test_feed_manage_list_and_unsubscribe(tmp_path):
    store = FeedStore(tmp_path / "feeds.json")
    tool = FeedManageTool(store)
    await tool.execute(action="subscribe", name="AA", url="https://example.com/a.xml")

    listed = await tool.execute(action="list")
    assert "RSS 订阅列表" in listed
    assert "AA" in listed

    removed = await tool.execute(action="unsubscribe", name="AA")
    assert "已取消订阅" in removed
    assert store.load() == []


class _DummyRegistry:
    def __init__(self, items):
        self._items = items

    async def fetch_all(self, limit_per_source: int = 3):
        return self._items


@pytest.mark.asyncio
async def test_feed_query_latest_and_search(tmp_path):
    store = FeedStore(tmp_path / "feeds.json")
    manage = FeedManageTool(store)
    await manage.execute(action="subscribe", name="Blog", url="https://example.com/feed.xml")

    items = [
        FeedItem(
            source_name="Blog",
            source_type="rss",
            title="New Post",
            content="about codex and agent",
            url="https://example.com/new",
            author=None,
            published_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
        ),
        FeedItem(
            source_name="Blog",
            source_type="rss",
            title="Old Post",
            content="random",
            url="https://example.com/old",
            author=None,
            published_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
    ]
    query = FeedQueryTool(store, _DummyRegistry(items))

    latest = await query.execute(action="latest", source="blog", limit=1)
    assert "New Post" in latest
    assert "https://example.com/new" in latest

    search = await query.execute(action="search", keyword="codex")
    assert "New Post" in search
    assert "Old Post" not in search


@pytest.mark.asyncio
async def test_feed_query_catalog_pagination(tmp_path):
    store = FeedStore(tmp_path / "feeds.json")
    manage = FeedManageTool(store)
    await manage.execute(action="subscribe", name="Blog", url="https://example.com/feed.xml")

    items = []
    for idx in range(7):
        items.append(
            FeedItem(
                source_name="Blog",
                source_type="rss",
                title=f"P{idx}",
                content="x",
                url=f"https://example.com/p{idx}",
                author=None,
                published_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
            )
        )
    query = FeedQueryTool(store, _DummyRegistry(items))

    page1 = json.loads(await query.execute(action="catalog", source="blog", page=1, page_size=3))
    assert page1["total"] == 7
    assert page1["has_more"] is True
    assert page1["next_page"] == 2
    assert len(page1["items"]) == 3

    page3 = json.loads(await query.execute(action="catalog", source="blog", page=3, page_size=3))
    assert page3["has_more"] is False
    assert page3["next_page"] is None
    assert len(page3["items"]) == 1
