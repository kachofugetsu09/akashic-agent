import pytest

from feeds.store import FeedStore
from feeds.tools import FeedGenerateSubscribeTool


async def _fetch_fail(url: str):
    return False, "", "skip"


@pytest.mark.asyncio
async def test_generate_subscribe_success(tmp_path):
    store = FeedStore(tmp_path / "feeds.json")

    async def validator(url: str):
        return True, "ok"

    tool = FeedGenerateSubscribeTool(store, validator=validator, page_fetcher=_fetch_fail)
    result = await tool.execute(name="Paul Graham", page_url="https://paulgraham.com/")

    assert "已生成并订阅" in result
    subs = store.load()
    assert len(subs) == 1
    assert subs[0].name == "Paul Graham"
    assert subs[0].url.startswith("https://rss.diffbot.com/rss?url=")


@pytest.mark.asyncio
async def test_generate_subscribe_validation_failure(tmp_path):
    store = FeedStore(tmp_path / "feeds.json")

    async def validator(url: str):
        return False, "HTTP 400"

    tool = FeedGenerateSubscribeTool(store, validator=validator, page_fetcher=_fetch_fail)
    result = await tool.execute(name="Bad", page_url="https://example.com/bad")

    assert result == "生成的 RSS 无法使用：HTTP 400"
    assert store.load() == []


@pytest.mark.asyncio
async def test_generate_subscribe_duplicate_blocked(tmp_path):
    store = FeedStore(tmp_path / "feeds.json")

    async def validator(url: str):
        return True, "ok"

    tool = FeedGenerateSubscribeTool(store, validator=validator, page_fetcher=_fetch_fail)
    await tool.execute(name="A", page_url="https://example.com/list")
    result = await tool.execute(name="A2", page_url="https://example.com/list")

    assert "已经订阅过该地址" in result
    assert len(store.load()) == 1


@pytest.mark.asyncio
async def test_generate_subscribe_requires_http_url(tmp_path):
    store = FeedStore(tmp_path / "feeds.json")
    tool = FeedGenerateSubscribeTool(store, validator=None, page_fetcher=_fetch_fail)
    result = await tool.execute(name="A", page_url="ftp://example.com/list")

    assert result == "错误：page_url 必须是 http/https 链接"


@pytest.mark.asyncio
async def test_generate_subscribe_auto_discovers_archive_page(tmp_path):
    store = FeedStore(tmp_path / "feeds.json")

    async def validator(url: str):
        return True, "ok"

    async def page_fetcher(url: str):
        if url.endswith("/archive.html"):
            html = """
            <div class="archive-item"><a href="/post1.html"><span class="archive-date">2026-01-07</span></a></div>
            <div class="archive-item"><a href="/post2.html"><span class="archive-date">2025-12-13</span></a></div>
            """
            return True, html, "ok"
        return True, '<a href="/main.html">main</a>', "ok"

    tool = FeedGenerateSubscribeTool(store, validator=validator, page_fetcher=page_fetcher)
    result = await tool.execute(name="Blog", page_url="https://kachofugetsu09.github.io/")

    assert "自动选择页面" in result
    assert "archive.html" in result
    assert len(store.load()) == 1
    assert "archive.html" in (store.load()[0].url or "")


@pytest.mark.asyncio
async def test_generate_subscribe_rejects_low_quality_feed(tmp_path):
    store = FeedStore(tmp_path / "feeds.json")
    tool = FeedGenerateSubscribeTool(store, validator=None, page_fetcher=_fetch_fail)

    async def fake_validate(url: str):
        # 模拟被质量门禁拒绝的结果
        return False, "质量不足：条目过少（1）"

    tool._validator = fake_validate
    result = await tool.execute(name="BadFeed", page_url="https://example.com/")

    assert result == "生成的 RSS 无法使用：质量不足：条目过少（1）"
    assert store.load() == []


@pytest.mark.asyncio
async def test_generate_subscribe_prefers_sitemap_local_feed(tmp_path):
    store = FeedStore(tmp_path / "feeds.json")

    async def validator(url: str):
        return False, "should not call validator when local discovery succeeds"

    async def page_fetcher(url: str):
        if url.endswith("/sitemap.xml"):
            return True, """<?xml version="1.0"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/notes/a.html</loc><lastmod>2026-01-01T00:00:00Z</lastmod></url>
  <url><loc>https://example.com/notes/b.html</loc><lastmod>2026-01-02T00:00:00Z</lastmod></url>
  <url><loc>https://example.com/notes/c.html</loc><lastmod>2026-01-03T00:00:00Z</lastmod></url>
</urlset>""", "ok"
        if url in ("https://example.com/", "https://example.com"):
            return True, "<html></html>", "ok"
        return False, "", "not found"

    tool = FeedGenerateSubscribeTool(store, validator=validator, page_fetcher=page_fetcher)
    result = await tool.execute(name="Example", page_url="https://example.com/")

    assert "来源：站点结构发现" in result
    subs = store.load()
    assert len(subs) == 1
    assert subs[0].url.startswith("file://")


@pytest.mark.asyncio
async def test_generate_subscribe_uses_tree_crawl_when_no_sitemap_or_vp(tmp_path):
    store = FeedStore(tmp_path / "feeds.json")

    async def validator(url: str):
        return False, "should not call validator when crawl discovery succeeds"

    pages = {
        "https://example.org/": """
            <a href="/notes/index.html">Notes</a>
            <a href="/books/index.html">Books</a>
        """,
        "https://example.org/notes/index.html": """
            <a href="/notes/a.html">A</a>
            <a href="/notes/b.html">B</a>
        """,
        "https://example.org/books/index.html": """
            <a href="/books/c.html">C</a>
        """,
        "https://example.org/notes/a.html": "<h1>A</h1>",
        "https://example.org/notes/b.html": "<h1>B</h1>",
        "https://example.org/books/c.html": "<h1>C</h1>",
    }

    async def page_fetcher(url: str):
        if url.endswith("/sitemap.xml"):
            return False, "", "404"
        if url in pages:
            return True, pages[url], "ok"
        return False, "", "404"

    tool = FeedGenerateSubscribeTool(store, validator=validator, page_fetcher=page_fetcher)
    result = await tool.execute(name="Tree", page_url="https://example.org/")

    assert "来源：站点结构发现" in result
    subs = store.load()
    assert len(subs) == 1
    assert subs[0].url.startswith("file://")
