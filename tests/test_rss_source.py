import asyncio

import httpx

from feeds.base import FeedSubscription
from feeds.rss import RSSFeedSource


def _make_source(name: str = "test") -> RSSFeedSource:
    sub = FeedSubscription.new(type="rss", name=name, url="https://example.com/rss")
    return RSSFeedSource(sub)


def test_parse_rss_with_leading_whitespace_before_xml_decl():
    source = _make_source()
    xml_text = """  <?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Post A</title>
      <link>https://example.com/a</link>
      <description>Hello</description>
      <pubDate>Mon, 23 Feb 2026 00:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""
    items = source._parse(xml_text, limit=5)
    assert len(items) == 1
    assert items[0].title == "Post A"
    assert items[0].url == "https://example.com/a"


def test_parse_xcancel_whitelist_feed_returns_empty_items():
    source = _make_source("xcancel")
    xml_text = """  <?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>RSS reader not yet whitelisted!</title>
    <description>RSS reader not yet whitelist!</description>
    <item>
      <title>RSS reader not yet whitelisted!</title>
      <link>https://rss.xcancel.com/foo/rss</link>
      <description>placeholder</description>
    </item>
  </channel>
</rss>
"""
    items = source._parse(xml_text, limit=5)
    assert items == []


def test_fetch_via_curl_retries_then_succeeds(monkeypatch):
    sub = FeedSubscription.new(
        type="rss", name="xcancel", url="https://rss.xcancel.com/foo/rss"
    )
    source = RSSFeedSource(sub)

    xml_ok = b"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<rss version=\"2.0\"><channel><item><title>A</title><link>https://e/a</link><description>ok</description></item></channel></rss>"""
    plans = [
        (35, b"", b"curl: (35) TLS connect error"),
        (0, xml_ok, b""),
    ]
    calls: list[tuple] = []

    class _Proc:
        def __init__(self, rc: int, out: bytes, err: bytes) -> None:
            self.returncode = rc
            self._out = out
            self._err = err

        async def communicate(self):
            return self._out, self._err

    async def _fake_create_subprocess_exec(*args, **kwargs):
        calls.append((args, kwargs))
        rc, out, err = plans.pop(0)
        return _Proc(rc, out, err)

    monkeypatch.setattr("feeds.rss._FETCH_RETRY_DELAYS_S", (0.0,))
    monkeypatch.setattr(
        "feeds.rss.asyncio.create_subprocess_exec", _fake_create_subprocess_exec
    )

    items = asyncio.run(source._fetch_via_curl(limit=5))

    assert len(calls) == 2
    assert len(items) == 1
    assert items[0].title == "A"


def test_fetch_http_retries_then_succeeds(monkeypatch):
    sub = FeedSubscription.new(type="rss", name="http", url="https://example.com/rss")
    source = RSSFeedSource(sub)

    xml_ok = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><item><title>B</title><link>https://e/b</link><description>ok</description></item></channel></rss>"""

    class _Resp:
        text = xml_ok

        def raise_for_status(self):
            return None

    class _Client:
        calls = 0

        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url: str):
            _Client.calls += 1
            if _Client.calls == 1:
                raise httpx.ReadTimeout("", request=None)
            return _Resp()

    monkeypatch.setattr("feeds.rss._FETCH_RETRY_DELAYS_S", (0.0,))
    monkeypatch.setattr("feeds.rss.httpx.AsyncClient", _Client)

    items = asyncio.run(source.fetch(limit=5))

    assert _Client.calls == 2
    assert len(items) == 1
    assert items[0].title == "B"


def test_fetch_http_non_retryable_status_fails_fast(monkeypatch):
    sub = FeedSubscription.new(type="rss", name="http", url="https://example.com/rss")
    source = RSSFeedSource(sub)

    class _Client:
        calls = 0

        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url: str):
            _Client.calls += 1
            req = httpx.Request("GET", url)
            resp = httpx.Response(404, request=req)
            raise httpx.HTTPStatusError("not found", request=req, response=resp)

    monkeypatch.setattr("feeds.rss._FETCH_RETRY_DELAYS_S", (0.0, 0.0))
    monkeypatch.setattr("feeds.rss.httpx.AsyncClient", _Client)

    items = asyncio.run(source.fetch(limit=5))

    assert items == []
    assert _Client.calls == 1


def test_fetch_http_parse_error_retries_once(monkeypatch):
    sub = FeedSubscription.new(type="rss", name="http", url="https://example.com/rss")
    source = RSSFeedSource(sub)

    xml_ok = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><item><title>C</title><link>https://e/c</link><description>ok</description></item></channel></rss>"""

    class _Resp:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self):
            return None

    class _Client:
        calls = 0

        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url: str):
            _Client.calls += 1
            if _Client.calls == 1:
                return _Resp("<rss><channel><item>")
            return _Resp(xml_ok)

    monkeypatch.setattr("feeds.rss._FETCH_RETRY_DELAYS_S", (0.0, 0.0))
    monkeypatch.setattr("feeds.rss.httpx.AsyncClient", _Client)

    items = asyncio.run(source.fetch(limit=5))

    assert _Client.calls == 2
    assert len(items) == 1
    assert items[0].title == "C"


def test_fetch_via_curl_timeout_kills_process_before_retry(monkeypatch):
    sub = FeedSubscription.new(
        type="rss", name="xcancel", url="https://rss.xcancel.com/foo/rss"
    )
    source = RSSFeedSource(sub)

    xml_ok = b"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<rss version=\"2.0\"><channel><item><title>D</title><link>https://e/d</link><description>ok</description></item></channel></rss>"""

    procs = []

    class _Proc:
        def __init__(self, out: bytes, err: bytes, rc: int = 0) -> None:
            self.returncode = rc
            self._out = out
            self._err = err
            self.killed = 0
            self.communicate_calls = 0

        async def communicate(self):
            self.communicate_calls += 1
            return self._out, self._err

        def kill(self):
            self.killed += 1

    async def _fake_create_subprocess_exec(*args, **kwargs):
        p = _Proc(xml_ok, b"", 0)
        procs.append(p)
        return p

    wait_calls = {"n": 0}

    async def _fake_wait_for(coro, timeout=None):
        wait_calls["n"] += 1
        if wait_calls["n"] == 1:
            coro.close()
            raise asyncio.TimeoutError()
        return await coro

    monkeypatch.setattr("feeds.rss._FETCH_RETRY_DELAYS_S", (0.0,))
    monkeypatch.setattr(
        "feeds.rss.asyncio.create_subprocess_exec", _fake_create_subprocess_exec
    )
    monkeypatch.setattr("feeds.rss.asyncio.wait_for", _fake_wait_for)

    items = asyncio.run(source._fetch_via_curl(limit=5))

    assert len(procs) == 2
    assert procs[0].killed == 1
    assert procs[0].communicate_calls >= 1
    assert len(items) == 1
    assert items[0].title == "D"
