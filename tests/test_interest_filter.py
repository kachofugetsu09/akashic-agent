from feeds.base import FeedItem
from proactive.interest import InterestFilterConfig, select_interesting_items


def _item(title: str, content: str) -> FeedItem:
    return FeedItem(
        source_name="rss",
        source_type="rss",
        title=title,
        content=content,
        url=None,
        author=None,
        published_at=None,
    )


def test_interest_filter_prefers_memory_matched_items():
    cfg = InterestFilterConfig(
        enabled=True, min_score=0.10, top_k=2, exploration_ratio=0.0
    )
    memory = "只关心 PC 游戏、Elden Ring、Path of Exile。"
    items = [
        _item("Elden Ring DLC", "new trailer for pc"),
        _item("NBA trade", "sports news"),
        _item("Path of Exile 2 update", "pc build guide"),
    ]
    kept, ranked = select_interesting_items(items, memory, cfg)

    assert len(kept) == 2
    titles = {x.title for x in kept}
    assert "Elden Ring DLC" in titles
    assert "Path of Exile 2 update" in titles
    assert ranked[0][1] >= ranked[-1][1]


def test_interest_filter_keeps_exploration_samples():
    cfg = InterestFilterConfig(
        enabled=True, min_score=0.50, top_k=3, exploration_ratio=0.34
    )
    memory = "只关心魂类、PC游戏。"
    items = [
        _item("souls game", "pc souls"),
        _item("movie", "cinema"),
        _item("finance", "market"),
        _item("weather", "rain"),
    ]
    kept, _ = select_interesting_items(items, memory, cfg)
    assert len(kept) == 3
