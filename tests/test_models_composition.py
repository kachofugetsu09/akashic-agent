"""Context composition keeps token shares when display labels overlap."""

from plugins.models.composition import bounded, totals
from plugins.models.usage import _scaled


def test_remainder_label_collision_preserves_category_shares() -> None:
    parts = {
        "injected": {"其余 1 项": 100, **{f"source-{i}": 10 for i in range(29)}, "last": 1},
        "user": {"": 391},
    }

    detail = bounded(parts)

    assert totals(detail) == totals(parts)
    assert detail["injected"]["其余 1 项"] == 101
    assert len(detail["injected"]) <= 31
    scaled = _scaled(detail, 782)
    assert scaled is not None
    assert totals(scaled) == _scaled(totals(parts), 782)
