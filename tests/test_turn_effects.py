from __future__ import annotations

import pytest

from agent.turn_effects import (
    PostCommitEffect,
    post_commit_effect,
    set_post_commit_effect,
)


def test_live_effect_ignores_legacy_memory_flag() -> None:
    assert post_commit_effect({"skip_post_memory": True}) is PostCommitEffect.ALLOW


def test_structured_effect_rejects_invalid_shape_and_value() -> None:
    with pytest.raises(ValueError, match="必须是 object"):
        post_commit_effect({"effects": "suppress"})
    with pytest.raises(ValueError):
        post_commit_effect({"effects": {"post_commit": "unknown"}})


def test_effect_writer_preserves_sibling_effect_fields() -> None:
    metadata: dict[str, object] = {"effects": {"audit": "allow"}}

    set_post_commit_effect(metadata, PostCommitEffect.SUPPRESS)

    assert metadata == {"effects": {"audit": "allow", "post_commit": "suppress"}}


def test_effect_writer_rejects_malformed_existing_container() -> None:
    with pytest.raises(ValueError, match="必须是 object"):
        set_post_commit_effect(
            {"effects": "suppress"},
            PostCommitEffect.SUPPRESS,
        )
