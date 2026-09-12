from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from enum import StrEnum

TURN_EFFECTS_KEY = "effects"
POST_COMMIT_EFFECT_KEY = "post_commit"


class TurnStorage(StrEnum):
    """Declare whether one Turn is part of durable Session history."""

    DURABLE = "durable"
    IN_MEMORY = "in_memory"


class PostCommitEffect(StrEnum):
    """Declare whether durable projections may consume a closed Turn."""

    ALLOW = "allow"
    SUPPRESS = "suppress"


def post_commit_effect(metadata: Mapping[str, object] | None) -> PostCommitEffect:
    """Read the generic post-commit effect from a live Turn."""

    if not metadata:
        return PostCommitEffect.ALLOW
    raw_effects = metadata.get(TURN_EFFECTS_KEY)
    if raw_effects is not None:
        if not isinstance(raw_effects, Mapping):
            raise ValueError("Turn effects metadata 必须是 object")
        raw_post_commit = raw_effects.get(POST_COMMIT_EFFECT_KEY)
        if raw_post_commit is not None:
            return PostCommitEffect(raw_post_commit)
    return PostCommitEffect.ALLOW


def set_post_commit_effect(
    metadata: MutableMapping[str, object],
    effect: PostCommitEffect,
) -> None:
    """Write the owned effect field without replacing other plugin effects."""

    raw_effects = metadata.get(TURN_EFFECTS_KEY)
    if raw_effects is not None and not isinstance(raw_effects, Mapping):
        raise ValueError("Turn effects metadata 必须是 object")
    effects = dict(raw_effects) if raw_effects is not None else {}
    effects[POST_COMMIT_EFFECT_KEY] = effect.value
    metadata[TURN_EFFECTS_KEY] = effects


def suppresses_post_commit(metadata: Mapping[str, object] | None) -> bool:
    return post_commit_effect(metadata) is PostCommitEffect.SUPPRESS
