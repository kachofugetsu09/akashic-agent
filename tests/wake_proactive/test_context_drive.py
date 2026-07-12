from plugins.wake_proactive.context_drive import evaluate_context


def test_ordinary_context_refresh_never_contacts_user() -> None:
    first = evaluate_context(
        {
            "presence": "active",
            "interruptibility": "high",
            "confidence": 0.9,
        }
    )
    refreshed = evaluate_context(
        {
            "presence": "active",
            "interruptibility": 0.82,
            "confidence": 0.9,
        },
        previous=first.context,
    )

    assert refreshed.signal == "refresh"
    assert refreshed.should_contact is False
    assert refreshed.context.transition == ""
    assert refreshed.changed_fields == ()


def test_presence_transition_only_requests_reevaluation() -> None:
    sleeping = evaluate_context(
        {"sleeping": True, "confidence": 0.95}
    ).context

    result = evaluate_context(
        {"presence": "awake", "confidence": 0.9},
        previous=sleeping,
    )

    assert result.context.presence == "active"
    assert result.context.transition == "sleeping->active"
    assert result.signal == "reevaluate"
    assert result.should_contact is False
    assert result.changed_fields == ("presence", "interruptibility")


def test_low_confidence_transition_remains_refresh() -> None:
    previous = evaluate_context(
        {"presence": "offline", "confidence": 0.9}
    ).context

    result = evaluate_context(
        {"presence": "active", "confidence": 0.3},
        previous=previous,
    )

    assert result.context.transition == "offline->active"
    assert result.signal == "refresh"
    assert result.should_contact is False


def test_explicit_domain_transition_is_preserved() -> None:
    result = evaluate_context(
        {
            "presence": "idle",
            "transition": "in_game->idle",
            "confidence": 0.8,
            "observed_at": "2026-07-12T12:00:00Z",
            "expires_at": "2026-07-12T12:05:00Z",
        }
    )

    assert result.signal == "reevaluate"
    assert result.context.transition == "in_game->idle"
    assert result.context.observed_at is not None
    assert result.context.expires_at is not None
