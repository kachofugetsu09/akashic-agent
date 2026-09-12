"""Offline legacy migration bundle entrypoint."""

api_version = 3
name = "legacy_upgrade"
version = "1.0.0"
inject = ()


def is_active(_services):
    """Keep the offline owner out of the runtime composition graph."""
    return False


def apply(ctx, config):
    """Provide the required v3 entrypoint without starting a runtime service."""
    _ = (ctx, config)
