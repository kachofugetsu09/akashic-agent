import pytest

from core.net.http import (
    SharedHttpResources,
    clear_default_shared_http_resources,
    configure_default_shared_http_resources,
    get_default_shared_http_resources,
)


@pytest.mark.asyncio
async def test_default_shared_http_resources_requires_explicit_configuration():
    clear_default_shared_http_resources()

    with pytest.raises(RuntimeError, match="not configured"):
        get_default_shared_http_resources()

    resources = SharedHttpResources()
    try:
        configure_default_shared_http_resources(resources)
        assert get_default_shared_http_resources() is resources
    finally:
        clear_default_shared_http_resources(resources)
        await resources.aclose()
