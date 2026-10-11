"""Shared fixtures and test bootstrap helpers."""
from collections.abc import Iterator

import pytest


# 宿主缓存（字节码、manifest 事实）写在插件 home；测试不得写入开发者真实目录。
@pytest.fixture(autouse=True, scope="session")
def _isolated_plugin_home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path_factory.mktemp("plugin-home")))
        yield
