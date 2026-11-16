from __future__ import annotations

from pathlib import Path
import shutil
import httpx
import pytest

from agent.plugins.manager import PluginManager
from core.common.unix_socket import socket_alias_path
from plugins.ui.contract import WEB_UI
from tests.fixtures.plugin_workspace import initialize_plugin_workspace


def _copy_plugins(sources: Path, names: tuple[str, ...]) -> None:
    root = Path(__file__).resolve().parents[1] / "plugins"
    for name in names:
        dst = sources / name
        shutil.copytree(
            root / name,
            dst,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    # CI 环境不预先构建前端产物；补最小静态桩满足 UI 注册存在性检查。
    shell_js = sources / "shell_ui" / "web_module.js"
    if not shell_js.exists():
        shell_js.write_text("export function activate() { return () => {}; }\n", encoding="utf-8")
    host_js = sources / "host_execution" / "plugin_ui.js"
    if (sources / "host_execution").exists() and not host_js.exists():
        host_js.write_text("export default {};\n", encoding="utf-8")
    other = sources / "other_web"
    other.mkdir()
    (other / "web_module.js").write_text("export function activate() { return () => {}; }\n", encoding="utf-8")
    (other / "plugin.py").write_text(
        '''
from plugins.ui.contract import UI
api_version = 3
name = "other_web"
version = "1.0.0"
inject = (UI,)

async def apply(ctx):
    await ctx.require(UI).register(ctx, web="web_module.js")
''',
        encoding="utf-8",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("with_host_execution", [True, False])
async def test_shell_ui_runtime_version_allowed_under_shell_ui_identity(
    tmp_path: Path,
    with_host_execution: bool,
) -> None:
    sources = tmp_path / "plugins"
    workspace = tmp_path / "workspace"
    initialize_plugin_workspace(workspace)
    names = ("ui", "shell_ui", *(("host_execution",) if with_host_execution else ()))
    _copy_plugins(sources, names)

    host = PluginManager(
        [sources],
        workspace=workspace,
        installed_cache_root=tmp_path / "home/cache",
    )
    await host.load_all()
    try:
        root = host.live_root
        assert root is not None
        web_ui = root.context.require(WEB_UI)
        catalog = web_ui.catalog()
        state = await web_ui.state()
        snapshot = str(state["snapshotId"])
        catalog_id = catalog.identity
        shell_mod = next(item for item in catalog.modules if item.plugin_id.startswith("shell-ui"))
        other_mod = next(item for item in catalog.modules if item.plugin_id.startswith("other_web"))

        sock = socket_alias_path(workspace / "runtime", "dashboard.sock")
        transport = httpx.AsyncHTTPTransport(uds=str(sock))
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
            resp = await client.get(
                "/api/dashboard/runtime-version",
                headers={
                    "x-akashic-web-snapshot": snapshot,
                    "x-akashic-web-catalog": catalog_id,
                    "x-akashic-web-module": shell_mod.plugin_id,
                    "x-akashic-web-generation": shell_mod.generation_id,
                },
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert "runtime" in body and "bridge" in body and "history" in body
            if not with_host_execution:
                assert body["bridge"]["state"] == "disabled"

            # 跨插件请求同一接口仍须被拒绝（403）
            denied = await client.get(
                "/api/dashboard/runtime-version",
                headers={
                    "x-akashic-web-snapshot": snapshot,
                    "x-akashic-web-catalog": catalog_id,
                    "x-akashic-web-module": other_mod.plugin_id,
                    "x-akashic-web-generation": other_mod.generation_id,
                },
            )
            assert denied.status_code == 403
    finally:
        await host.terminate_all()
