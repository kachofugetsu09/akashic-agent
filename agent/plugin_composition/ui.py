"""Web 与 Dashboard 的领域合同；注册和校验由普通 provider 执行。"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from types import ModuleType

from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.routing import Match, WebSocketRoute

from agent.plugin_composition.context import Context
from agent.plugin_composition.effect import Effect
from agent.plugin_composition.model import ServiceKey

DashboardRoute = APIRoute | WebSocketRoute


@dataclass(frozen=True)
class WebModuleAsset:
    module: str
    module_sha256: str
    module_bytes: int
    stylesheet: str
    stylesheet_sha256: str | None
    stylesheet_bytes: int
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    contract_digests: tuple[tuple[str, str], ...] = ()
    contract_sha256: str = ""


@dataclass(frozen=True)
class WebModuleDescriptor:
    plugin_id: str
    registration_uuid: str
    generation_id: str
    asset: WebModuleAsset


@dataclass(frozen=True)
class WebUiCatalog:
    identity: str
    modules: tuple[WebModuleDescriptor, ...]

    def encode_bootstrap(self, snapshot_id: str) -> bytes:
        """Encode one exact catalog together with every executable byte."""

        payload = {
            "schemaVersion": 1,
            "snapshotId": snapshot_id,
            "catalogId": self.identity,
            "modules": [
                {
                    "pluginId": item.plugin_id,
                    "generationId": item.generation_id,
                    "module": item.asset.module,
                    "moduleSha256": item.asset.module_sha256,
                    "moduleBytes": item.asset.module_bytes,
                    "stylesheet": item.asset.stylesheet,
                    "stylesheetSha256": item.asset.stylesheet_sha256,
                    "stylesheetBytes": item.asset.stylesheet_bytes,
                    "requires": list(item.asset.requires),
                    "provides": list(item.asset.provides),
                    "contractDigests": dict(item.asset.contract_digests),
                    "contractSha256": item.asset.contract_sha256,
                }
                for item in self.modules
            ],
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")


@dataclass(frozen=True)
class DashboardBinding:
    plugin_id: str
    app: FastAPI
    routes: tuple[DashboardRoute, ...]
    context: Context
    generation_id: str
    has_web: bool
    runtime_workspace: Path
    runtime_data_root: Path
    module_name: str

    def matches(self, scope: dict[str, Any]) -> bool:
        return any(route.matches(scope)[0] is Match.FULL for route in self.routes)


class UiRegistry(Protocol):
    @property
    def root_instance_token(self) -> object: ...

    async def register(
        self, ctx: Context, *, web: str | None = None, dashboard: Callable[[], ModuleType] | None = None,
        requires: tuple[str, ...] = (), provides: tuple[str, ...] = (),
        contract_digests: Mapping[str, str] | None = None,
    ) -> Effect: ...

    def catalog(self) -> WebUiCatalog: ...

    def bindings(self) -> tuple[DashboardBinding, ...]: ...


class WebUiProvider(Protocol):
    async def bootstrap(self) -> bytes: ...

    async def state(self) -> dict[str, str]: ...


UI = ServiceKey[UiRegistry]("ui.v1")
WEB_UI = ServiceKey[WebUiProvider]("core.web_ui.v1")
DASHBOARD_ROUTES = ServiceKey[tuple[object, ...]]("core.dashboard_routes.v1")
