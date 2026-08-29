from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import tree_sitter_javascript
from tree_sitter import Language, Node, Parser

from agent.plugins.generation import PluginGeneration, WebModuleAsset

if TYPE_CHECKING:
    from agent.plugins.snapshot import RuntimeSnapshotStore


WEB_MODULE_MAX_BYTES = 4 * 1024 * 1024
WEB_STYLESHEET_MAX_BYTES = 1024 * 1024
WEB_CATALOG_MAX_BYTES = 16 * 1024 * 1024

_CSS_URL = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.IGNORECASE | re.DOTALL)
_JAVASCRIPT = Language(tree_sitter_javascript.language())


@dataclass(frozen=True)
class WebModuleDescriptor:
    plugin_id: str
    generation_id: str
    source_revision: str
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
                    "sourceRevision": item.source_revision,
                    "module": item.asset.module,
                    "moduleSha256": item.asset.module_sha256,
                    "moduleBytes": item.asset.module_bytes,
                    "stylesheet": item.asset.stylesheet,
                    "stylesheetSha256": item.asset.stylesheet_sha256,
                    "stylesheetBytes": item.asset.stylesheet_bytes,
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


class PluginWebUiProvider:
    """Materialize a complete browser bootstrap from one snapshot lease."""

    def __init__(self, snapshot_store: RuntimeSnapshotStore) -> None:
        self._snapshot_store = snapshot_store

    async def bootstrap(self) -> bytes:
        async with self._snapshot_store.lease() as snapshot:
            catalog = snapshot.web_ui_catalog
            if catalog is None:
                raise RuntimeError("当前 snapshot 缺少 Web UI catalog")
            return catalog.encode_bootstrap(snapshot.snapshot_id)

    async def state(self) -> dict[str, str]:
        async with self._snapshot_store.lease() as snapshot:
            catalog = snapshot.web_ui_catalog
            if catalog is None:
                raise RuntimeError("当前 snapshot 缺少 Web UI catalog")
            return {"snapshotId": snapshot.snapshot_id, "catalogId": catalog.identity}


def resolve_web_module(plugin_dir: Path, declared: str | None) -> WebModuleAsset | None:
    """Freeze one self-contained browser module inside its plugin artifact."""

    if declared is None:
        return None
    root = plugin_dir.resolve(strict=False)
    raw_module = plugin_dir / declared
    module_path = raw_module.resolve(strict=False)
    if (
        not module_path.is_relative_to(root)
        or raw_module.suffix != ".js"
        or not module_path.is_file()
    ):
        raise RuntimeError(f"插件 web module 无效: {declared}")
    module = _read_text(module_path, WEB_MODULE_MAX_BYTES, "web module")
    _validate_javascript_module(module)

    stylesheet_path = raw_module.with_suffix(".css")
    stylesheet = ""
    stylesheet_sha256: str | None = None
    stylesheet_bytes = 0
    if stylesheet_path.exists():
        resolved_stylesheet = stylesheet_path.resolve(strict=False)
        if not resolved_stylesheet.is_relative_to(root) or not resolved_stylesheet.is_file():
            raise RuntimeError(f"插件 web stylesheet 无效: {stylesheet_path.name}")
        stylesheet = _read_text(
            resolved_stylesheet,
            WEB_STYLESHEET_MAX_BYTES,
            "web stylesheet",
        )
        if re.search(r"@import\b", stylesheet, re.IGNORECASE):
            raise RuntimeError("插件 web stylesheet 不得导入其他资源")
        if any(
            not value.strip().lower().startswith(("data:", "#"))
            for _, value in _CSS_URL.findall(stylesheet)
        ):
            raise RuntimeError("插件 web stylesheet 只能引用内联资源")
        stylesheet_bytes = len(stylesheet.encode("utf-8"))
        stylesheet_sha256 = _sha256_text(stylesheet)

    module_bytes = len(module.encode("utf-8"))
    return WebModuleAsset(
        module=module,
        module_sha256=_sha256_text(module),
        module_bytes=module_bytes,
        stylesheet=stylesheet,
        stylesheet_sha256=stylesheet_sha256,
        stylesheet_bytes=stylesheet_bytes,
    )


def freeze_web_ui_catalog(
    generations: Mapping[str, PluginGeneration],
    active_plugin_ids: frozenset[str],
) -> WebUiCatalog:
    """Project active generation assets into one immutable browser catalog."""

    modules = tuple(
        WebModuleDescriptor(
            plugin_id=generation.plugin_id,
            generation_id=generation.generation_id,
            source_revision=generation.source_revision,
            asset=asset,
        )
        for plugin_id in sorted(active_plugin_ids)
        for generation in (generations[plugin_id],)
        for asset in (generation.contributions.web_module,)
        if asset is not None
    )
    total_bytes = sum(
        item.asset.module_bytes + item.asset.stylesheet_bytes for item in modules
    )
    if total_bytes > WEB_CATALOG_MAX_BYTES:
        raise RuntimeError(
            f"插件 web catalog 超过 {WEB_CATALOG_MAX_BYTES} bytes: {total_bytes}"
        )
    identity_source = "\n".join(
        "\0".join(
            (
                item.plugin_id,
                item.generation_id,
                item.source_revision,
                item.asset.module_sha256,
                item.asset.stylesheet_sha256 or "",
            )
        )
        for item in modules
    )
    identity = hashlib.sha256(identity_source.encode("utf-8")).hexdigest()
    return WebUiCatalog(identity=identity, modules=modules)
def _read_text(path: Path, limit: int, label: str) -> str:
    size = path.stat().st_size
    if size <= 0 or size > limit:
        raise RuntimeError(f"插件 {label} 大小无效: {size} bytes")
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"插件 {label} 必须是 UTF-8") from error
    if "\x00" in content:
        raise RuntimeError(f"插件 {label} 不得包含 NUL")
    actual_size = len(content.encode("utf-8"))
    if actual_size > limit:
        raise RuntimeError(f"插件 {label} 超过 {limit} bytes: {actual_size}")
    return content


def _sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _validate_javascript_module(source: str) -> None:
    """Require one synchronous activate export and no module dependencies."""

    tree = Parser(_JAVASCRIPT).parse(source.encode("utf-8"))
    if tree.root_node.has_error:
        raise RuntimeError("插件 web module 不是有效的 JavaScript ESM")

    activate_exports = 0
    pending = [tree.root_node]
    while pending:
        node = pending.pop()
        if node.type in {"import", "import_statement"}:
            raise RuntimeError("插件 web module 必须是无 import 的自包含 ESM")
        if node.type == "export_statement":
            if node.child_by_field_name("source") is not None:
                raise RuntimeError("插件 web module 必须是无 import 的自包含 ESM")
            activate_exports += _is_sync_activate_export(node, source)
        pending.extend(node.named_children)

    if activate_exports != 1:
        raise RuntimeError("插件 web module 必须导出一个同步 activate(ctx)")


def _is_sync_activate_export(node: Node, source: str) -> int:
    declaration = node.child_by_field_name("declaration")
    if declaration is None or any(child.type == "default" for child in node.children):
        return 0
    if declaration.type == "function_declaration":
        name = declaration.child_by_field_name("name")
        return int(
            name is not None
            and _node_text(name, source) == "activate"
            and not any(child.type == "async" for child in declaration.children)
        )
    if declaration.type != "lexical_declaration":
        return 0
    for item in declaration.named_children:
        name = item.child_by_field_name("name")
        value = item.child_by_field_name("value")
        if (
            name is not None
            and _node_text(name, source) == "activate"
            and value is not None
            and value.type in {"arrow_function", "function_expression"}
            and not any(child.type == "async" for child in value.children)
        ):
            return 1
    return 0


def _node_text(node: Node, source: str) -> str:
    return source.encode("utf-8")[node.start_byte : node.end_byte].decode("utf-8")
