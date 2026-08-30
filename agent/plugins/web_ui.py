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
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_CSS_UNSCOPED_AT_RULE = re.compile(
    r"@(?:font-face|property|counter-style|layer|(?:-webkit-)?keyframes)\b",
    re.IGNORECASE,
)
_JAVASCRIPT = Language(tree_sitter_javascript.language())
_WEB_MODULE_IMPORTS = frozenset(
    {
        "react",
        "react/jsx-runtime",
        "react-dom/client",
        "@akashic/web-ui-v1",
    }
)


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


def resolve_web_module(
    plugin_dir: Path,
    declared: str | None,
    *,
    requires: tuple[str, ...] = (),
    provides: tuple[str, ...] = (),
    contract_digests: tuple[tuple[str, str], ...] = (),
) -> WebModuleAsset | None:
    """Freeze one browser module and its Host SDK imports inside the artifact."""

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
        _validate_stylesheet_namespace(stylesheet)
        stylesheet_bytes = len(stylesheet.encode("utf-8"))
        stylesheet_sha256 = _sha256_text(stylesheet)

    module_bytes = len(module.encode("utf-8"))
    contract = json.dumps(
        {
            "contractDigests": dict(contract_digests),
            "provides": provides,
            "requires": requires,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return WebModuleAsset(
        module=module,
        module_sha256=_sha256_text(module),
        module_bytes=module_bytes,
        stylesheet=stylesheet,
        stylesheet_sha256=stylesheet_sha256,
        stylesheet_bytes=stylesheet_bytes,
        requires=requires,
        provides=provides,
        contract_digests=contract_digests,
        contract_sha256=_sha256_text(contract),
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
        for plugin_id in sorted(generations)
        if plugin_id in active_plugin_ids
        for generation in (generations[plugin_id],)
        for asset in (generation.contributions.web_module,)
        if asset is not None
    )
    _validate_web_contracts(modules)
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
                item.asset.contract_sha256,
            )
        )
        for item in modules
    )
    identity = hashlib.sha256(identity_source.encode("utf-8")).hexdigest()
    return WebUiCatalog(identity=identity, modules=modules)


def _validate_web_contracts(modules: tuple[WebModuleDescriptor, ...]) -> None:
    providers: dict[str, tuple[str, str | None]] = {
        "web.root.v1": ("core", None)
    }
    for item in modules:
        digests = dict(item.asset.contract_digests)
        for contract in item.asset.provides:
            previous = providers.get(contract)
            if previous is not None:
                raise RuntimeError(
                    f"Web contract 重复提供: {contract}: {previous[0]}, {item.plugin_id}"
                )
            providers[contract] = (item.plugin_id, digests.get(contract))
    for item in modules:
        digests = dict(item.asset.contract_digests)
        mismatched = tuple(
            contract
            for contract in item.asset.requires
            if contract in providers
            if digests.get(contract) != providers[contract][1]
            and (digests.get(contract) is not None or providers[contract][1] is not None)
        )
        if mismatched:
            raise RuntimeError(
                f"Web module contract digest 不匹配: {item.plugin_id}: {', '.join(mismatched)}"
            )


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


def _validate_stylesheet_namespace(stylesheet: str) -> None:
    """Keep document-global CSS names out of otherwise scoped plugin styles."""

    source = _CSS_COMMENT.sub("", stylesheet)
    if _CSS_UNSCOPED_AT_RULE.search(source):
        raise RuntimeError("插件 web stylesheet 不得声明全局命名 at-rule")


def _validate_javascript_module(source: str) -> None:
    """Require one synchronous activate export and only Host SDK imports."""

    tree = Parser(_JAVASCRIPT).parse(source.encode("utf-8"))
    if tree.root_node.has_error:
        raise RuntimeError("插件 web module 不是有效的 JavaScript ESM")

    local_sync_functions = frozenset(
        name
        for node in tree.root_node.named_children
        for name in _sync_function_names(node, source)
    )
    activate_exports = 0
    pending = [tree.root_node]
    while pending:
        node = pending.pop()
        if node.type == "import":
            raise RuntimeError("插件 web module 不得动态 import")
        if node.type == "import_statement":
            module = node.child_by_field_name("source")
            raw_module = _node_text(module, source) if module is not None else ""
            if (
                len(raw_module) < 2
                or raw_module[0] not in "\"'"
                or raw_module[-1] != raw_module[0]
                or raw_module[1:-1] not in _WEB_MODULE_IMPORTS
            ):
                raise RuntimeError("插件 web module 只能 import Host SDK")
        if node.type == "export_statement":
            if node.child_by_field_name("source") is not None:
                raise RuntimeError("插件 web module 不得从其他 module 导出")
            activate_exports += _is_sync_activate_export(node, source)
            if _exports_sync_activate(node, local_sync_functions, source):
                activate_exports += 1
        pending.extend(node.named_children)

    if activate_exports != 1:
        raise RuntimeError("插件 web module 必须导出一个同步 activate(ctx)")


def _is_sync_activate_export(node: Node, source: str) -> int:
    declaration = node.child_by_field_name("declaration")
    if declaration is None or any(child.type == "default" for child in node.children):
        return 0
    return int("activate" in _sync_function_names(declaration, source))


def _sync_function_names(node: Node, source: str) -> tuple[str, ...]:
    if node.type == "function_declaration":
        name = node.child_by_field_name("name")
        if name is not None and not any(child.type == "async" for child in node.children):
            return (_node_text(name, source),)
        return ()
    if node.type != "lexical_declaration":
        return ()
    names: list[str] = []
    for item in node.named_children:
        name = item.child_by_field_name("name")
        value = item.child_by_field_name("value")
        if (
            name is not None
            and value is not None
            and value.type in {"arrow_function", "function_expression"}
            and not any(child.type == "async" for child in value.children)
        ):
            names.append(_node_text(name, source))
    return tuple(names)


def _exports_sync_activate(
    node: Node,
    local_sync_functions: frozenset[str],
    source: str,
) -> bool:
    clause = next(
        (child for child in node.named_children if child.type == "export_clause"),
        None,
    )
    if clause is None:
        return False
    for child in clause.named_children:
        if child.type != "export_specifier":
            continue
        name = child.child_by_field_name("name")
        alias = child.child_by_field_name("alias")
        if (
            name is not None
            and _node_text(name, source) in local_sync_functions
            and _node_text(alias or name, source) == "activate"
        ):
            return True
    return False


def _node_text(node: Node, source: str) -> str:
    return source.encode("utf-8")[node.start_byte : node.end_byte].decode("utf-8")
