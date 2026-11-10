#!/usr/bin/env python3
"""发布前门禁：目标发行版中变更插件的构建产物能否与同一发行版的宿主（外壳/主题/SDK）产物并存。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import tree_sitter_javascript
from tree_sitter import Language, Node, Parser

# 宿主产物：ui 提供外壳、主题运行时与 SDK 垫片，akashic_clients 提供聊天样式与 chat-app 入口。
HOST_DIRS = (("ui", "static/dashboard"), ("akashic_clients", "static/chat"))
# 与 build_plugin_distribution._GENERATED_PLUGIN_UI_FILES 同一份清单：构建期生成并拷进插件制品的界面资产。
GENERATED_UI_FILES = frozenset({"plugin_ui.js", "web_module.js", "web_module.css", "message_ui.js", "message_ui.css"})
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_PUBLIC_VAR = re.compile(r"var\(\s*(--(?:ak|md|chat)-[\w-]+)\s*([,)])")
_CSS_DEFINITION = re.compile(r"(--[\w-]+)\s*:")
# 与 scripts/check_css_tokens.mjs 的 SCRIPT_DEFINITION 同一规则：脚本真正写入变量的三种形式。
_SCRIPT_DEFINITION = re.compile(r"""setProperty\(\s*["'`](--[\w-]+)|["'`](--[\w-]+)["'`]\s*[:\]]|[`"';\s{](--[\w-]+):""")
# theme-runtime 的 LEGACY_COLOR_ALIASES 在压缩产物里形如 bgCanvas:["material","background"]；先按字面尾部定位再回看键名。
_THEME_ALIAS = re.compile(r""":\[\s*["'](?:material|domain)["']\s*,""")
_ALIAS_KEY = re.compile(r"""(\w+)["']?\s*$""")
_IMPORT_MAP = re.compile(r"<script type=\"importmap\">(.*?)</script>", re.DOTALL)
_PYTHON_VERSION = re.compile(rb"--python-version\s+(\d+)\.(\d+)")
_INPUT_API = re.compile(rb"^PLUGIN_INPUT_API\s*=\s*(\d+)\s*$", re.MULTILINE)
_JAVASCRIPT = Language(tree_sitter_javascript.language())


@dataclass(frozen=True)
class Failure:
    plugin: str
    check: str
    detail: str


@dataclass
class Report:
    failures: list[Failure] = field(default_factory=list[Failure])
    notes: list[str] = field(default_factory=list[str])
    counts: dict[str, int] = field(default_factory=dict[str, int])

    def fail(self, plugin: str, check: str, detail: str) -> None:
        self.failures.append(Failure(plugin, check, detail))

    def count(self, key: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _kebab(role: str) -> str:
    return re.sub(r"[A-Z]", lambda match: "-" + match.group(0).lower(), role)


# 发行版里的 Git bundle 是插件制品的记录：取出 content_commit 中声明资产的原始字节，与解包源码逐字节核对。
class BundleRecords:
    def __init__(self, distribution: Path, workdir: Path) -> None:
        self._distribution = distribution
        self._repo = workdir / "records.git"
        _ = subprocess.run(["git", "init", "-q", "--bare", str(self._repo)], check=True)
        # 临时仓库关掉自动 gc 与后台维护：否则 fetch 结束后的维护进程还在写 objects，
        # TemporaryDirectory 清理时会与它撞成 "Directory not empty"（实测默认配置约 1/3 概率）。
        for key, value in (("gc.auto", "0"), ("maintenance.auto", "false")):
            _ = self._git("config", key, value)

    def _git(self, *args: str, stdin: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
        # 临时仓库关闭自动 gc/maintenance：连续 fetch 几十个 bundle 会触发后台整理，与清理临时目录竞争。
        return subprocess.run(["git", "-c", "gc.auto=0", "-c", "maintenance.auto=false", "-C", str(self._repo), *args],
                              input=stdin, capture_output=True, check=False)

    def fetch(self, row: dict[str, str], report: Report) -> bool:
        # 1. bundle 自身摘要必须等于发行记录 2. 头提交必须是记录的 source_revision
        bundle = self._distribution / row["file"]
        if not bundle.is_file() or _sha256(bundle.read_bytes()) != row["sha256"]:
            report.fail(row["name"], "asset", f"bundle {row['file']} 缺失或 sha256 与 distribution.json 不符")
            return False
        ref = f"refs/p/{row['name']}"
        fetched = self._git("fetch", "-q", "--no-tags", str(bundle), f"refs/heads/source:{ref}")
        if fetched.returncode or self._git("rev-parse", ref).stdout.decode().strip() != row["source_revision"]:
            report.fail(row["name"], "asset", f"bundle 头提交不是 source_revision: {fetched.stderr.decode().strip()}")
            return False
        return True

    def blobs(self, queries: list[tuple[str, str]]) -> dict[tuple[str, str], bytes | None]:
        # cat-file --batch 按输入顺序逐条应答：缺失时一行 "<name> missing"，否则头行后跟定长内容与换行。
        output = self._git("cat-file", "--batch", stdin="".join(f"{c}:{p}\n" for c, p in queries).encode()).stdout
        result: dict[tuple[str, str], bytes | None] = {}
        offset = 0
        for query in queries:
            end = output.index(b"\n", offset)
            header = output[offset:end].split()
            offset = end + 1
            if header[-1] == b"missing":
                result[query] = None
                continue
            size = int(header[2])
            result[query] = output[offset:offset + size] if header[1] == b"blob" else None
            offset += size + 1
        return result


# 插件源码里以字面量声明的界面资产：PluginUiDefinition(module=, stylesheet=) 与 register(web=)。
def _literal_assets(source: Path) -> set[str]:
    assets: set[str] = set()
    for path in source.rglob("*.py"):
        content = path.read_bytes()
        if (any(part in {"static", "node_modules", "__pycache__"} for part in path.relative_to(source).parts)
                or not any(marker in content for marker in (b"web=", b"module=", b"stylesheet="))):
            continue
        for node in ast.walk(ast.parse(content, str(path))):
            for keyword in node.keywords if isinstance(node, ast.Call) else ():
                value = cast(object, keyword.value.value) if isinstance(keyword.value, ast.Constant) else None
                if keyword.arg in {"web", "module", "stylesheet"} and isinstance(value, str) and value.endswith((".js", ".css")):
                    assets.add(value)
    return assets


# 检查 1：声明资产存在、非空、与 bundle 记录的字节同摘要；返回可继续检查的制品路径。
def check_assets(row: dict[str, str], source: Path, records: BundleRecords, report: Report) -> list[Path]:
    name, commit = row["name"], row["content_commit"]
    provenance = source / ".akashic-source.json"
    if not provenance.is_file() or json.loads(provenance.read_bytes()) != {"commit": commit, "path": row["source_path"]}:
        report.fail(name, "asset", f"sources/{name}/.akashic-source.json 与发行记录 content_commit 不符")
    # 生成文件名以 bundle 记录为准：mount_settings 等宿主助手里的 web="web_module.js" 属于调用方，不属于定义它的插件。
    literal = (_literal_assets(source) if source.is_dir() else set[str]()) - GENERATED_UI_FILES
    candidates = sorted(literal | GENERATED_UI_FILES)
    recorded = records.blobs([(commit, path) for path in candidates])
    present: list[Path] = []
    for path in candidates:
        expected = recorded[(commit, path)]
        if expected is None and path not in literal:
            continue
        report.count("assets")
        artifact = source / path
        content = artifact.read_bytes() if artifact.is_file() else None
        if content is None:
            report.fail(name, "asset", f"声明的 {path} 不在制品 sources/{name}/ 中")
        elif not content.strip():
            report.fail(name, "asset", f"{path} 为空")
        elif expected is not None and _sha256(content) != _sha256(expected):
            report.fail(name, "asset", f"{path} sha256 {_sha256(content)[:12]} 与 bundle 记录 {_sha256(expected)[:12]} 不符")
        else:
            present.append(artifact)
    return present


def _host_files(distribution: Path, suffixes: tuple[str, ...]) -> Iterator[Path]:
    for plugin, directory in HOST_DIRS:
        root = distribution / "sources" / plugin / directory
        yield from (path for path in sorted(root.rglob("*")) if path.suffix in suffixes and path.is_file())


# 宿主定义的变量：宿主样式表的自定义属性、宿主脚本写入的变量、主题运行时按目录角色生成的颜色变量。
def host_tokens(distribution: Path, report: Report) -> set[str]:
    tokens: set[str] = set()
    for path in _host_files(distribution, (".css", ".html")):
        tokens.update(_CSS_DEFINITION.findall(_CSS_COMMENT.sub("", path.read_text(encoding="utf-8"))))
    aliases: set[str] = set()
    for path in _host_files(distribution, (".js", ".html")):
        text = path.read_text(encoding="utf-8", errors="replace")
        tokens.update(match.group(1) or match.group(2) or match.group(3) for match in _SCRIPT_DEFINITION.finditer(text))
        for match in _THEME_ALIAS.finditer(text):
            key = _ALIAS_KEY.search(text, max(0, match.start() - 64), match.start())
            aliases.update((key.group(1),) if key is not None else ())
    catalog = distribution / "sources/ui/static/dashboard/akashic-theme-catalog.json"
    if not catalog.is_file():
        report.fail("ui", "css-var", "宿主缺少 static/dashboard/akashic-theme-catalog.json，主题颜色变量无从确定")
        return tokens
    # 与 theme-runtime.themeCss() 同形：颜色角色带 -rgb 通道变量，图表色不带。
    theme = cast(dict[str, dict[str, str]], json.loads(catalog.read_bytes())["themes"][0])
    colors = [f"--md-sys-color-{_kebab(role)}" for role in theme["material"]]
    colors += [f"--ak-sys-color-{_kebab(role)}" for role in theme["domain"]]
    colors += [f"--ak-color-{_kebab(role)}" for role in aliases]
    tokens.update(name for item in colors for name in (item, item + "-rgb"))
    tokens.update(f"--ak-chart-{role}" for role in theme["chart"])
    return tokens


# 检查 2：插件样式里没有回退值的公开命名空间变量必须由宿主定义。
def check_css_vars(name: str, assets: list[Path], tokens: set[str], report: Report) -> None:
    for path in (item for item in assets if item.suffix == ".css"):
        unresolved: dict[str, int] = {}
        for match in _PUBLIC_VAR.finditer(_CSS_COMMENT.sub("", path.read_text(encoding="utf-8"))):
            report.count("var-refs")
            if match.group(2) == ")" and match.group(1) not in tokens:
                unresolved[match.group(1)] = unresolved.get(match.group(1), 0) + 1
        for token, times in sorted(unresolved.items()):
            report.fail(name, "css-var", f"{path.name} 引用 var({token}) 无回退值，宿主产物未定义（{times} 处）")


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8").strip("\"'")


def _parse(path: Path) -> tuple[Node, bytes]:
    source = path.read_bytes()
    return Parser(_JAVASCRIPT).parse(source).root_node, source


def _pattern_names(node: Node, source: bytes) -> Iterator[str]:
    if node.type in {"identifier", "shorthand_property_identifier_pattern"}:
        yield _text(node, source)
    elif node.type in {"pair_pattern", "object_assignment_pattern", "assignment_pattern"}:
        target = node.child_by_field_name("value") or node.child_by_field_name("left")
        yield from _pattern_names(target, source) if target is not None else ()
    elif node.type in {"object_pattern", "array_pattern", "rest_pattern"}:
        for child in node.named_children:
            yield from _pattern_names(child, source)


# 宿主 SDK 模块的导出名；"*" 表示存在 export * 无法静态封闭。
def module_exports(path: Path) -> set[str]:
    root, source = _parse(path)
    names: set[str] = set()
    for node in (item for item in root.named_children if item.type == "export_statement"):
        names.update(kind for kind in ("default", "*") if any(child.type == kind for child in node.children))
        declaration = node.child_by_field_name("declaration")
        if declaration is not None and declaration.type in {"lexical_declaration", "variable_declaration"}:
            for declarator in declaration.named_children:
                target = declarator.child_by_field_name("name")
                names.update(_pattern_names(target, source) if target is not None else ())
        elif declaration is not None and (identifier := declaration.child_by_field_name("name")) is not None:
            names.add(_text(identifier, source))
        for clause in (child for child in node.named_children if child.type == "export_clause"):
            for specifier in clause.named_children:
                exported = specifier.child_by_field_name("alias") or specifier.child_by_field_name("name")
                names.update((_text(exported, source),) if exported is not None else ())
    return names


# 插件模块的静态 import：(说明符, 导入名集合)；默认导入记为 "default"，namespace 导入记为 "*"。
def module_imports(path: Path) -> list[tuple[str, set[str]]]:
    root, source = _parse(path)
    imports: list[tuple[str, set[str]]] = []
    pending = [root]
    while pending:
        node = pending.pop()
        pending.extend(node.named_children)
        specifier = node.child_by_field_name("source")
        if node.type != "import_statement" or specifier is None:
            continue
        names: set[str] = set()
        for clause in (child for child in node.named_children if child.type == "import_clause"):
            for part in clause.named_children:
                names.update({"identifier": ("default",), "namespace_import": ("*",)}.get(part.type, ()))
                for item in part.named_children if part.type == "named_imports" else ():
                    imported = item.child_by_field_name("name")
                    names.update((_text(imported, source),) if imported is not None else ())
        imports.append((_text(specifier, source), names))
    return imports


# 外壳 index.html 的 import map 决定说明符落到哪个宿主文件；/assets 先找聊天产物再找外壳产物（两者都挂在 /assets）。
def host_import_map(distribution: Path, report: Report) -> dict[str, Path]:
    index = distribution / "sources/ui/static/dashboard/index.html"
    match = _IMPORT_MAP.search(index.read_text(encoding="utf-8")) if index.is_file() else None
    if match is None:
        report.fail("ui", "sdk-export", "宿主 static/dashboard/index.html 缺少 import map")
        return {}
    resolved: dict[str, Path] = {}
    for specifier, url in cast(dict[str, str], json.loads(match.group(1))["imports"]).items():
        path = url.split("?", 1)[0]
        roots = [("/dashboard/assets/", "ui/static/dashboard")] if path.startswith("/dashboard/assets/") else [
            ("/assets/", "akashic_clients/static/chat"), ("/assets/", "ui/static/dashboard")]
        candidates = [distribution / "sources" / root / path.removeprefix(prefix) for prefix, root in roots]
        resolved[specifier] = next((item for item in candidates if item.is_file()), candidates[0])
    return resolved


# 检查 3：插件模块从 import map 外部依赖导入的名字必须由同一发行版的宿主模块导出。
def check_sdk_imports(
    name: str, assets: list[Path], import_map: dict[str, Path], exports: dict[str, set[str]], report: Report,
) -> None:
    for path in (item for item in assets if item.suffix == ".js"):
        for specifier, names in module_imports(path):
            if specifier.startswith((".", "/")):
                continue
            report.count("imports")
            target = import_map.get(specifier)
            if target is None or not target.is_file():
                report.fail(name, "sdk-export", f"{path.name} 导入 {specifier}，宿主 import map 没有可用模块")
                continue
            if specifier not in exports:
                exports[specifier] = module_exports(target)
            available = exports[specifier]
            missing = [] if "*" in available else sorted(names - available - {"*"})
            if missing:
                report.fail(name, "sdk-export", f"{path.name} 从 {specifier} 导入 {', '.join(missing)}，宿主 {target.name} 未导出")


# Core 的运行合同：binding_api 取 core.tar 的 PLUGIN_INPUT_API，python_tag 取 Core 锁文件声明的解释器版本。
def core_runtime(distribution: Path, python_tag: str | None, report: Report) -> dict[str, object] | None:
    with tarfile.open(distribution / "core.tar") as archive:
        member = archive.extractfile("agent/plugins/input_preparation.py")
        api = _INPUT_API.search(member.read()) if member is not None else None
    lock = distribution / "requirements.lock"
    version = _PYTHON_VERSION.search(lock.read_bytes()) if lock.is_file() else None
    tag = python_tag or (f"cpython-{version.group(1).decode()}{version.group(2).decode()}" if version else None)
    if api is None or tag is None:
        report.fail("core", "runtime", "无法从 core.tar 与 requirements.lock 确定 binding_api/python_tag")
        return None
    return {"python_tag": tag, "binding_api": int(api.group(1))}


# 检查 4：行内记录的 runtime 必须与 Core 合同逐字段相等；未记录时计数并写进说明，不当作通过证据。
def check_runtime(row: dict[str, object], core: dict[str, object] | None, report: Report) -> None:
    runtime = row.get("runtime")
    report.count("runtime-unrecorded" if runtime is None else "runtime-checked")
    if runtime is not None and core is not None and runtime != core:
        report.fail(str(row["name"]), "runtime", f"runtime {json.dumps(runtime)} 与 Core {json.dumps(core)} 不符")


def run(distribution: Path, selected: list[str] | None, python_tag: str | None) -> Report:
    # 1. 读取发行记录并确定检查集合 2. 宿主事实只算一次 3. 逐插件四项检查
    report = Report()
    rows = cast(list[dict[str, str]], json.loads((distribution / "distribution.json").read_bytes())["plugins"])
    by_name = {row["name"]: row for row in rows}
    unknown = sorted(set(selected or ()) - by_name.keys())
    if unknown:
        raise SystemExit(f"发行版没有这些插件: {', '.join(unknown)}")
    tokens = host_tokens(distribution, report)
    import_map = host_import_map(distribution, report)
    core = core_runtime(distribution, python_tag, report)
    exports: dict[str, set[str]] = {}
    # 清理失败不能让检查结论失真：残留的临时目录交给 /tmp 回收，不抛清理异常。
    with tempfile.TemporaryDirectory(prefix="akashic-compat-gate-", ignore_cleanup_errors=True) as directory:
        records = BundleRecords(distribution, Path(directory))
        for row in (by_name[name] for name in (selected or sorted(by_name))):
            report.count("plugins")
            fetched = records.fetch(row, report)
            assets = check_assets(row, distribution / "sources" / row["name"], records, report) if fetched else []
            check_css_vars(row["name"], assets, tokens, report)
            check_sdk_imports(row["name"], assets, import_map, exports, report)
            check_runtime(cast(dict[str, object], row), core, report)
    if report.counts.get("runtime-unrecorded"):
        report.notes.append(f"{report.counts['runtime-unrecorded']} 个插件行没有 runtime 字段，"
                            f"检查 4 只确定了 Core 合同 {json.dumps(core)}")
    report.counts["host-tokens"] = len(tokens)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--distribution", type=Path, required=True)
    _ = parser.add_argument("--plugins", help="逗号分隔的变更插件；默认全部")
    _ = parser.add_argument("--python-tag", help="覆盖从 requirements.lock 推出的 Core python_tag")
    _ = parser.add_argument("--json", type=Path, dest="json_path")
    args = parser.parse_args()
    started = time.perf_counter()
    plugins = cast(str | None, args.plugins)
    selected = [item.strip() for item in plugins.split(",") if item.strip()] if plugins else None
    report = run(cast(Path, args.distribution).resolve(), selected, cast(str | None, args.python_tag))
    for failure in report.failures:
        print(f"FAIL [{failure.check}] {failure.plugin}: {failure.detail}")
    for note in report.notes:
        print(f"NOTE {note}")
    summary = ", ".join(f"{key}={value}" for key, value in sorted(report.counts.items()))
    verdict = "FAIL" if report.failures else "PASS"
    print(f"{verdict}: {len(report.failures)} 处失败；{summary}；{time.perf_counter() - started:.2f}s")
    if args.json_path is not None:
        payload: dict[str, object] = {
            "pass": not report.failures,
            "failures": [{"plugin": item.plugin, "check": item.check, "detail": item.detail} for item in report.failures],
            "notes": report.notes,
        }
        _ = cast(Path, args.json_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sys.exit(1 if report.failures else 0)


if __name__ == "__main__":
    main()
