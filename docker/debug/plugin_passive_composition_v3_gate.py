from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.core.response_parser import ResponseMetadata  # noqa: E402
from agent.lifecycle.composition import (  # noqa: E402
    AFTER_REASONING_CLEANUP_EVENT,
    AFTER_REASONING_PREPROCESS_EVENT,
    PROMPT_RENDER_EVENT,
)
from agent.lifecycle.types import AfterReasoningCtx, PromptRenderCtx  # noqa: E402
from agent.plugins.dashboard_host import (  # noqa: E402
    DashboardBinding,
    PluginDashboardHost,
)
from agent.plugins.manager import PluginManager  # noqa: E402
from agent.plugins.snapshot import (  # noqa: E402
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)
from bus.event_bus import EventBus  # noqa: E402

DEFAULT_LOCK = ROOT / "docker" / "debug" / "plugin-passive-composition-v3.lock.json"
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
GATE_VERSION = 1
PROTOCOL_SOURCE_REPOSITORY = "https://github.com/kachofugetsu09/akashic-agent.git"
PROTOCOL_SOURCE_COMMIT = "dbbd82b56fe39cc37d3c866048605bf82e3755b0"
PROTOCOL_SOURCE_PATHS = (
    "agent/lifecycle/composition.py",
    "agent/lifecycle/types.py",
    "agent/plugin_composition/context.py",
    "agent/plugins/dashboard_host.py",
    "agent/plugins/manager.py",
)
EXPECTED_PLUGIN_IDS = ("citation", "meme")
EXPECTED_LISTENERS = (
    "serial:turn.prompt_render:citation",
    "serial:turn.prompt_render:meme",
    "serial:turn.after_reasoning.preprocess:citation",
    "serial:turn.after_reasoning.preprocess:meme",
    "serial:turn.after_reasoning.cleanup:citation",
)
SCENARIO_PROFILE = "citation-meme-passive-v3-v1"
SCENARIO_INPUT = "答复正文\n§cited:[mem_1]§ <meme:shy>"


@dataclass(frozen=True)
class SourceLock:
    id: str
    repository: str
    requested_ref: str
    resolved_sha: str
    change_source_pr_head: str


@dataclass(frozen=True)
class SourceEvidence:
    id: str
    repository: str
    requested_ref: str
    resolved_sha: str
    change_source_pr_head: str
    tree: str


@dataclass(frozen=True)
class GateLock:
    contract: SourceLock
    plugins: tuple[SourceLock, ...]


@dataclass(frozen=True)
class ContractEvidence:
    contract: str
    plugin_ids: tuple[str, ...]
    source_sha256: tuple[str, ...]
    plugin_classes: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class ScenarioEvidence:
    prompt_sections: tuple[str, ...]
    final_reply: str
    cited_memory_ids: tuple[str, ...]
    media: tuple[str, ...]
    meme_tag: str | None


@dataclass(frozen=True)
class DashboardEvidence:
    plugin_id: str
    validation: bool
    categories: tuple[str, ...]
    runtime_workspace: str
    runtime_data_root: str


@dataclass(frozen=True)
class CleanupEvidence:
    listeners: tuple[str, ...]
    effects: tuple[str, ...]
    services: tuple[str, ...]
    dashboard_bindings: int
    dashboard_module_loaded: bool


def _load_lock(path: Path) -> GateLock:
    """Strictly load the immutable protocol and plugin source set."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "contract",
        "plugins",
    }:
        raise ValueError("passive v3 组合锁根结构无效")
    if raw["schema_version"] != 1:
        raise ValueError(f"不支持的 passive v3 组合锁版本: {raw['schema_version']}")
    contract = _parse_source_lock(raw["contract"])
    raw_plugins = raw["plugins"]
    if not isinstance(raw_plugins, list):
        raise ValueError("passive v3 组合锁 plugins 必须是列表")
    plugins = tuple(_parse_source_lock(item) for item in raw_plugins)
    if tuple(item.id for item in plugins) != EXPECTED_PLUGIN_IDS:
        raise ValueError("passive v3 组合锁的插件集合或顺序错误")
    if contract.id != "plugin_contracts":
        raise ValueError("passive v3 组合锁缺少 plugin_contracts owner")
    return GateLock(contract=contract, plugins=plugins)


def _parse_source_lock(raw: object) -> SourceLock:
    expected = {
        "id",
        "repository",
        "requested_ref",
        "resolved_sha",
        "change_source_pr_head",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError(f"passive v3 组合锁字段无效: {raw}")
    item = cast(dict[str, object], raw)
    values = {name: _required_string(item, name) for name in expected}
    repository = values["repository"]
    if not repository.startswith("https://github.com/") or not repository.endswith(
        ".git"
    ):
        raise ValueError(f"源仓库必须是公开 GitHub HTTPS Git 地址: {repository}")
    revisions = ("requested_ref", "resolved_sha", "change_source_pr_head")
    for field in revisions:
        if COMMIT_PATTERN.fullmatch(values[field]) is None:
            raise ValueError(f"{field} 必须是完整 SHA: {values[field]}")
    if len({values[field] for field in revisions}) != 1:
        raise ValueError(f"三个 revision 必须固定到同一提交: {values['id']}")
    return SourceLock(**values)


def _required_string(item: dict[str, object], name: str) -> str:
    value = item[name]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"passive v3 组合锁字段必须是非空字符串: {name}")
    return value


def _checkout_locked_source(lock: SourceLock, checkout: Path) -> SourceEvidence:
    """Fetch one declared public object into a fresh, detached repository."""

    _run(("git", "init", "--quiet", str(checkout)), cwd=ROOT)
    _run(("git", "remote", "add", "origin", lock.repository), cwd=checkout)
    _run(
        ("git", "fetch", "--quiet", "--depth=1", "origin", lock.resolved_sha),
        cwd=checkout,
    )
    _run(("git", "checkout", "--quiet", "--detach", "FETCH_HEAD"), cwd=checkout)
    if _git_output(checkout, "rev-parse", "HEAD") != lock.resolved_sha:
        raise RuntimeError(f"检出提交与锁不一致: {lock.id}")
    if _git_output(checkout, "status", "--porcelain"):
        raise RuntimeError(f"检出后工作树不干净: {lock.id}")
    return SourceEvidence(
        id=lock.id,
        repository=lock.repository,
        requested_ref=lock.requested_ref,
        resolved_sha=lock.resolved_sha,
        change_source_pr_head=lock.change_source_pr_head,
        tree=_git_output(checkout, "rev-parse", "HEAD^{tree}"),
    )


def _verify_static_contract(
    contract_checkout: Path,
    plugin_paths: tuple[Path, ...],
) -> ContractEvidence:
    """Run the exact public contract checker and require pure v3 entrypoints."""

    command = (
        sys.executable,
        "-m",
        "akashic_plugin_contracts",
        "check",
        *(str(path) for path in plugin_paths),
    )
    completed = _run(command, cwd=contract_checkout)
    raw = json.loads(completed.stdout)
    reports = raw.get("reports")
    if raw.get("passed") is not True or raw.get("contract") != "akashic-plugin-api-v3":
        raise RuntimeError(f"插件静态合同失败: {raw}")
    if not isinstance(reports, list) or len(reports) != len(plugin_paths):
        raise RuntimeError(f"插件静态合同报告数量错误: {raw}")
    plugin_classes = tuple(tuple(item["plugin_classes"]) for item in reports)
    if any(plugin_classes):
        raise RuntimeError(f"纯 v3 Gate 发现 v2 Plugin 类: {plugin_classes}")
    return ContractEvidence(
        contract=str(raw["contract"]),
        plugin_ids=EXPECTED_PLUGIN_IDS,
        source_sha256=tuple(str(item["sha256"]) for item in reports),
        plugin_classes=plugin_classes,
    )


async def _verify_composition(providers: Path, sandbox: Path) -> dict[str, object]:
    """Load, exercise, and dispose one real stable Citation + Meme snapshot."""

    # 1. Establish a formal workspace fixture before plugins are loaded.
    workspace = sandbox / "workspace"
    image = _write_meme_fixture(workspace)
    meme_tree_before = _tree_digest(workspace / "memes")
    manager = PluginManager(
        plugin_dirs=[providers],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=workspace,
        installed_cache_root=sandbox / "plugin-home" / "cache",
    )
    root = None
    dashboard_host = PluginDashboardHost(
        core_routes=(),
    )
    dashboard_module = ""
    evidence: dict[str, object] | None = None
    try:
        # 2. Use the published stable snapshot as the only execution source.
        await manager.load_all()
        snapshot = manager.current_snapshot
        if snapshot is None or snapshot.composition_root is None:
            raise RuntimeError("正式 snapshot 缺少 v3 CompositionRoot")
        root = snapshot.composition_root
        topology = root.topology_view()
        _assert_topology(topology)
        _assert_skill(snapshot)
        dashboard_host.prepare_snapshot(snapshot)
        dashboard = _assert_dashboard(snapshot, workspace)
        dashboard_module = cast(
            DashboardBinding, snapshot.dashboard_bindings[0]
        ).module_name

        lease = manager.snapshot_store.lease()
        token = bind_runtime_snapshot(lease)
        try:
            scenario = await _run_passive_scenario(root, image)
        finally:
            reset_runtime_snapshot(token)
            await lease.release()

        # 3. Plugin apply and passive observation must not mutate the product assets.
        meme_tree_after = _tree_digest(workspace / "memes")
        if meme_tree_after != meme_tree_before:
            raise RuntimeError("Citation + Meme 被动链路改写了正式 memes 资产")
        evidence = {
            "topology": {
                "identity": topology.identity,
                "revision": topology.composition_revision,
                "listeners": list(topology.listeners),
                "services": list(topology.services),
                "fibers": [asdict(item) for item in topology.fibers],
            },
            "skill_names": sorted(snapshot.plugin_skill_index.records),
            "dashboard": asdict(dashboard),
            "scenario": asdict(scenario),
            "workspace": {
                "memes_before_sha256": meme_tree_before,
                "memes_after_sha256": meme_tree_after,
                "plugin_data_entries": _relative_entries(workspace / "plugin-data"),
            },
        }
    finally:
        await manager.terminate_all()

    if root is None or evidence is None:
        raise AssertionError("passive v3 Gate 成功路径没有保留正式 Root 证据")
    cleanup = CleanupEvidence(
        listeners=root.topology_view().listeners,
        effects=root.receipt().effects,
        services=root.receipt().services,
        dashboard_bindings=len(dashboard_host._bindings),
        dashboard_module_loaded=dashboard_module in sys.modules,
    )
    if (
        cleanup.listeners
        or cleanup.effects
        or cleanup.services
        or cleanup.dashboard_bindings
        or cleanup.dashboard_module_loaded
    ):
        raise RuntimeError(f"正式插件资源终止后仍有残留: {cleanup}")
    evidence["cleanup"] = asdict(cleanup)
    return evidence


def _write_meme_fixture(workspace: Path) -> Path:
    memes = workspace / "memes"
    (memes / "shy").mkdir(parents=True)
    image = memes / "shy" / "001.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    (memes / "manifest.json").write_text(
        json.dumps(
            {"categories": {"shy": {"desc": "害羞", "enabled": True}}},
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return image


def _assert_topology(topology: object) -> None:
    listeners = tuple(getattr(topology, "listeners"))
    services = tuple(getattr(topology, "services"))
    fibers = {item.name: item for item in getattr(topology, "fibers")}
    if listeners != EXPECTED_LISTENERS:
        raise RuntimeError(f"passive v3 listener 顺序错误: {listeners}")
    if services != ("citation.protocol", "core.commands"):
        raise RuntimeError(f"passive v3 service 集合错误: {services}")
    if set(fibers) != set(EXPECTED_PLUGIN_IDS):
        raise RuntimeError(f"passive v3 Fiber 集合错误: {tuple(fibers)}")
    if fibers["citation"].dependencies:
        raise RuntimeError("Citation 不应依赖 Meme capability")
    if fibers["meme"].dependencies != ("citation.protocol",):
        raise RuntimeError(f"Meme capability 依赖错误: {fibers['meme'].dependencies}")


def _assert_skill(snapshot: object) -> None:
    index = getattr(snapshot, "plugin_skill_index")
    if index is None or set(index.records) != {"meme-manage"}:
        raise RuntimeError(
            f"Meme Skill 静态投影错误: {None if index is None else index.records}"
        )


def _assert_dashboard(snapshot: object, workspace: Path) -> DashboardEvidence:
    bindings = getattr(snapshot, "dashboard_bindings")
    if len(bindings) != 1 or not isinstance(bindings[0], DashboardBinding):
        raise RuntimeError(f"Meme Dashboard binding 数量错误: {bindings}")
    binding = bindings[0]
    if binding.plugin_id != "meme" or binding.validation:
        raise RuntimeError(f"Meme 正式 Dashboard 身份错误: {binding}")
    route = next(
        item for item in binding.routes if item.path == "/api/dashboard/meme/categories"
    )
    result = route.endpoint()
    if inspect.isawaitable(result):
        close = getattr(result, "close", None)
        if callable(close):
            _ = close()
        raise RuntimeError("Meme Dashboard categories route 意外返回 awaitable")
    categories = cast(Mapping[str, object], result)["categories"]
    tags = tuple(str(item["tag"]) for item in cast(list[dict[str, object]], categories))
    if tags != ("shy",):
        raise RuntimeError(f"Meme Dashboard 未读取正式 workspace fixture: {tags}")
    if binding.runtime_workspace != workspace.resolve():
        raise RuntimeError(
            f"Meme Dashboard workspace 错误: {binding.runtime_workspace}"
        )
    return DashboardEvidence(
        plugin_id=binding.plugin_id,
        validation=binding.validation,
        categories=tags,
        runtime_workspace=str(binding.runtime_workspace),
        runtime_data_root=str(binding.runtime_data_root),
    )


async def _run_passive_scenario(root: object, image: Path) -> ScenarioEvidence:
    prompt = PromptRenderCtx(
        session_key="webui:gate",
        channel="webui",
        chat_id="gate",
        content="你好",
        media=None,
        timestamp=datetime.now(UTC),
        history=[],
        skill_names=None,
        disabled_sections=set(),
        turn_injection_prompt="",
    )
    context = getattr(root, "context")
    _ = await context.serial(PROMPT_RENDER_EVENT, prompt)
    answer = AfterReasoningCtx(
        session_key="webui:gate",
        channel="webui",
        chat_id="gate",
        tools_used=(),
        thinking=None,
        response_metadata=ResponseMetadata(raw_text=SCENARIO_INPUT),
        streamed=False,
        tool_chain=(),
        context_retry={},
        reply=SCENARIO_INPUT,
    )
    _ = await context.serial(AFTER_REASONING_PREPROCESS_EVENT, answer)
    _ = await context.serial(AFTER_REASONING_CLEANUP_EVENT, answer)
    evidence = ScenarioEvidence(
        prompt_sections=tuple(item.name for item in prompt.system_sections_bottom),
        final_reply=answer.reply,
        cited_memory_ids=tuple(answer.persist_assistant_metadata["cited_memory_ids"]),
        media=tuple(answer.media),
        meme_tag=answer.meme_tag,
    )
    expected = ScenarioEvidence(
        prompt_sections=("citation_protocol", "memes"),
        final_reply="答复正文",
        cited_memory_ids=("mem_1",),
        media=(str(image),),
        meme_tag="shy",
    )
    if evidence != expected:
        raise RuntimeError(f"Citation + Meme 被动链路行为错误: {evidence}")
    return evidence


def _tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        relative = item.relative_to(path).as_posix()
        kind = "d" if item.is_dir() else "f"
        digest.update(f"{kind}:{relative}\0".encode())
        if item.is_file():
            digest.update(item.read_bytes())
    return digest.hexdigest()


def _relative_entries(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted(item.relative_to(path).as_posix() for item in path.rglob("*"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scenario_catalog_sha256() -> str:
    payload = {
        "profile": SCENARIO_PROFILE,
        "input": SCENARIO_INPUT,
        "plugins": EXPECTED_PLUGIN_IDS,
        "listeners": EXPECTED_LISTENERS,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _protocol_source_evidence() -> dict[str, object]:
    files: list[dict[str, str]] = []
    for path in PROTOCOL_SOURCE_PATHS:
        blob = _git_output(ROOT, "rev-parse", f"{PROTOCOL_SOURCE_COMMIT}:{path}")
        content = subprocess.run(
            ("git", "cat-file", "blob", blob),
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        files.append(
            {
                "path": path,
                "git_blob": blob,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return {
        "repository": PROTOCOL_SOURCE_REPOSITORY,
        "commit": PROTOCOL_SOURCE_COMMIT,
        "files": files,
    }


def _git_output(cwd: Path, *args: str) -> str:
    return _run(("git", *args), cwd=cwd).stdout.strip()


def _run(command: tuple[str, ...], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
