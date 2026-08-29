#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
import tomllib
import uuid
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from websockets.sync.client import connect as connect_websocket

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from agent.plugins.artifacts import (  # noqa: E402
    ArtifactPointer,
    read_pointers,
    write_pointers,
)
from agent.plugins.manifest import write_plugin_manifest  # noqa: E402
from docker.debug import (
    plugin_passive_composition_v3_gate as composition_gate,
)  # noqa: E402
from docker.debug import plugin_v3_fleet_gate as fleet_gate  # noqa: E402
from docker.debug import programmatic_control_probe as control_probe  # noqa: E402

DEFAULT_REPORT = ROOT / "docker/debug/reports/plugin-passive-webui-v3" / "gate.json"
COMPOSE_FILE = ROOT / "docker/debug/docker-compose.control-gate.yml"
GATE_VERSION = 2
SCENARIO_PROFILE = "citation-meme-webui-v3-v1"
MODEL_RESPONSE = "答复正文\n§cited:[mem_1]§ <meme:shy>"
USER_INPUT = "请给我一条带引用和表情的回复"
MARKETPLACE = "webui"
EXPECTED_PLUGIN_IDS = (
    "citation@webui",
    "meme@webui",
    "models",
    "openai-compatible",
)
EXPECTED_SOURCE_IDS = ("citation", "meme")
READINESS_TIMEOUT_S = 60.0
SCENARIO_TIMEOUT_S = 30.0


class GateFailure(RuntimeError):
    pass


def main() -> int:
    """Run the host controller or the in-container public WebUI scenario."""

    parser = argparse.ArgumentParser(description="验证纯 v3 Citation/Meme WebUI E2E")
    parser.add_argument("--inside-container", action="store_true")
    parser.add_argument("--require-clean-core", action="store_true")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.inside_container:
        _write_json(args.report, _run_inside())
        return 0
    return _run_host(args.report.resolve(), require_clean=args.require_clean_core)


def _run_host(report_path: Path, *, require_clean: bool) -> int:
    """Own exact checkouts, the disposable runtime, evidence, and cleanup."""

    # 1. Freeze the source identities before creating ignored reports.
    core_status = _git_output(ROOT, "status", "--porcelain").splitlines()
    if require_clean and core_status:
        raise GateFailure(f"核心工作树不干净: {core_status}")
    core_head = _git_output(ROOT, "rev-parse", "HEAD")
    core_tree = _git_output(ROOT, "rev-parse", "HEAD^{tree}")
    source_before = control_probe._repository_digest(
        ROOT
    )  # pyright: ignore[reportPrivateUsage]
    lock = _load_final_sources()
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    sandbox = Path(tempfile.mkdtemp(prefix="akashic-passive-webui-v3-"))
    project = f"akashic-passive-webui-{run_id.lower()}"
    env = {
        **os.environ,
        "AKASHIC_CONTROL_SANDBOX": str(sandbox),
        "UID": str(os.getuid()),
        "GID": str(os.getgid()),
    }
    compose = ["docker", "compose", "-p", project, "-f", str(COMPOSE_FILE)]
    cleanup: dict[str, object] = {"residuals": ["not_started"]}
    runtime: dict[str, object] | None = None
    controller_error = ""
    source_evidence: list[dict[str, object]] = []
    contract_report: object = None
    config_sha256 = ""
    meme_before_sha256 = ""
    try:
        # 2. Build a private app/workspace and install exact immutable artifacts.
        providers = sandbox / "providers"
        providers.mkdir()
        contract_checkout = sandbox / "contract"
        contract_source = composition_gate._checkout_locked_source(  # pyright: ignore[reportPrivateUsage]
            lock.contract,
            contract_checkout,
        )
        source_evidence.append({"kind": "contract", **contract_source.__dict__})
        plugin_checkouts: dict[str, Path] = {}
        for item in lock.plugins:
            checkout = providers / item.id
            evidence = composition_gate._checkout_locked_source(  # pyright: ignore[reportPrivateUsage]
                item,
                checkout,
            )
            source_evidence.append({"kind": "plugin", **evidence.__dict__})
            plugin_checkouts[item.id] = checkout
        contract_report = composition_gate._verify_static_contract(  # pyright: ignore[reportPrivateUsage]
            contract_checkout,
            tuple(plugin_checkouts[item.id] / "plugin.py" for item in lock.plugins),
        ).__dict__
        control_probe._prepare_host_sandbox(  # pyright: ignore[reportPrivateUsage]
            sandbox,
            ROOT,
            max_iterations=2,
        )
        _restrict_builtin_plugins(sandbox / "app/plugins")
        _write_runtime_config(sandbox / "config.toml")
        installed = _install_exact_plugins(sandbox, lock.plugins, plugin_checkouts)
        image = _write_meme_fixture(sandbox / "workspace")
        config_sha256 = _sha256(sandbox / "config.toml")
        meme_before_sha256 = _tree_sha256(image.parents[1])

        # 3. Exercise one real supervised runtime through its public WebUI entry.
        build = _run([*compose, "build", "model-gate"], cwd=ROOT, env=env)
        if build.returncode != 0:
            raise GateFailure(f"Docker image build 失败: {build.returncode}")
        up = _run(
            [*compose, "up", "-d", "model-gate", "akashic-control-gate"],
            cwd=ROOT,
            env=env,
        )
        if up.returncode != 0:
            raise GateFailure(f"Docker runtime 启动失败: {up.returncode}")
        inside_report = Path("/sandbox/reports/plugin-passive-webui-v3.json")
        inside = _run(
            [
                *compose,
                "exec",
                "-T",
                "akashic-control-gate",
                "python",
                "/app/docker/debug/plugin_passive_webui_v3_e2e.py",
                "--inside-container",
                "--report",
                str(inside_report),
            ],
            cwd=ROOT,
            env=env,
        )
        host_inside_report = sandbox / "reports/plugin-passive-webui-v3.json"
        if not host_inside_report.is_file():
            raise GateFailure(
                "inside WebUI Gate 没有生成报告: "
                f"returncode={inside.returncode} output={inside.stdout[-4000:]}"
            )
        runtime = cast(dict[str, object], json.loads(host_inside_report.read_text()))
        if inside.returncode != 0 or runtime.get("status") != "passed":
            raise GateFailure(f"inside WebUI Gate 失败: {inside.returncode} {runtime}")
        runtime["immutability_after_runtime"] = _verify_runtime_immutability(
            sandbox,
            installed,
            meme_before_sha256,
            phase="after_runtime",
        )
        runtime["installed_artifacts"] = installed

        # 4. Stop through the supervised lifecycle before removing the project.
        stopped = _run(
            [*compose, "stop", "-t", "15", "akashic-control-gate"],
            cwd=ROOT,
            env=env,
        )
        if stopped.returncode != 0:
            raise GateFailure(f"Gateway 优雅停止失败: {stopped.returncode}")
        runtime["immutability_after_stop"] = _verify_runtime_immutability(
            sandbox,
            installed,
            meme_before_sha256,
            phase="after_stop",
        )
    except BaseException as error:
        controller_error = f"{type(error).__name__}: {error}"
    finally:
        logs = _run([*compose, "logs", "--no-color"], cwd=ROOT, env=env)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        (report_path.parent / "runtime.log").write_text(logs.stdout, encoding="utf-8")
        down = _run(
            [*compose, "down", "--volumes", "--remove-orphans"],
            cwd=ROOT,
            env=env,
        )
        cleanup = _cleanup_evidence(compose, project, env, down.returncode)
        source_after = control_probe._repository_digest(
            ROOT
        )  # pyright: ignore[reportPrivateUsage]
        cleanup["source_unchanged"] = source_after == source_before
        shutil.rmtree(sandbox)
        cleanup["sandbox_removed"] = not sandbox.exists()

    passed = (
        not controller_error
        and runtime is not None
        and runtime.get("status") == "passed"
        and cleanup.get("residuals") == []
        and cleanup.get("source_unchanged") is True
        and cleanup.get("sandbox_removed") is True
    )
    report = {
        "status": "passed" if passed else "failed",
        "gate_version": GATE_VERSION,
        "run_id": run_id,
        "scenario_profile": SCENARIO_PROFILE,
        "scenario_sha256": _scenario_sha256(),
        "core": {"head": core_head, "tree": core_tree, "dirty_status": core_status},
        "lock": str(fleet_gate.DEFAULT_LOCK.relative_to(ROOT)),
        "lock_sha256": _sha256(fleet_gate.DEFAULT_LOCK),
        "contract_lock": str(composition_gate.DEFAULT_LOCK.relative_to(ROOT)),
        "contract_lock_sha256": _sha256(composition_gate.DEFAULT_LOCK),
        "config_sha256": config_sha256,
        "sources": source_evidence,
        "contract_report": contract_report,
        "meme_before_sha256": meme_before_sha256,
        "runtime": runtime,
        "cleanup": cleanup,
        "controller_error": controller_error,
    }
    _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 1


def _load_final_sources() -> composition_gate.GateLock:
    """把 WebUI 场景绑定到最终 fleet identity 与 contract owner。"""

    # 1. Passive lock 只拥有固定的公开 contract checker。
    passive = composition_gate._load_lock(  # pyright: ignore[reportPrivateUsage]
        composition_gate.DEFAULT_LOCK
    )

    # 2. Citation 与 Meme 统一从最终 fleet lock 取得。
    fleet = {
        item.id: item
        for item in fleet_gate._load_lock(  # pyright: ignore[reportPrivateUsage]
            fleet_gate.DEFAULT_LOCK
        )
    }
    missing = sorted(set[str](EXPECTED_SOURCE_IDS) - set(fleet))
    if missing:
        raise GateFailure(f"最终 fleet lock 缺少 WebUI source: {missing}")
    plugins = tuple(
        composition_gate.SourceLock(
            id=fleet[plugin_id].id,
            repository=fleet[plugin_id].repository,
            requested_ref=fleet[plugin_id].requested_ref,
            resolved_sha=fleet[plugin_id].resolved_sha,
            change_source_pr_head=fleet[plugin_id].change_source_pr_head,
        )
        for plugin_id in EXPECTED_SOURCE_IDS
    )
    return composition_gate.GateLock(contract=passive.contract, plugins=plugins)


def _run_inside() -> dict[str, object]:
    """Drive the public WebUI and verify its durable and plugin projections."""

    # 1. Wait for the one public shell and assert the channel configuration.
    base_url = "http://127.0.0.1:2236"
    shell_state = _wait_json(f"{base_url}/api/shell/state")
    health = _http_json("GET", f"{base_url}/api/chat/health")
    _assert_ready(health, "/api/chat/health")
    control_probe._configure_model_gate()  # pyright: ignore[reportPrivateUsage]
    config = tomllib.loads(Path("/sandbox/config.toml").read_text(encoding="utf-8"))
    _assert_webui_only(config)
    capabilities = cast(
        dict[str, object],
        _http_json("GET", f"{base_url}/api/chat/runtime/capabilities"),
    )
    _assert_capabilities(capabilities)

    # 2. Queue one deterministic provider response and send one public WS turn.
    model_url = "http://model-gate:8090"
    _ = _http_json(
        "PUT",
        f"{model_url}/control/script",
        {"mode": "complete", "content": MODEL_RESPONSE},
    )
    frames: list[dict[str, object]] = []
    with connect_websocket(
        "ws://127.0.0.1:2236/ws",
        open_timeout=READINESS_TIMEOUT_S,
    ) as websocket:
        websocket.send(json.dumps({"type": "session.create", "request_id": "create"}))
        created = cast(dict[str, object], json.loads(websocket.recv(timeout=10)))
        if created.get("type") != "session.created":
            raise GateFailure(f"WebUI session.create 响应错误: {created}")
        session_id = str(created.get("session_id") or "")
        if not session_id.startswith("akashic:"):
            raise GateFailure(f"WebUI session identity 错误: {session_id}")
        websocket.send(
            json.dumps(
                {
                    "type": "message.send",
                    "request_id": "message",
                    "session_id": session_id,
                    "text": USER_INPUT,
                    "media": [],
                },
                ensure_ascii=False,
            )
        )
        final = _receive_final(websocket, frames, session_id)

    # 3. Read the same committed facts through public HTTP surfaces.
    encoded_session = quote(session_id, safe="")
    messages = cast(
        dict[str, object],
        _http_json(
            "GET",
            f"{base_url}/api/chat/sessions/{encoded_session}/messages?"
            + urlencode(
                {"page": 1, "page_size": 20, "sort_by": "seq", "sort_order": "asc"}
            ),
        ),
    )
    persisted = _assert_messages(messages, session_id)
    media = cast(list[object], final.get("media"))
    descriptor = _assert_artifact_descriptor(
        media[0],
        expected_filename="001.png",
    )
    media_bytes = _http_bytes(f"{base_url}{descriptor['url']}")
    if media_bytes != b"\x89PNG\r\n\x1a\n":
        raise GateFailure("WebUI media API 未返回 exact fixture bytes")
    dashboard = cast(
        dict[str, object],
        _http_json("GET", f"{base_url}/api/dashboard/meme/categories"),
    )
    _assert_dashboard(dashboard)

    # 4. Prove prompt ordering and the append-only database integrity.
    requests = cast(
        dict[str, object],
        _http_json("GET", f"{model_url}/control/requests"),
    )
    prompt = _assert_model_request(requests)
    with sqlite3.connect("/sandbox/workspace/sessions.db") as connection:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    if integrity != "ok":
        raise GateFailure(f"SessionDB integrity_check 失败: {integrity}")
    return {
        "status": "passed",
        "shell_state": shell_state,
        "health": health,
        "capabilities": capabilities,
        "session_id": session_id,
        "frames": frames,
        "final": final,
        "messages": persisted,
        "dashboard": dashboard,
        "model_request": prompt,
        "database_integrity": integrity,
    }


def _restrict_builtin_plugins(root: Path) -> None:
    """Keep only ordinary model plugins required to drive the WebUI turn."""

    for child in root.iterdir():
        if child.name in {"__init__.py", "models", "openai_compatible"}:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _write_runtime_config(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    content += "\n[mobile_realtime]\nenabled = false\n"
    path.write_text(content, encoding="utf-8")


def _install_exact_plugins(
    sandbox: Path,
    locks: tuple[composition_gate.SourceLock, ...],
    checkouts: dict[str, Path],
) -> list[dict[str, object]]:
    """Publish exact checkouts in the installed stable artifact layout."""

    cache = sandbox / "home/.akashic-plugin/cache"
    installed: list[dict[str, object]] = []
    manifest: dict[str, bool] = {}
    for item in locks:
        plugin_base = cache / MARKETPLACE / item.id
        relative = f".artifacts/{item.resolved_sha}"
        artifact = plugin_base / relative
        shutil.copytree(
            checkouts[item.id],
            artifact,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
        )
        pointer = ArtifactPointer(relative)
        _ = write_pointers(plugin_base, stable=pointer, latest=pointer)
        plugin_id = f"{item.id}@{MARKETPLACE}"
        manifest[plugin_id] = True
        installed.append(
            {
                "plugin_id": plugin_id,
                "resolved_sha": item.resolved_sha,
                "pointer": relative,
                "tree": _git_output(checkouts[item.id], "rev-parse", "HEAD^{tree}"),
                "artifact_sha256_before": _tree_sha256(artifact),
                "pointers_before": _pointer_paths(plugin_base),
                "artifact_inventory_before": _artifact_inventory(plugin_base),
            }
        )
    _ = write_plugin_manifest(manifest, plugins_home=sandbox / "home/.akashic-plugin")
    return installed


def _write_meme_fixture(workspace: Path) -> Path:
    memes = workspace / "memes"
    (memes / "shy").mkdir(parents=True)
    image = memes / "shy/001.png"
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


def _assert_webui_only(config: dict[str, object]) -> None:
    channels = cast(dict[str, dict[str, object]], config.get("channels"))
    if channels["chat"] != {"enabled": True}:
        raise GateFailure(f"WebUI channel 配置漂移: {channels['chat']}")
    if channels["telegram"].get("enabled") is not False:
        raise GateFailure("Telegram 必须关闭")
    if channels["qq"].get("enabled") is not False:
        raise GateFailure("QQ 必须关闭")
    mobile = cast(dict[str, object], config.get("mobile_realtime"))
    if mobile.get("enabled") is not False:
        raise GateFailure("Mobile 必须关闭")
    if "proactive" in config:
        raise GateFailure("retired proactive 配置段必须不存在")


def _assert_capabilities(payload: dict[str, object]) -> None:
    plugins = cast(list[dict[str, object]], payload.get("plugins"))
    plugin_ids = tuple(str(item.get("id")) for item in plugins)
    if plugin_ids != EXPECTED_PLUGIN_IDS:
        raise GateFailure(f"active plugin 集合错误: {plugin_ids}")
    skills = cast(list[dict[str, object]], payload.get("skills"))
    plugin_skills = [item for item in skills if item.get("source") != "builtin"]
    if [str(item.get("name")) for item in plugin_skills] != ["meme-manage"]:
        raise GateFailure(f"插件 Skill 投影错误: {plugin_skills}")


def _receive_final(
    websocket: Any,
    frames: list[dict[str, object]],
    expected_session_id: str,
) -> dict[str, object]:
    deadline = time.monotonic() + SCENARIO_TIMEOUT_S
    while time.monotonic() < deadline:
        frame = cast(
            dict[str, object],
            json.loads(websocket.recv(timeout=deadline - time.monotonic())),
        )
        frames.append(frame)
        if frame.get("type") != "message.final":
            continue
        if frame.get("session_id") != expected_session_id:
            raise GateFailure(f"WebUI final session 错误: {frame}")
        if frame.get("content") != "答复正文":
            raise GateFailure(f"WebUI final 正文错误: {frame}")
        media = frame.get("media")
        if not isinstance(media, list) or len(media) != 1:
            raise GateFailure(f"WebUI final media 错误: {frame}")
        _assert_artifact_descriptor(media[0], expected_filename="001.png")
        return frame
    raise GateFailure("WebUI 未在 deadline 内返回 message.final")


def _assert_messages(
    payload: dict[str, object], session_id: str
) -> list[dict[str, object]]:
    items = cast(list[dict[str, object]], payload.get("items"))
    if payload.get("total") != 2 or len(items) != 2:
        raise GateFailure(f"WebUI 持久消息数量错误: {payload}")
    user, assistant = items
    if user.get("session_key") != session_id or user.get("role") != "user":
        raise GateFailure(f"持久 user 消息错误: {user}")
    if user.get("content") != USER_INPUT:
        raise GateFailure(f"持久 user 正文错误: {user}")
    if assistant.get("session_key") != session_id:
        raise GateFailure(f"持久 assistant session 错误: {assistant}")
    if assistant.get("role") != "assistant" or assistant.get("content") != "答复正文":
        raise GateFailure(f"持久 assistant 正文错误: {assistant}")
    if assistant.get("cited_memory_ids") != ["mem_1"]:
        raise GateFailure(f"持久 citation metadata 错误: {assistant}")
    attachment_ids = assistant.get("attachment_ids")
    attachments = assistant.get("attachments")
    if (
        not isinstance(attachment_ids, list)
        or len(attachment_ids) != 1
        or not isinstance(attachment_ids[0], str)
        or not isinstance(attachments, list)
        or len(attachments) != 1
    ):
        raise GateFailure(f"持久 Meme attachment 错误: {assistant}")
    descriptor = _assert_artifact_descriptor(
        attachments[0],
        expected_filename="001.png",
    )
    if descriptor["artifact_id"] != attachment_ids[0]:
        raise GateFailure(f"持久 Meme attachment identity 漂移: {assistant}")
    return items


def _assert_artifact_descriptor(
    value: object,
    *,
    expected_filename: str,
) -> dict[str, object]:
    """验证 Web 只公开 opaque artifact descriptor，不泄露正式路径。"""

    if not isinstance(value, dict):
        raise GateFailure(f"Web artifact descriptor 类型错误: {value!r}")
    descriptor = cast(dict[str, object], value)
    artifact_id = descriptor.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise GateFailure(f"Web artifact identity 缺失: {descriptor}")
    if descriptor.get("kind") != "image":
        raise GateFailure(f"Web artifact kind 错误: {descriptor}")
    if descriptor.get("filename") != expected_filename:
        raise GateFailure(f"Web artifact filename 错误: {descriptor}")
    if descriptor.get("media_type") != "image/png":
        raise GateFailure(f"Web artifact media_type 错误: {descriptor}")
    if descriptor.get("size_bytes") != 8:
        raise GateFailure(f"Web artifact size 错误: {descriptor}")
    sha256 = descriptor.get("sha256")
    expected_sha256 = hashlib.sha256(b"\x89PNG\r\n\x1a\n").hexdigest()
    if sha256 != expected_sha256:
        raise GateFailure(f"Web artifact sha256 错误: {descriptor}")
    if descriptor.get("url") != f"/api/chat/artifacts/{artifact_id}":
        raise GateFailure(f"Web artifact URL 错误: {descriptor}")
    if "/sandbox/" in json.dumps(descriptor, ensure_ascii=False):
        raise GateFailure(f"Web artifact descriptor 泄露正式路径: {descriptor}")
    return descriptor


def _assert_dashboard(payload: dict[str, object]) -> None:
    categories = cast(list[dict[str, object]], payload.get("categories"))
    if len(categories) != 1:
        raise GateFailure(f"Meme Dashboard categories 错误: {payload}")
    category = categories[0]
    if category.get("tag") != "shy" or category.get("count") != 1:
        raise GateFailure(f"Meme Dashboard fixture 错误: {category}")


def _assert_model_request(payload: dict[str, object]) -> dict[str, object]:
    requests = cast(list[dict[str, object]], payload.get("requests"))
    if len(requests) != 1:
        raise GateFailure(f"模型请求数量错误: {requests}")
    record = requests[0]
    request_payload = cast(dict[str, object], record.get("payload"))
    encoded = json.dumps(request_payload, ensure_ascii=False, sort_keys=True)
    citation_index = encoded.find("### 记忆引用协议")
    meme_index = encoded.find("# Memes")
    if citation_index < 0 or meme_index <= citation_index:
        raise GateFailure("模型 prompt 未保持 Citation → Meme 顺序")
    if record.get("state") != "completed":
        raise GateFailure(f"模型请求没有完成: {record}")
    return {
        "index": record.get("index"),
        "model": record.get("model"),
        "stream": record.get("stream"),
        "state": record.get("state"),
        "citation_index": citation_index,
        "meme_index": meme_index,
        "payload_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
    }


def _cleanup_evidence(
    compose: list[str],
    project: str,
    env: dict[str, str],
    down_returncode: int,
) -> dict[str, object]:
    residuals: list[str] = []
    ps = _run([*compose, "ps", "-aq"], cwd=ROOT, env=env)
    if ps.stdout.strip():
        residuals.append("containers")
    for resource in ("network", "volume"):
        listed = _run(
            [
                "docker",
                resource,
                "ls",
                "-q",
                "--filter",
                f"label=com.docker.compose.project={project}",
            ],
            cwd=ROOT,
            env=env,
        )
        if listed.stdout.strip():
            residuals.append(f"{resource}s")
    if down_returncode != 0:
        residuals.append("compose_down")
    return {"compose_down_returncode": down_returncode, "residuals": residuals}


def _verify_runtime_immutability(
    sandbox: Path,
    installed: list[dict[str, object]],
    meme_before_sha256: str,
    *,
    phase: str,
) -> dict[str, object]:
    """Verify plugin publication state and Meme assets at a lifecycle boundary."""

    # 1. Keep the product asset tree immutable through runtime and shutdown.
    meme_sha256 = _tree_sha256(sandbox / "workspace/memes")
    if meme_sha256 != meme_before_sha256:
        raise GateFailure(f"{phase} 改写了 workspace/memes")

    # 2. Keep both pointer facts and the full artifact inventory frozen.
    for item in installed:
        plugin_id = str(item["plugin_id"])
        plugin_name = plugin_id.removesuffix(f"@{MARKETPLACE}")
        plugin_base = sandbox / "home/.akashic-plugin/cache" / MARKETPLACE / plugin_name
        pointers = _pointer_paths(plugin_base)
        if pointers != item["pointers_before"]:
            raise GateFailure(f"{phase} 改写 installed pointers: {plugin_id}")
        inventory = _artifact_inventory(plugin_base)
        if inventory != item["artifact_inventory_before"]:
            raise GateFailure(f"{phase} 重发布 installed artifact: {plugin_id}")
        artifact = plugin_base / str(item["pointer"])
        digest = _tree_sha256(artifact)
        if digest != item["artifact_sha256_before"]:
            raise GateFailure(f"{phase} 改写 immutable artifact: {plugin_id}")
        item[f"pointers_{phase}"] = pointers
        item[f"artifact_inventory_{phase}"] = inventory
        item[f"artifact_sha256_{phase}"] = digest
    return {"meme_sha256": meme_sha256, "plugins": len(installed)}


def _pointer_paths(plugin_base: Path) -> dict[str, str | None]:
    pointers = read_pointers(plugin_base)
    if pointers is None:
        raise GateFailure(f"installed pointer state 丢失: {plugin_base}")
    return {"stable": pointers.stable.path, "latest": pointers.latest.path}


def _artifact_inventory(plugin_base: Path) -> list[str]:
    root = plugin_base / ".artifacts"
    if not root.is_dir() or root.is_symlink():
        raise GateFailure(f"installed artifact root 无效: {root}")
    return sorted(path.name for path in root.iterdir())


def _assert_ready(payload: object, endpoint: str) -> None:
    if not isinstance(payload, dict) or payload.get("status") != "ready":
        raise GateFailure(f"{endpoint} 未 ready: {payload}")


def _wait_json(url: str) -> object:
    deadline = time.monotonic() + READINESS_TIMEOUT_S
    last_error = ""
    while time.monotonic() < deadline:
        try:
            payload = _http_json("GET", url, timeout=2)
            if isinstance(payload, dict) and payload.get("status") == "ready":
                return payload
            last_error = repr(payload)
        except Exception as error:
            last_error = str(error)
        time.sleep(0.2)
    raise GateFailure(f"WebUI readiness 超时: {last_error}")


def _http_json(
    method: str,
    url: str,
    payload: object | None = None,
    *,
    timeout: float = SCENARIO_TIMEOUT_S,
) -> object:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    request = Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _http_bytes(url: str) -> bytes:
    with urlopen(url, timeout=SCENARIO_TIMEOUT_S) as response:
        return response.read()


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _git_output(repo: Path, *args: str) -> str:
    completed = _run(["git", *args], cwd=repo)
    if completed.returncode != 0:
        raise GateFailure(f"git {' '.join(args)} 失败: {completed.stdout}")
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _scenario_sha256() -> str:
    payload = json.dumps(
        {
            "profile": SCENARIO_PROFILE,
            "input": USER_INPUT,
            "model_response": MODEL_RESPONSE,
            "plugins": EXPECTED_PLUGIN_IDS,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateFailure as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        raise SystemExit(1) from error
