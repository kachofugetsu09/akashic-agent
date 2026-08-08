#!/usr/bin/env python3
"""Exercise Akasha V2 through the real Docker runtime and replay boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid
from dataclasses import asdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from docker.debug.programmatic_control_probe import (
    CheckResult,
    JsonRpcSocketClient,
    _connect_client,
    _event_turn,
    _extract_id,
    _http_json,
    _model_requests,
    _prepare_host_sandbox,
    _recorded_turn_notifications,
    _repository_digest,
    _tool_lifecycle,
    _wait_barrier,
    _wait_http_ready,
    _wait_socket,
)
from plugins.akasha.infrastructure.persistence import logical_state_sha256

READINESS_DEADLINE_S = 30.0
TURN_DEADLINE_S = 60.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "sha256": _sha256(path),
        "size": stat.st_size,
        "mtimeNs": stat.st_mtime_ns,
    }


def _formal_identity(workspace: Path) -> dict[str, dict[str, object]]:
    return {
        relative: _file_identity(workspace / relative)
        for relative in ("sessions.db", "memory/akasha.db")
    }


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _write_runtime_config(sandbox: Path, source_config: Path) -> None:
    """Write a private model-gate config with the real embedding boundary."""

    # 1. Read only the embedding fields needed by the isolated runtime.
    payload = tomllib.loads(source_config.read_text(encoding="utf-8"))
    embedding = payload["memory"]["embedding"]
    required = ("model", "api_key", "base_url")
    missing = [key for key in required if not str(embedding.get(key) or "")]
    if missing:
        raise ValueError(f"debug embedding config missing fields: {missing}")

    # 2. Keep chat scripted while embeddings use the configured provider.
    config = f"""\
[runtime]
workspace = "/sandbox/workspace"

[llm]
main = "model_gate"

[llm.runtimes.model_gate]
provider = "openai"
model = "model-gate"
api_key = "model-gate-local"
base_url = "http://model-gate:8090/v1"
context_window = 64000
max_output_tokens = 256
input_modalities = ["text"]

[agent]
system_prompt = "Use memory when relevant and follow the scripted response."
max_iterations = 4
max_tokens = 256
spawn_enabled = false

[agent.context]
[agent.context.compaction]
keep_recent_tokens = 20000

[agent.maintenance]
memory_optimizer_enabled = false

[memory]
enabled = true
engine = "akasha"

[memory.embedding]
model = {_toml_string(str(embedding["model"]))}
api_key = {_toml_string(str(embedding["api_key"]))}
base_url = {_toml_string(str(embedding["base_url"]))}
output_dimensionality = 1024

[app_server]
enabled = true
listen = "/sandbox/akashic.sock"
max_connections = 8
ingress_queue_size = 32
outbound_queue_size = 64

[channels.chat]
enabled = true
channel_name = "web"

[channels.telegram]
token = ""

[channels.qq]
bot_uin = ""

[proactive]
enabled = false
profile = "quiet"
"""
    path = sandbox / "config.toml"
    path.write_text(config, encoding="utf-8")
    path.chmod(0o600)


def _load_scripts(model_url: str, scripts: object) -> None:
    _http_json("PUT", f"{model_url}/control/script", scripts)


def _inside_probe(report_dir: Path) -> int:
    """Drive two real turns and freeze evidence before container shutdown."""

    # 1. Establish the real programmatic control and provider boundaries.
    report_dir.mkdir(parents=True, exist_ok=True)
    events_path = report_dir / "akasha-v2-events.jsonl"
    model_url = os.environ.get(
        "AKASHIC_MODEL_GATE_URL",
        "http://model-gate:8090",
    )
    endpoint = Path("/sandbox/akashic.sock")
    memory_path = Path("/sandbox/workspace/memory/akasha.db")
    checks: list[CheckResult] = []
    client: JsonRpcSocketClient | None = None
    try:
        _wait_http_ready(f"{model_url}/readyz", READINESS_DEADLINE_S)
        _wait_socket(endpoint, READINESS_DEADLINE_S)
        client = _connect_client(endpoint, events_path)

        # 2. Complete one turn through query, provider, persistence, and commit.
        _load_scripts(
            model_url,
            {"mode": "complete", "content": "first remembered answer"},
        )
        thread_id = _extract_id(
            client.request(
                "thread/start",
                {"metadata": {"gate": "akasha-v2"}},
            ),
            "thread",
        )
        first_turn = _extract_id(
            client.request(
                "turn/start",
                {
                    "threadId": thread_id,
                    "input": "alpha medical story",
                    "metadata": {},
                },
            ),
            "turn",
        )
        first_terminal = _event_turn(
            client.wait_terminal(first_turn, timeout=TURN_DEADLINE_S)
        )
        first_hash = logical_state_sha256(memory_path)

        # 3. Pause after recall but before final response to prove read-only state.
        _http_json(
            "PUT",
            f"{model_url}/control/barriers/akasha-v2-after-recall",
        )
        _load_scripts(
            model_url,
            [
                {
                    "mode": "complete",
                    "tool_calls": [
                        {
                            "id": "call_akasha_recall",
                            "name": "recall_memory",
                            "arguments": {
                                "query": "alpha medical story",
                                "limit": 5,
                            },
                        }
                    ],
                },
                {
                    "mode": "complete",
                    "content": "second answer after recall",
                    "barrier": "akasha-v2-after-recall",
                },
            ],
        )
        second_turn = _extract_id(
            client.request(
                "turn/start",
                {
                    "threadId": thread_id,
                    "input": "what happened next",
                    "metadata": {},
                },
            ),
            "turn",
        )
        _wait_barrier(model_url, "akasha-v2-after-recall")
        during_recall_hash = logical_state_sha256(memory_path)
        _http_json(
            "POST",
            f"{model_url}/control/barriers/akasha-v2-after-recall/release",
        )
        second_terminal = _event_turn(
            client.wait_terminal(second_turn, timeout=TURN_DEADLINE_S)
        )
        second_hash = logical_state_sha256(memory_path)

        # 4. Inspect provider payload and the actual tool lifecycle.
        requests = _model_requests(
            _http_json("GET", f"{model_url}/control/requests")
        )
        request_payloads = [
            item.get("payload")
            for item in requests
            if isinstance(item, dict)
        ]
        automatic_context_seen = any(
            "# Akasha memory" in json.dumps(
                payload,
                ensure_ascii=False,
            )
            for payload in request_payloads[1:]
        )
        notifications = _recorded_turn_notifications(
            events_path,
            second_turn,
        )
        started, completed = _tool_lifecycle(
            notifications,
            "recall_memory",
        )
        tool_completed = completed.get("data", {}).get("status") == "success"
        checks.extend(
            [
                CheckResult(
                    "AKV2-01",
                    first_terminal.get("status") == "completed"
                    and first_terminal.get("finalResponse")
                    == "first remembered answer",
                    {
                        "threadId": thread_id,
                        "turnId": first_turn,
                        "logicalState": first_hash,
                    },
                ),
                CheckResult(
                    "AKV2-02",
                    automatic_context_seen,
                    {"modelRequestCount": len(requests)},
                ),
                CheckResult(
                    "AKV2-03",
                    first_hash == during_recall_hash and tool_completed,
                    {
                        "beforeRecall": first_hash,
                        "duringRecall": during_recall_hash,
                        "toolStarted": started,
                        "toolCompleted": completed,
                    },
                ),
                CheckResult(
                    "AKV2-04",
                    second_terminal.get("status") == "completed"
                    and second_hash != first_hash,
                    {
                        "turnId": second_turn,
                        "beforeCommit": first_hash,
                        "afterCommit": second_hash,
                    },
                ),
            ]
        )
    except Exception as error:
        checks.append(
            CheckResult(
                "AKV2-controller",
                False,
                {"type": type(error).__name__, "message": str(error)},
            )
        )
    finally:
        if client is not None:
            client.close()

    report = {
        "checks": [asdict(check) for check in checks],
        "passed": bool(checks) and all(check.passed for check in checks),
    }
    (report_dir / "akasha-v2-inside.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if report["passed"] else 1


def _run_controller(
    repo: Path,
    source_config: Path,
    formal_workspace: Path,
) -> int:
    """Create one isolated runtime, run Docker, replay, and publish evidence."""

    # 1. Freeze protected identities and prepare a private sandbox.
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    report_dir = (
        repo
        / "docker/debug/reports/akasha-v2-runtime"
        / run_id
    )
    report_dir.mkdir(parents=True)
    sandbox = Path(
        tempfile.mkdtemp(prefix="akashic-akasha-v2-gate-", dir="/tmp")
    )
    _prepare_host_sandbox(sandbox, repo)
    _write_runtime_config(sandbox, source_config)
    formal_before = _formal_identity(formal_workspace)
    repository_before = _repository_digest(repo)

    env = {
        **os.environ,
        "AKASHIC_CONTROL_SANDBOX": str(sandbox),
        "UID": str(os.getuid()),
        "GID": str(os.getgid()),
    }
    project = f"akashic-akasha-v2-{run_id.lower()}"
    compose = [
        "docker",
        "compose",
        "-p",
        project,
        "-f",
        str(repo / "docker/debug/docker-compose.control-gate.yml"),
    ]
    checks: list[CheckResult] = []
    controller_error = ""
    try:
        # 2. Run the actual gateway and the in-container two-turn scenario.
        subprocess.run(
            [*compose, "build", "model-gate"],
            cwd=repo,
            env=env,
            check=True,
        )
        subprocess.run(
            [
                *compose,
                "up",
                "-d",
                "model-gate",
                "akashic-control-gate",
            ],
            cwd=repo,
            env=env,
            check=True,
        )
        inside = subprocess.run(
            [
                *compose,
                "run",
                "--rm",
                "-T",
                "control-probe",
                "python",
                "docker/debug/akasha_v2_runtime_probe.py",
                "--inside-container",
                "--report-dir",
                "/sandbox/reports",
            ],
            cwd=repo,
            env=env,
            check=False,
        )
        inside_payload = json.loads(
            (sandbox / "reports/akasha-v2-inside.json").read_text(
                encoding="utf-8"
            )
        )
        checks.extend(
            CheckResult(**item) for item in inside_payload["checks"]
        )
        if inside.returncode != 0:
            raise RuntimeError(
                f"inside Akasha V2 probe failed: {inside.returncode}"
            )

        # 3. Stop online growth and replay the exact persisted source.
        subprocess.run(
            [*compose, "stop", "-t", "15", "akashic-control-gate"],
            cwd=repo,
            env=env,
            check=True,
        )
        replay = subprocess.run(
            [
                *compose,
                "run",
                "--rm",
                "-T",
                "--no-deps",
                "control-probe",
                "python",
                "scripts/build_akasha_db.py",
                "--sessions-db",
                "/sandbox/workspace/sessions.db",
                "--db-path",
                "/sandbox/workspace/memory/akasha-replay.db",
                "--embedding-model",
                "text-embedding-v4",
                "--embedding-dim",
                "1024",
                "--require-complete-embeddings",
            ],
            cwd=repo,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (report_dir / "replay.log").write_text(
            replay.stdout,
            encoding="utf-8",
        )
        online_hash = logical_state_sha256(
            sandbox / "workspace/memory/akasha.db"
        )
        replay_hash = logical_state_sha256(
            sandbox / "workspace/memory/akasha-replay.db"
        )
        checks.append(
            CheckResult(
                "AKV2-05",
                replay.returncode == 0 and online_hash == replay_hash,
                {
                    "replayReturncode": replay.returncode,
                    "onlineLogicalState": online_hash,
                    "replayLogicalState": replay_hash,
                },
            )
        )
    except Exception as error:
        controller_error = f"{type(error).__name__}: {error}"
    finally:
        logs = subprocess.run(
            [*compose, "logs", "--no-color"],
            cwd=repo,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (report_dir / "compose.log").write_text(
            logs.stdout,
            encoding="utf-8",
        )
        cleanup = subprocess.run(
            [*compose, "down", "--remove-orphans", "--volumes"],
            cwd=repo,
            env=env,
            check=False,
        )
        residual = subprocess.run(
            [*compose, "ps", "-aq"],
            cwd=repo,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        ).stdout.split()

    # 4. Prove protected paths and repository bytes did not change.
    formal_after = _formal_identity(formal_workspace)
    repository_after = _repository_digest(repo)
    checks.append(
        CheckResult(
            "AKV2-06",
            cleanup.returncode == 0
            and not residual
            and formal_before == formal_after
            and repository_before == repository_after,
            {
                "cleanupReturncode": cleanup.returncode,
                "residualContainers": residual,
                "formalWorkspaceUnchanged": formal_before == formal_after,
                "repositoryUnchanged": repository_before == repository_after,
                "formalBefore": formal_before,
                "formalAfter": formal_after,
            },
        )
    )
    passed = (
        not controller_error
        and bool(checks)
        and all(check.passed for check in checks)
    )
    report = {
        "runId": run_id,
        "status": "passed" if passed else "failed",
        "checks": [asdict(check) for check in checks],
        "controllerError": controller_error,
        "reportDir": str(report_dir),
    }
    (report_dir / "gate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    shutil.rmtree(sandbox)
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inside-container", action="store_true")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("/sandbox/reports"),
    )
    parser.add_argument(
        "--source-config",
        type=Path,
        default=Path("/mnt/data/coding/akasic-agent/config.toml"),
    )
    parser.add_argument(
        "--formal-workspace",
        type=Path,
        default=Path("/home/huashen/.akashic/workspace"),
    )
    arguments = parser.parse_args()
    if arguments.inside_container:
        return _inside_probe(arguments.report_dir)
    return _run_controller(
        Path(__file__).resolve().parents[2],
        arguments.source_config.resolve(strict=True),
        arguments.formal_workspace.resolve(strict=True),
    )


if __name__ == "__main__":
    raise SystemExit(main())
