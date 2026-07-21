# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from programmatic_control_probe import (
    _prepare_host_sandbox,
    _repository_digest,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_COMPOSE_FILE = _PROJECT_ROOT / "docker/debug/docker-compose.control-gate.yml"
_TEST_TARGETS = (
    "tests/test_migration_runner.py",
    "tests/test_provider_runtime_akasha_migration.py",
    "tests/test_migration_append_only.py",
    "tests/test_main_lightweight_commands.py",
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _case_results(junit_path: Path) -> list[dict[str, str]]:
    """把 JUnit testcase 转为便于审阅的迁移 case 清单。"""

    root = ET.parse(junit_path).getroot()
    results: list[dict[str, str]] = []
    for testcase in root.iter("testcase"):
        status = "passed"
        if testcase.find("failure") is not None:
            status = "failed"
        elif testcase.find("error") is not None:
            status = "error"
        elif testcase.find("skipped") is not None:
            status = "skipped"
        results.append(
            {
                "case": f"{testcase.attrib.get('classname', '')}::{testcase.attrib['name']}",
                "status": status,
            }
        )
    return results


def _compose_command(project: str) -> list[str]:
    return [
        "docker",
        "compose",
        "-p",
        project,
        "-f",
        str(_COMPOSE_FILE),
    ]


def run() -> int:
    """在隔离 Docker sandbox 中执行完整迁移 case matrix 并落盘证据。"""

    # 1. 创建只属于本次 Gate 的源码快照、状态目录和报告目录。
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    project = f"akashic-migration-{run_id.lower()}"
    report_dir = _PROJECT_ROOT / "docker/debug/reports/migrations" / run_id
    sandbox = Path(tempfile.mkdtemp(prefix="akashic-migration-gate-"))
    before = _repository_digest(_PROJECT_ROOT)
    _prepare_host_sandbox(sandbox, _PROJECT_ROOT)
    env = os.environ.copy()
    env.update(
        {
            "AKASHIC_CONTROL_SANDBOX": str(sandbox),
            "UID": str(os.getuid()),
            "GID": str(os.getgid()),
        }
    )
    compose = _compose_command(project)

    # 2. 使用与 runtime Gate 相同的镜像，只读挂载候选源码并执行真实 Git case。
    build = subprocess.run(
        [*compose, "build", "model-gate"],
        cwd=_PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    (report_dir / "build.log").parent.mkdir(parents=True, exist_ok=True)
    (report_dir / "build.log").write_text(build.stdout, encoding="utf-8")
    completed: subprocess.CompletedProcess[str] | None = None
    cleanup: subprocess.CompletedProcess[str] | None = None
    try:
        if build.returncode == 0:
            completed = subprocess.run(
                [
                    *compose,
                    "run",
                    "--rm",
                    "-T",
                    "--no-deps",
                    "control-probe",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "-W",
                    "error",
                    "--junitxml=/sandbox/reports/migration-junit.xml",
                    *_TEST_TARGETS,
                ],
                cwd=_PROJECT_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            (report_dir / "pytest.log").write_text(completed.stdout, encoding="utf-8")
    finally:
        cleanup = subprocess.run(
            [*compose, "down", "--volumes", "--remove-orphans"],
            cwd=_PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        (report_dir / "cleanup.log").write_text(cleanup.stdout, encoding="utf-8")

    # 3. 汇总每个 testcase、源码不变性和容器清理结果。
    junit_path = sandbox / "reports/migration-junit.xml"
    cases = _case_results(junit_path) if junit_path.is_file() else []
    if cases:
        shutil.copy2(junit_path, report_dir / "migration-junit.xml")
    after = _repository_digest(_PROJECT_ROOT)
    passed = (
        build.returncode == 0
        and completed is not None
        and completed.returncode == 0
        and cleanup.returncode == 0
        and bool(cases)
        and all(case["status"] == "passed" for case in cases)
        and before == after
    )
    report = {
        "runId": run_id,
        "gate": "migration-case-matrix",
        "status": "passed" if passed else "failed",
        "caseCount": len(cases),
        "cases": cases,
        "buildReturncode": build.returncode,
        "pytestReturncode": completed.returncode if completed is not None else None,
        "cleanupReturncode": cleanup.returncode,
        "repositoriesUnchanged": before == after,
        "reportDir": str(report_dir),
    }
    _write_json(report_dir / "gate.json", report)
    shutil.rmtree(sandbox)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(run())
