from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = ROOT / "docker" / "debug" / "plugin-api-v2.lock.json"
DEFAULT_REPORT = (
    ROOT / "docker" / "debug" / "reports" / "plugin-api-v2" / "gate.json"
)
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
REPOSITORY_PATTERN = re.compile(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
RUNTIME_PHASES = ("atomic-reload", "all-plugins", "fitbit")
HOST_CHANNEL_CONTRACT_PLUGIN_IDS = ("feishu", "qqbot")
EXPECTED_PLUGIN_IDS = {
    "calendar-mcp",
    "citation",
    "computer-use-linux",
    "context_pressure",
    "daynight_gate",
    "emotion",
    "feed-mcp",
    "feishu",
    "fitbit-mcp",
    "huayue-skills",
    "meme",
    "observe",
    "plugin_undo",
    "proactive_feedback",
    "qqbot",
    "setup_helper",
    "shell_restore",
    "shell_safety",
    "status_commands",
    "steam-mcp",
    "tool_loop_guard",
}


@dataclass(frozen=True)
class LockedRepository:
    id: str
    repository: str
    commit: str


@dataclass(frozen=True)
class PluginApiV2Lock:
    contract: LockedRepository
    plugins: tuple[LockedRepository, ...]


def main() -> int:
    """检出不可变发布组合并运行静态合同与隔离 Runtime Gate。"""

    # 1. 固定输入和核心源码身份
    args = _parse_args()
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    core_status = _git_output(ROOT, "status", "--porcelain").splitlines()
    report: dict[str, object] = {
        "status": "failed",
        "checked_at": datetime.now(UTC).isoformat(),
        "core_head": _git_output(ROOT, "rev-parse", "HEAD"),
        "core_dirty_status": core_status,
        "lock": str(args.lock.resolve()),
        "lock_sha256": hashlib.sha256(args.lock.read_bytes()).hexdigest(),
        "runtime_phases": {},
    }

    # 2. 在一次性目录精确检出合同和所有插件
    try:
        if args.require_clean_core and core_status:
            raise RuntimeError(f"核心工作树不干净: {core_status}")
        release = _load_lock(args.lock.resolve())
        report["contract"] = asdict(release.contract)
        report["plugins"] = [asdict(item) for item in release.plugins]
        with tempfile.TemporaryDirectory(prefix="akashic-plugin-api-v2-") as raw_temp:
            temp_root = Path(raw_temp)
            contract_root = temp_root / "contract"
            plugin_root = temp_root / "plugins"
            _checkout_locked_commit(release.contract, contract_root)
            for plugin in release.plugins:
                _checkout_locked_commit(plugin, plugin_root / plugin.id)

            # 3. 先做全量静态合同，再运行三个真实业务 Gate
            static_report = _run_static_contract(
                contract_root=contract_root,
                plugin_root=plugin_root,
                plugins=release.plugins,
                report_dir=report_path.parent,
            )
            report["static_contract"] = static_report
            if static_report["returncode"] != 0:
                raise RuntimeError("Plugin API v2 静态合同失败")

            host_channel_report = _run_host_channel_contract(
                plugin_root=plugin_root,
                report_dir=report_path.parent,
            )
            report["host_channel_contract"] = host_channel_report
            if host_channel_report["returncode"] != 0:
                raise RuntimeError("外部渠道 Host 合同失败")

            phase_reports = cast(dict[str, object], report["runtime_phases"])
            for phase in RUNTIME_PHASES:
                phase_report = _run_runtime_phase(
                    phase=phase,
                    plugin_root=plugin_root,
                    report_dir=report_path.parent,
                )
                phase_reports[phase] = phase_report
                if phase_report["returncode"] != 0:
                    raise RuntimeError(f"Runtime Gate 失败: {phase}")
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"

    # 4. 无论成功失败都留下可上传的精确组合证据
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if report["status"] != "passed":
        print(f"plugin api v2 gate failed: {report.get('error')}", file=sys.stderr)
        print(f"evidence: {report_path}", file=sys.stderr)
        return 1
    print(f"plugin api v2 gate passed: {report_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="验证固定 Core 与外部插件组合的 Plugin API v2 合同",
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--require-clean-core", action="store_true")
    return parser.parse_args()


def _load_lock(path: Path) -> PluginApiV2Lock:
    """严格解析发布锁，不接受缺仓库、额外字段或浮动引用。"""

    # 1. 校验发布锁根结构
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "contract",
        "plugins",
    }:
        raise ValueError("Plugin API v2 发布锁根结构无效")
    if raw["schema_version"] != 1:
        raise ValueError(f"不支持的 Plugin API v2 发布锁版本: {raw['schema_version']}")
    plugins_raw = raw["plugins"]
    if not isinstance(plugins_raw, list):
        raise ValueError("Plugin API v2 发布锁 plugins 必须是数组")

    # 2. 每个远端只能由 HTTPS 地址和完整 commit 标识
    contract = _parse_repository(raw["contract"])
    plugins = tuple(_parse_repository(item) for item in plugins_raw)
    ids = [plugin.id for plugin in plugins]
    if len(ids) != len(set(ids)):
        raise ValueError("Plugin API v2 发布锁包含重复插件")
    actual_ids = set(ids)
    if actual_ids != EXPECTED_PLUGIN_IDS:
        missing = sorted(EXPECTED_PLUGIN_IDS - actual_ids)
        extra = sorted(actual_ids - EXPECTED_PLUGIN_IDS)
        raise ValueError(f"Plugin API v2 发布锁插件集合错误: missing={missing} extra={extra}")
    if contract.id != "plugin-contracts":
        raise ValueError("Plugin API v2 发布锁 contract id 必须是 plugin-contracts")
    return PluginApiV2Lock(contract=contract, plugins=plugins)


def _parse_repository(raw: object) -> LockedRepository:
    expected = {"id", "repository", "commit"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError(f"Plugin API v2 仓库字段无效: {raw}")
    item = cast(dict[str, object], raw)
    values: dict[str, str] = {}
    for name in expected:
        value = item[name]
        if not isinstance(value, str) or not value:
            raise ValueError(f"Plugin API v2 仓库字段必须是非空字符串: {name}")
        values[name] = value
    if REPOSITORY_PATTERN.fullmatch(values["repository"]) is None:
        raise ValueError(f"插件仓库必须是 GitHub HTTPS 地址: {values['repository']}")
    if COMMIT_PATTERN.fullmatch(values["commit"]) is None:
        raise ValueError(f"插件 commit 必须是完整 SHA: {values['commit']}")
    return LockedRepository(**values)


def _checkout_locked_commit(repository: LockedRepository, checkout: Path) -> None:
    """只获取发布锁声明的公开 Git 对象。"""

    # 1. 创建不复用宿主 checkout 的临时仓库
    _run(("git", "init", "--quiet", str(checkout)), cwd=ROOT)
    _run(
        ("git", "remote", "add", "origin", repository.repository),
        cwd=checkout,
    )

    # 2. 精确获取并核对完整 SHA
    _run(
        ("git", "fetch", "--quiet", "--depth=1", "origin", repository.commit),
        cwd=checkout,
    )
    _run(("git", "checkout", "--quiet", "--detach", "FETCH_HEAD"), cwd=checkout)
    actual = _git_output(checkout, "rev-parse", "HEAD")
    if actual != repository.commit:
        raise RuntimeError(
            f"检出提交与发布锁不一致: {repository.id} expected={repository.commit} actual={actual}"
        )
    if _git_output(checkout, "status", "--porcelain"):
        raise RuntimeError(f"插件检出后工作树不干净: {repository.id}")


def _run_static_contract(
    *,
    contract_root: Path,
    plugin_root: Path,
    plugins: tuple[LockedRepository, ...],
    report_dir: Path,
) -> dict[str, object]:
    """用锁定版本的合同检查器验证全部插件入口。"""

    output_path = report_dir / "static-contract.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(contract_root)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "akashic_plugin_contracts",
            "check",
            *(str(plugin_root / plugin.id / "plugin.py") for plugin in plugins),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output_path.write_text(result.stdout, encoding="utf-8")
    return {
        "returncode": result.returncode,
        "report": output_path.name,
        "sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
    }


def _run_runtime_phase(
    *,
    phase: str,
    plugin_root: Path,
    report_dir: Path,
) -> dict[str, object]:
    """在 Docker Debug sandbox 运行一个可观察的插件业务场景。"""

    output_path = report_dir / f"runtime-{phase}.log"
    env = os.environ.copy()
    env["AKASHIC_PLUGIN_SOURCE"] = str(plugin_root)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "docker" / "debug" / "plugin_hot_reload_probe.py"),
            "--scenario",
            "full-runtime",
            "--phase",
            phase,
        ],
        cwd=ROOT,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output_path.write_text(result.stdout, encoding="utf-8")
    print(f"runtime {phase}: {'passed' if result.returncode == 0 else 'failed'}")
    return {
        "returncode": result.returncode,
        "report": output_path.name,
        "sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
    }


def _run_host_channel_contract(
    *,
    plugin_root: Path,
    report_dir: Path,
) -> dict[str, object]:
    """用当前 Core 接口运行锁定渠道仓库的启动与投递合同测试。"""

    # 1. 每个仓库独立运行，避免同名测试模块互相污染
    env = os.environ.copy()
    env["AKASHIC_AGENT_ROOT"] = str(ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    outputs: list[str] = []
    returncode = 0
    for plugin_id in HOST_CHANNEL_CONTRACT_PLUGIN_IDS:
        plugin_path = plugin_root / plugin_id
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "tests",
            ],
            cwd=plugin_path,
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        outputs.append(f"===== {plugin_id} =====\n{result.stdout}")
        if result.returncode != 0:
            returncode = result.returncode

    # 2. 保存可审计的固定组合结果
    output = "\n".join(outputs)
    output_path = report_dir / "host-channel-contract.log"
    output_path.write_text(output, encoding="utf-8")
    return {
        "returncode": returncode,
        "plugins": list(HOST_CHANNEL_CONTRACT_PLUGIN_IDS),
        "report": output_path.name,
        "sha256": hashlib.sha256(output.encode()).hexdigest(),
    }


def _run(command: tuple[str, ...], *, cwd: Path) -> None:
    subprocess.run(
        command,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repo), *args),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
