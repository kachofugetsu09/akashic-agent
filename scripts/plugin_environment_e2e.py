#!/usr/bin/env python3
"""在一次性 HOME 中通过真实安装链验证依赖环境不需要代码归档。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.plugins.install import install_git_plugin
from agent.plugins.python_environment import PythonEnvironments, read_environment_refs
from agent.plugins.static_manifest import load_static_plugin_manifest


def run(output: Path) -> dict[str, object]:
    """安装、重复安装和运行真实解释器，并保留源码与元数据证据。"""
    # 1. 目标必须全新，Git 来源和业务状态只写入此隔离目录。
    output.mkdir()
    source = output / "source"
    source.mkdir()
    (source / "plugin.py").write_text(
        "api_version=3\nname='environment_probe'\nversion='1'\nasync def apply(ctx):\n    pass\n"
    )
    (source / "requirements.txt").write_text("")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=e2e", "-c",
                    "user.email=e2e@example.invalid", "-c", "commit.gpgsign=false",
                    "commit", "-qm", "fixture"], check=True)
    workspace = output / "workspace"
    workspace.mkdir()
    home = output / "plugins"
    result = install_git_plugin(workspace=workspace, source=str(source), marketplace="lab", plugins_home=home)
    code = result.installed_path
    refs = read_environment_refs(code, load_static_plugin_manifest(code))
    environments = PythonEnvironments(workspace)
    root = environments.open(refs["."])
    # 2. 环境解释器、pip 入口和直接目录引用均经过真实进程验证。
    command = [str(root / ".venv/bin/python"), "-I", "-c", "import sys; print(sys.version_info.major)"]
    assert subprocess.check_output(command, text=True).strip() == "3"
    assert json.loads((root / "environment.json").read_text())["version"] == 2
    again = install_git_plugin(workspace=workspace, source=str(source), marketplace="lab", plugins_home=home)
    assert again.installed_path == code
    assert read_environment_refs(code, load_static_plugin_manifest(code)) == refs
    assert not (workspace / "runtime/plugin-archives").exists()
    report: dict[str, object] = {"install": "passed", "reinstall": "passed",
                               "real_interpreter": "passed", "archive_created": False}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    """要求显式隔离 HOME，避免安装流程写入用户目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    home = Path.home().resolve()
    if not home.is_relative_to(output.parent) or home == output.parent:
        parser.error("先设置此证据目录中的独立 HOME，再运行 E2E")
    print(json.dumps(run(output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
