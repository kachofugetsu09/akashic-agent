"""运行内置系统的公开协议场景，以及默认附带的提交故障和生命周期回归。"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


SYSTEM = ("test_builtin_behavior.py", "test_builtin_system.py")
BOUNDARIES = (
    "test_core_messages.py",
    "test_message_plugin_dashboards.py", "test_message_delivery.py", "test_agent_restart_tool.py",
    "test_scheduler_messages.py", "test_subagent_messages.py", "test_wake_messages.py",
    "test_wake_durable_delivery.py", "test_eventmail_alert_claims.py", "test_message_compaction_records.py",
    "test_message_compaction_summary.py", "test_message_markdown_memory.py", "test_akasha_message_plugin.py",
    "test_plugin_bindings.py", "test_plugin_overlay_scope.py", "test_plugin_hot_reload.py",
)


def main() -> int:
    """公开进程场景可单独重放；默认保留故障边界，不把较小集合称为完整验收。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-only", action="store_true", help="只运行独立 App 场景，跳过局部提交故障和热更新回归")
    args, pytest_args = parser.parse_known_args()
    files = SYSTEM if args.system_only else SYSTEM + BOUNDARIES
    return subprocess.call([sys.executable, "-m", "pytest", *(f"tests/{name}" for name in files), *pytest_args],
                           cwd=Path(__file__).parents[2])


if __name__ == "__main__":
    raise SystemExit(main())
