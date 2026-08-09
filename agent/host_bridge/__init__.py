"""Host execution bridge for containerized Akashic runtimes."""

from agent.host_bridge.factory import build_shell_process_manager

__all__ = ["build_shell_process_manager"]
