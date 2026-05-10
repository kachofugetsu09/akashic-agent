"""
调试脚本：直接对沙盒 workspace 的 MEMORY.md 运行一次 MemoryOptimizer，
观察 prompt 修改后是否会破坏已修正的内容。

用法（在容器内）：
  python /app/docker/debug/run_optimizer_once.py \
    --config /sandbox/config.toml \
    --workspace /sandbox/workspace
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.config_models import Config
from bootstrap.memory import build_memory_admin_runtime
from bootstrap.providers import build_providers
from core.net.http import SharedHttpResources
from proactive_v2.memory_optimizer import MemoryOptimizer


async def main(config_path: str, workspace_str: str) -> None:
    workspace = Path(workspace_str)
    config = Config.load(config_path)
    http_resources = SharedHttpResources()
    provider, _, _ = build_providers(config)
    memory_runtime = build_memory_admin_runtime(
        config=config,
        workspace=workspace,
        provider=provider,
        light_provider=provider,
        http_resources=http_resources,
    )

    store = memory_runtime.markdown.store
    print("=== MEMORY.md (before) ===")
    print(store.read_long_term())
    print("\n=== PENDING.md (before) ===")
    print(store.read_pending() or "(empty)")
    print("\n>>> 开始 optimize...\n")

    optimizer = MemoryOptimizer(
        memory=store,
        provider=provider,
        model=config.model,
        max_tokens=8192,
    )
    await optimizer.optimize()

    print("\n=== MEMORY.md (after) ===")
    print(store.read_long_term())
    print("\n=== SELF.md (after) ===")
    print(store.read_self())

    await memory_runtime.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/sandbox/config.toml")
    parser.add_argument("--workspace", default="/sandbox/workspace")
    args = parser.parse_args()
    asyncio.run(main(args.config, args.workspace))
