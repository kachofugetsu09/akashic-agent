from __future__ import annotations

from agent.plugin_composition import (
    MCP_SERVERS,
    WORKLOADS,
    Context,
    McpServerDefinition,
    Workload,
    WorkloadData,
    WorkloadEnv,
    WorkloadHealth,
    WorkloadLimits,
    WorkloadPort,
)

api_version = 3
name = "computer"
version = "1.0.0"
desc = "Persistent browser and visual control"
author = "Akashic Core"
inject = (WORKLOADS, MCP_SERVERS)
skill_roots = ("skills",)
drift_skill_roots = ()
workspace_roots = ()
workspace_files = ()
dashboard_module = "dashboard.py"
web_module = "web_module.js"
web_requires = ("conversation.tools.v1",)
web_provides = ()
web_contract_digests = {
    "conversation.tools.v1": "1ce9b1dfe70907c50c00c17bc428c8ddab91d4c4839db9da13b0283aad1035fb",
}

_IMAGE = (
    "ghcr.io/kachofugetsu09/akashic-computer@"
    "sha256:4e3abb85a16d081303bf520e67eb725781aa39f459048fbdc5b081b217abae3c"
)


async def apply(ctx: Context, config: object) -> None:
    """Register the Computer workload and its narrow MCP adapter."""

    _ = config
    await ctx.require(WORKLOADS).register(
        ctx,
        Workload(
            name="computer",
            image=_IMAGE,
            command=("/opt/computer/start.sh",),
            ports=(
                WorkloadPort("gateway", 8080),
                WorkloadPort("opencli", 19826, loopback=19826),
            ),
            data=(WorkloadData("state", "/data"),),
            health=WorkloadHealth("gateway", "/health", 90.0),
            limits=WorkloadLimits(2048, 2.0, 512),
        ),
    )
    await ctx.require(MCP_SERVERS).register(
        ctx,
        McpServerDefinition(
            name="computer",
            command=("mcp_server.py",),
            required_tools=(
                "browser_observe",
                "browser_action",
                "computer_observe",
                "computer_action",
            ),
            candidate_read_only_tools=("browser_observe", "computer_observe"),
            workload_env=(WorkloadEnv("COMPUTER_URL", "computer", "gateway"),),
        ),
    )
