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
desc = "Persistent Linux desktop, browser, and visual control"
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
    "conversation.tools.v1": "ed47d69b84e946e27a2e297634e96bcc6afc72a3d3089caac1a14632703efb54",
}

_IMAGE = (
    "ghcr.io/kachofugetsu09/akashic-computer@"
    "sha256:6fd3c605380a3daef5ddebb34f2905ee992d2b4e1490fbfb78dcce9f06a3dadb"
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
                WorkloadPort("display", 6080),
                WorkloadPort("opencli", 19826, loopback=19825),
            ),
            data=(WorkloadData("state", "/data"),),
            health=WorkloadHealth("gateway", "/health", 90.0),
            limits=WorkloadLimits(0, 0.0, 0),
            user_namespaces=True,
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
