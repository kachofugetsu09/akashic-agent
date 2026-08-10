from pathlib import Path

from agent.skills import SkillsLoader


REPO_ROOT = Path(__file__).parents[1]


def test_manage_workspace_mcp_is_discoverable_builtin(tmp_path: Path) -> None:
    loader = SkillsLoader(tmp_path, builtin_skills_dir=REPO_ROOT / "skills")
    record = loader.load_skill_record("manage-workspace-mcp")

    assert record is not None
    assert record.source == "builtin"
    assert record.available is True
    for trigger in (
        "常驻 MCP",
        "MCP 热重载",
        "MCP 工具为何没出现",
        "非插件 MCP",
    ):
        assert trigger in record.description


def test_manage_workspace_mcp_forbids_legacy_paths(tmp_path: Path) -> None:
    loader = SkillsLoader(tmp_path, builtin_skills_dir=REPO_ROOT / "skills")
    body = loader.load_skill_body("manage-workspace-mcp")

    assert body is not None
    for contract in (
        "`workspace_mcp_apply`",
        "`workspace_mcp_status`",
        "`workspace_mcp_remove`",
        "下一轮开始可用",
        "不要添加 `[mcp_servers]`",
        "不要调用 `agent_restart`",
    ):
        assert contract in body


def test_plugin_skill_routes_standalone_mcp_to_workspace_skill(tmp_path: Path) -> None:
    loader = SkillsLoader(tmp_path, builtin_skills_dir=REPO_ROOT / "skills")
    record = loader.load_skill_record("plugin-system")
    body = loader.load_skill_body("plugin-system")

    assert record is not None and body is not None
    assert "插件内 MCP" in record.description
    assert "独立本地 MCP server 使用 manage-workspace-mcp" in record.when_to_use
    assert "加载 `manage-workspace-mcp`" in body


def test_plugin_skill_explains_turn_boundary_uninstall_result(tmp_path: Path) -> None:
    loader = SkillsLoader(tmp_path, builtin_skills_dir=REPO_ROOT / "skills")
    body = loader.load_skill_body("plugin-system")

    assert body is not None
    for contract in (
        "卸载成功返回表示意图已绑定当前 turn，不表示代码已经删除",
        "当前 turn 可以完成",
        "本轮结束后 Core 自动停止 endpoint",
        "plugin-data 保留",
        "下一 turn 不再加载",
        "manifest entry 和 cache 已移除",
        "清理失败时说明残留路径和错误",
    ):
        assert contract in body
