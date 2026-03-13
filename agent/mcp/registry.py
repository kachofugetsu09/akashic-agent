"""McpServerRegistry: 管理多个 MCP server 连接，持久化到 mcp_servers.json。"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from agent.mcp.client import McpClient, McpToolInfo
from agent.mcp.tool import McpToolWrapper
from agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _mcp_search_keywords(info: McpToolInfo, server_name: str) -> list[str]:
    """从 MCP 工具信息中提取搜索关键词。

    来源：server 名、工具名拆词、description 前几个词。
    """
    keywords: list[str] = [server_name.lower()]

    # 工具名拆词：下划线、连字符、驼峰
    raw_name = re.sub(r"([a-z])([A-Z])", r"\1_\2", info.name)  # camelCase → snake
    parts = re.split(r"[_\-]+", raw_name)
    keywords.extend(p.lower() for p in parts if len(p) > 1)

    # description 前 8 个词（过滤短词和标点）
    desc_words = re.split(r"\s+", info.description[:120])
    for w in desc_words[:8]:
        w_clean = w.strip(".,;:!?\"'()[]").lower()
        if len(w_clean) > 2:
            keywords.append(w_clean)

    # 去重保序
    seen: set[str] = set()
    result: list[str] = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            result.append(kw)
    if server_name == "feed":
        if info.name == "feed_manage":
            result.extend(["rss订阅", "添加订阅", "删除订阅", "取消订阅", "订阅管理"])
        if info.name == "feed_query":
            result.extend(["查询订阅", "最近新闻", "最新资讯", "这个源有什么新闻", "rss查询"])
    return result


class McpServerRegistry:
    """管理 MCP server 连接生命周期，并将工具同步进 ToolRegistry。

    持久化格式（mcp_servers.json）：
    {
      "servers": {
        "calendar": {
          "command": ["python", "/path/to/run_server.py"],
          "env": {"GOOGLE_CLIENT_ID": "..."}
        }
      }
    }
    """

    def __init__(self, config_path: Path, tool_registry: ToolRegistry) -> None:
        self._config_path = config_path
        self._tool_registry = tool_registry
        self._clients: dict[str, McpClient] = {}
        self._server_tools: dict[str, list[str]] = (
            {}
        )  # server_name -> 已注册的工具名列表

    async def load_and_connect_all(self) -> None:
        """启动时读取持久化配置，重连所有 server。"""
        for name, cfg in self._load_raw_configs().items():
            try:
                await self._connect(name, cfg["command"], cfg.get("env"), cfg.get("cwd"))
            except Exception as e:
                logger.error("[mcp] 重连 %r 失败: %s", name, e)

    async def add(
        self,
        name: str,
        command: list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> str:
        if name in self._clients:
            return f"MCP server {name!r} 已存在。如需更新，请先 mcp_remove 再重新添加。"
        try:
            tool_names = await self._connect(name, command, env, cwd)
        except Exception as e:
            return f"连接 MCP server {name!r} 失败：{e}"
        self._save()
        return (
            f"已连接 MCP server {name!r}，注册了 {len(tool_names)} 个工具：\n"
            + "\n".join(f"- {n}" for n in tool_names)
        )

    async def remove(self, name: str) -> str:
        if name not in self._clients:
            return f"MCP server {name!r} 不存在，当前已注册：{list(self._clients.keys()) or '无'}"
        for tool_name in self._server_tools.pop(name, []):
            self._tool_registry.unregister(tool_name)
        await self._clients.pop(name).disconnect()
        self._save()
        return f"已注销 MCP server {name!r}。"

    def list_servers(self) -> str:
        if not self._clients:
            return "当前没有已注册的 MCP server。"
        lines = []
        for name in self._clients:
            tools = self._server_tools.get(name, [])
            lines.append(f"- {name}（{len(tools)} 个工具）：{', '.join(tools) or '无'}")
        return "\n".join(lines)

    async def _connect(
        self,
        name: str,
        command: list[str],
        env: dict[str, str] | None,
        cwd: str | None = None,
    ) -> list[str]:
        client = McpClient(name=name, command=command, env=env, cwd=cwd)
        tool_infos = await client.connect()
        tool_names = []
        for info in tool_infos:
            wrapper = McpToolWrapper(client, info)
            self._tool_registry.register(
                wrapper,
                tags=["mcp", name],
                risk="external-side-effect",
                search_keywords=_mcp_search_keywords(info, name),
            )
            tool_names.append(wrapper.name)
        self._clients[name] = client
        self._server_tools[name] = tool_names
        return tool_names

    def _load_raw_configs(self) -> dict[str, Any]:
        if not self._config_path.exists():
            return {}
        try:
            data = json.loads(self._config_path.read_text(encoding="utf-8"))
            return data.get("servers", {})
        except Exception as e:
            logger.warning("[mcp] 读取配置失败 %s: %s", self._config_path, e)
            return {}

    def _save(self) -> None:
        servers = {
            name: {
                "command": client.command,
                "env": client.env,
                "cwd": client.cwd,
            }
            for name, client in self._clients.items()
        }
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            self._config_path.write_text(
                json.dumps({"servers": servers}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error("[mcp] 保存配置失败: %s", e)
