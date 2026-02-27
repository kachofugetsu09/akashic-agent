from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

import json_repair

from agent.provider import LLMProvider

logger = logging.getLogger(__name__)

_ANALYZER_TOOL_ALLOWLIST = {"read_file", "list_dir"}

_ANALYZER_SYSTEM_PROMPT = """你是 QueryAnalyzer（前置分析器）。
你的任务是轻量预分析：为主模型筛选“旧上下文中的 extra context”，并给出可选取证建议。

硬规则：
1. 你必须输出 history_pointers（历史消息编号，1-based），用于挑选 extra context。
2. required_evidence 是“建议”，不是“强制指令”；可以为空。
3. relevant_sops 是“建议相关 SOP”；命中后系统会自动补充 read_file 取证建议。
4. 只输出 JSON，不要 markdown，不要解释。
"""


@dataclass
class QueryAnalysis:
    required_evidence: list[dict[str, str]] = field(default_factory=list)
    relevant_sops: list[str] = field(default_factory=list)
    history_pointers: list[int] = field(default_factory=list)  # 0-based index
    keep_recent: int = 8
    target_files: list[str] = field(default_factory=list)
    reasoning: str = ""
    is_chitchat: bool = False

    @property
    def needs_tool(self) -> bool:
        return len(self.required_evidence) > 0


class QueryAnalyzer:
    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        workspace: Path,
        tool_schemas: list[dict[str, Any]],
        tool_executor: Callable[[str, dict[str, Any]], Awaitable[str]],
        max_iterations: int = 4,
        max_tokens: int = 768,
    ) -> None:
        self._provider = provider
        self._model = model
        self.workspace = workspace
        self._tool_executor = tool_executor
        self._max_iterations = max_iterations
        self._max_tokens = max_tokens
        self._all_tool_schemas = tool_schemas
        self._tool_schemas = self._filter_tool_schemas(tool_schemas)
        self._tool_names = {
            s.get("function", {}).get("name", "") for s in self._tool_schemas
        }

    async def analyze(
        self,
        message: str,
        history: list[dict[str, Any]],
        message_timestamp: datetime | None = None,
        selectable_history_len: int | None = None,
        forced_recent_count: int = 0,
    ) -> QueryAnalysis:
        history_len = len(history)
        selectable_len = history_len if selectable_history_len is None else max(
            0, min(int(selectable_history_len), history_len)
        )
        if self._is_obvious_chitchat(message):
            return QueryAnalysis(
                required_evidence=[],
                relevant_sops=[],
                history_pointers=self._default_history_pointers(
                    selectable_len, keep_recent=6
                ),
                keep_recent=min(6, selectable_len) if selectable_len > 0 else 0,
                reasoning="heuristic: obvious chitchat",
                is_chitchat=True,
            )

        user_prompt = self._build_user_prompt(
            message,
            history,
            message_timestamp,
            selectable_history_len=selectable_len,
            forced_recent_count=forced_recent_count,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _ANALYZER_SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": user_prompt},
        ]

        for _ in range(self._max_iterations):
            resp = await self._provider.chat(
                messages=messages,
                tools=self._tool_schemas,
                model=self._model,
                max_tokens=self._max_tokens,
                tool_choice="auto",
            )
            if resp.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": resp.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(
                                        tc.arguments, ensure_ascii=False
                                    ),
                                },
                            }
                            for tc in resp.tool_calls
                        ],
                    }
                )
                for tc in resp.tool_calls:
                    result = await self._execute_tool_call(tc.name, tc.arguments)
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": result}
                    )
                continue

            raw = self._parse_json(resp.content or "")
            if isinstance(raw, dict):
                return self._normalize(
                    raw,
                    message,
                    history_len=history_len,
                    selectable_history_len=selectable_len,
                )
            break

        logger.warning("[query_analyzer] fallback to default policy")
        return self._fallback_analysis(
            message,
            history_len=history_len,
            selectable_history_len=selectable_len,
        )

    def _build_user_prompt(
        self,
        message: str,
        history: list[dict[str, Any]],
        message_timestamp: datetime | None,
        selectable_history_len: int,
        forced_recent_count: int,
    ) -> str:
        tools = json.dumps(self._tool_schemas, ensure_ascii=False)
        all_tools = json.dumps(self._build_tool_summary(self._all_tool_schemas), ensure_ascii=False)
        history_index = self._build_history_index_guide(history)
        editable_docs = self._build_editable_docs_index()
        index_entries = self._build_index_entry_points()
        ts = self._format_time_anchor(message_timestamp)
        return (
            "请根据当前会话上下文和用户消息做 pre-flight 分析。\n\n"
            "## 本轮时间锚点（必须作为时间判断基准）\n"
            f"{ts}\n\n"
            "## 历史窗口说明\n"
            "你拿到的是“完整历史”（含最近对话），但最近尾部是主循环保底只读区，不属于 extra 候选。\n"
            f"- 可用于 extra 筛选的编号范围：1..{selectable_history_len}\n"
            f"- 主循环保底区（只读，不可指示为 extra）：{selectable_history_len + 1}..{len(history)}"
            f"（最近 {forced_recent_count} 条）\n\n"
            "## 目录索引入口（优先阅读）\n"
            f"{index_entries}\n\n"
            "## 工作区现状（实时派生）\n"
            f"{self._derive_workspace_state()}\n\n"
            "## 可编辑文档索引（用于规则/SOP修改）\n"
            f"{editable_docs}\n\n"
            "## 历史消息编号（1-based）\n"
            "主流程会用 history_pointers 在内存中拼装主模型上下文。\n"
            f"{history_index}\n\n"
            "## 可用工具 schema\n"
            f"{tools}\n\n"
            "## 主循环已注册工具（只读清单，用于理解工具能力边界）\n"
            f"{all_tools}\n\n"
            "## 用户消息\n"
            f"{message}\n\n"
            "你的输出主要用于“筛选旧上下文中的 extra context”，而不是替主循环决定完整执行计划。\n\n"
            "返回 JSON（无 markdown）：\n"
            "{\n"
            '  "required_evidence": [{"tool": "shell", "hint": "具体提示"}],\n'
            '  "relevant_sops": ["/abs/path/to/sop.md" 或 "novel-kb-query.md"],\n'
            '  "target_files": ["/abs/path/to/file.md"],\n'
            '  "history_pointers": [1, 4, 9],\n'
            '  "keep_recent": 8,\n'
            '  "is_chitchat": false,\n'
            '  "reasoning": "一句话"\n'
            "}\n"
        )

    def _build_index_entry_points(self) -> str:
        entries: list[str] = []
        sop_readme = self.workspace / "sop" / "README.md"
        skills_readme = self.workspace / "skills" / "README.md"
        if sop_readme.exists():
            try:
                content = sop_readme.read_text(encoding="utf-8")
                entries.append(f"SOP 索引（{sop_readme}）:\n{content}")
            except Exception:
                entries.append(f"- SOP 索引: {sop_readme}")
        if skills_readme.exists():
            entries.append(f"- Skills 索引: {skills_readme}")
        if not entries:
            return "（无 README 索引，按可编辑文档索引与目录结构判断）"
        return "\n".join(entries)

    @staticmethod
    def _format_time_anchor(message_timestamp: datetime | None) -> str:
        ts = message_timestamp
        if ts is None:
            ts = datetime.now().astimezone()
        elif ts.tzinfo is None:
            ts = ts.astimezone()
        return f"request_time={ts.isoformat()} ({ts.strftime('%Y-%m-%d %H:%M:%S %Z')})"

    async def _execute_tool_call(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._tool_names:
            return f"工具 {name} 不在 QueryAnalyzer 允许列表"
        try:
            return await self._tool_executor(name, arguments)
        except Exception as e:
            logger.warning("[query_analyzer] tool call failed name=%s err=%s", name, e)
            return f"工具执行失败: {e}"

    @staticmethod
    def _filter_tool_schemas(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for schema in schemas:
            name = schema.get("function", {}).get("name", "")
            if name in _ANALYZER_TOOL_ALLOWLIST:
                out.append(schema)
        return out

    @staticmethod
    def _build_history_index_guide(history: list[dict[str, Any]], max_rows: int = 500) -> str:
        if not history:
            return "（无历史）"
        lines: list[str] = []
        start = max(0, len(history) - max_rows)
        for idx in range(start, len(history)):
            msg = history[idx]
            role = str(msg.get("role", "unknown"))
            if role == "assistant" and msg.get("tool_calls"):
                tool_names = []
                for tc in msg.get("tool_calls") or []:
                    name = tc.get("function", {}).get("name", "")
                    if name:
                        tool_names.append(name)
                tag = f"assistant(tool_calls:{','.join(tool_names)})"
                content = str(msg.get("content", "") or "")
            elif role == "tool":
                tag = "tool"
                content = str(msg.get("content", "") or "")
            else:
                tag = role
                content = str(msg.get("content", "") or "")
            compact = content.replace("\n", " ").strip()
            if len(compact) > 70:
                compact = compact[:70] + "..."
            lines.append(f"{idx + 1}. [{tag}] {compact}")
        return "\n".join(lines)

    @staticmethod
    def _build_tool_summary(schemas: list[dict[str, Any]]) -> list[dict[str, str]]:
        summary: list[dict[str, str]] = []
        for s in schemas:
            fn = s.get("function", {})
            name = str(fn.get("name", "")).strip()
            if not name:
                continue
            desc = str(fn.get("description", "")).strip()
            summary.append({"name": name, "description": desc})
        return summary

    def _build_editable_docs_index(self) -> str:
        """动态构建可编辑文档索引，供 analyzer 判断修改落点。"""
        sections: list[tuple[str, list[str]]] = []

        def add_section(title: str, paths: list[Path]) -> None:
            valid = [str(p) for p in paths if p.exists() and p.is_file()]
            if valid:
                sections.append((title, sorted(valid)))

        # 1) 根文档：规则与身份描述
        add_section(
            "根文档",
            [
                self.workspace / "AGENTS.md",
                self.workspace / "SOUL.md",
                self.workspace / "USER.md",
            ],
        )

        # 2) SOP 文档
        sop_dir = self.workspace / "sop"
        sop_docs: list[Path] = sorted(sop_dir.glob("*.md")) if sop_dir.exists() else []
        add_section("SOP 文档", sop_docs)

        # 3) Memory 文档（用户长期画像/待办），排除历史与备份
        memory_dir = self.workspace / "memory"
        memory_docs: list[Path] = []
        if memory_dir.exists():
            for p in sorted(memory_dir.glob("*.md")):
                name = p.name.lower()
                if name.startswith("history"):
                    continue
                if ".bak" in name or "backup" in name:
                    continue
                memory_docs.append(p)
        add_section("Memory 文档", memory_docs)

        # 4) Skill 说明文档
        skills_dir = self.workspace / "skills"
        skill_docs: list[Path] = []
        if skills_dir.exists():
            for d in sorted(skills_dir.iterdir()):
                if not d.is_dir():
                    continue
                skill_md = d / "SKILL.md"
                if skill_md.exists():
                    skill_docs.append(skill_md)
        add_section("Skill 文档", skill_docs)

        # 5) 可人工编辑的配置（谨慎）
        add_section(
            "配置文件（谨慎）",
            [
                self.workspace / "feeds.json",
                self.workspace / "source_scores.json",
                self.workspace / "llm_config.json",
                self.workspace / "schedule.json",
                self.workspace / "schedules.json",
                self.workspace / "skill_actions.json",
            ],
        )

        if not sections:
            return "（无可编辑文档）"

        lines: list[str] = []
        for title, docs in sections:
            lines.append(f"{title}:")
            lines.extend(f"- {d}" for d in docs)
        lines.append(
            "排除运行态文件: sessions/*.jsonl, proactive_state.json, presence.json, *_state.json, proactive_quota.json"
        )
        return "\n".join(lines)

    def _derive_workspace_state(self) -> str:
        kb_dir = self.workspace / "kb"
        skills_dir = self.workspace / "skills"
        kb_ids: list[str] = []
        if kb_dir.exists():
            kb_ids = sorted(
                [d.name for d in kb_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
            )
        skills: list[str] = []
        if skills_dir.exists():
            skills = sorted([d.name for d in skills_dir.iterdir() if d.is_dir()])
        agents_path = self.workspace / "AGENTS.md"
        agents_text = ""
        if agents_path.exists():
            agents_text = agents_path.read_text(encoding="utf-8")[:800]
        return (
            f"知识库(kb/): {kb_ids}\n"
            f"技能(skills/): {skills}\n"
            "工作区说明(AGENTS.md 片段):\n"
            f"{agents_text or '（无）'}"
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        raw = (text or "").strip()
        if not raw:
            return None
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except Exception:
            try:
                obj = json_repair.loads(raw)
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None

    def _normalize(
        self,
        raw: dict[str, Any],
        message: str,
        history_len: int,
        selectable_history_len: int,
    ) -> QueryAnalysis:
        required: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in raw.get("required_evidence") or []:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool", "")).strip()
            hint = str(item.get("hint", "")).strip()
            if not tool and not hint:
                continue
            if not tool:
                tool = "read_file"
            if not hint:
                hint = f"先调用 {tool} 获取事实依据"
            key = (tool, hint)
            if key in seen:
                continue
            seen.add(key)
            required.append({"tool": tool, "hint": hint})

        raw_sops: list[str] = []
        for s in raw.get("relevant_sops") or []:
            if isinstance(s, str) and s.strip() and s.strip() not in raw_sops:
                raw_sops.append(s.strip())
        is_chitchat = bool(raw.get("is_chitchat", False))
        reasoning = str(raw.get("reasoning", "")).strip()
        keep_recent = self._sanitize_keep_recent(
            raw.get("keep_recent"), selectable_history_len
        )
        pointers = self._normalize_history_pointers(
            raw.get("history_pointers"),
            history_len=selectable_history_len,
            keep_recent=keep_recent,
        )
        target_files = self._normalize_target_files(raw.get("target_files"))

        required = self._normalize_required_evidence(required)

        # 只保留真实 SOP 文件路径，并做轻量归一化（兼容文件名/路径形式）
        valid_sops = self._list_workspace_sops()
        sops = self._normalize_relevant_sops(raw_sops, valid_sops)

        # SOP 相关任务：target_files 中的 SOP 文件始终补 read_file 提示（不受 required 是否为空影响）
        # 是否还需要 write_file 由 agent 根据用户意图自行判断，此处不强制添加
        seen_hints: set[tuple[str, str]] = {(e["tool"], e["hint"]) for e in required}
        for sop_path in sops:
            r_hint = f"读取 {sop_path} 了解并遵循对应规范。"
            if ("read_file", r_hint) not in seen_hints:
                required.append({"tool": "read_file", "hint": r_hint})
                seen_hints.add(("read_file", r_hint))
        for doc_path in target_files:
            if "/sop/" in doc_path and doc_path.endswith(".md") and not doc_path.endswith("README.md"):
                r_hint = f"读取 {doc_path} 了解相关规范或当前内容。"
                if ("read_file", r_hint) not in seen_hints:
                    required.append({"tool": "read_file", "hint": r_hint})
                    seen_hints.add(("read_file", r_hint))

        # 非 SOP 文档编辑任务：若 required 仍为空且有 target_files，补 read/write 提示
        if target_files and not required and not is_chitchat:
            primary = target_files[0]
            required.extend(
                [
                    {"tool": "read_file", "hint": f"先读取 {primary} 并定位需要改动的段落。"},
                    {"tool": "write_file", "hint": f"按用户要求修改并写回 {primary}。"},
                ]
            )

        # 轻分析模式：不再对 required_evidence 做强制补齐，保持“建议”属性

        return QueryAnalysis(
            required_evidence=required,
            relevant_sops=sops,
            history_pointers=pointers,
            keep_recent=keep_recent,
            target_files=target_files,
            reasoning=reasoning,
            is_chitchat=is_chitchat,
        )

    def _fallback_analysis(
        self,
        message: str,
        history_len: int,
        selectable_history_len: int,
    ) -> QueryAnalysis:
        keep_recent = min(8, selectable_history_len) if selectable_history_len > 0 else 0
        pointers = self._default_history_pointers(
            selectable_history_len, keep_recent=keep_recent
        )
        if self._is_obvious_chitchat(message):
            return QueryAnalysis(
                required_evidence=[],
                relevant_sops=[],
                history_pointers=pointers,
                keep_recent=keep_recent,
                reasoning="fallback: obvious chitchat",
                is_chitchat=True,
            )
        required: list[dict[str, str]] = []
        return QueryAnalysis(
            required_evidence=required,
            relevant_sops=[],
            history_pointers=pointers,
            keep_recent=keep_recent,
            target_files=[],
            reasoning="fallback: non-chitchat requires evidence",
            is_chitchat=False,
        )

    def _normalize_relevant_sops(
        self,
        raw_sops: list[str],
        valid_sops: set[str],
    ) -> list[str]:
        if not raw_sops or not valid_sops:
            return []

        by_token: dict[str, str] = {}
        by_name: dict[str, str] = {}
        for path in valid_sops:
            p = Path(path)
            name = p.name
            stem = p.stem
            # 文件名与 stem 两种 token 都支持，便于模型给出简写。
            by_token[self._norm_token(name)] = path
            by_token[self._norm_token(stem)] = path
            by_name[name] = path

        out: list[str] = []
        seen: set[str] = set()
        for item in raw_sops:
            candidate = item.strip()
            mapped: str | None = None
            if not candidate:
                continue

            # 1) 绝对路径直接命中
            if candidate in valid_sops:
                mapped = candidate
            else:
                # 2) 支持 "~" 路径与路径归一化
                try:
                    expanded = str(Path(candidate).expanduser().resolve())
                except Exception:
                    expanded = candidate
                if expanded in valid_sops:
                    mapped = expanded
                else:
                    # 3) 文件名/简写匹配（novel-kb-query.md / novel-kb-query）
                    name = Path(candidate).name
                    mapped = by_name.get(name) or by_token.get(self._norm_token(name))
                    if not mapped:
                        mapped = by_token.get(self._norm_token(candidate))
            if mapped and mapped not in seen:
                seen.add(mapped)
                out.append(mapped)
        return out

    @staticmethod
    def _norm_token(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text.lower())

    @staticmethod
    def _normalize_required_evidence(
        items: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in items:
            tool = str(item.get("tool", "")).strip() or "read_file"
            hint = str(item.get("hint", "")).strip() or f"先调用 {tool} 获取事实依据"
            key = (tool, hint)
            if key in seen:
                continue
            seen.add(key)
            out.append({"tool": tool, "hint": hint})
        return out

    @staticmethod
    def _sanitize_keep_recent(value: Any, history_len: int) -> int:
        if history_len <= 0:
            return 0
        default = min(8, history_len)
        try:
            v = int(value)
        except Exception:
            return default
        return max(0, min(v, min(20, history_len)))

    @staticmethod
    def _normalize_history_pointers(
        value: Any,
        history_len: int,
        keep_recent: int,
    ) -> list[int]:
        if history_len <= 0:
            return []
        out: list[int] = []
        seen: set[int] = set()
        if isinstance(value, list):
            for item in value:
                idx: int | None = None
                if isinstance(item, int):
                    if 1 <= item <= history_len:
                        idx = item - 1  # 1-based -> 0-based
                    elif 0 <= item < history_len:
                        idx = item  # backward compatible
                elif isinstance(item, str) and item.strip().isdigit():
                    p = int(item.strip())
                    if 1 <= p <= history_len:
                        idx = p - 1
                if idx is not None and idx not in seen:
                    seen.add(idx)
                    out.append(idx)
        if out:
            return sorted(out)
        return QueryAnalyzer._default_history_pointers(history_len, keep_recent)

    @staticmethod
    def _default_history_pointers(history_len: int, keep_recent: int = 8) -> list[int]:
        if history_len <= 0:
            return []
        tail = max(1, min(keep_recent, history_len))
        start = history_len - tail
        return list(range(start, history_len))

    def _normalize_target_files(self, value: Any) -> list[str]:
        allowed = self._editable_doc_paths()
        if not allowed:
            return []
        out: list[str] = []
        seen: set[str] = set()
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, str):
                    continue
                path = item.strip()
                if not path or path in seen:
                    continue
                if path in allowed:
                    seen.add(path)
                    out.append(path)
        return out

    @staticmethod
    def _is_obvious_chitchat(message: str) -> bool:
        msg = message.strip().lower()
        if not msg:
            return True
        pure = re.sub(r"[!?.,，。！？~\s]+", "", msg)
        greetings = {
            "hi",
            "hello",
            "hey",
            "yo",
            "你好",
            "在吗",
            "晚安",
            "早安",
            "哈哈",
            "hhh",
        }
        if pure in greetings:
            return True
        return len(pure) <= 4 and pure in greetings

    def _list_workspace_sops(self) -> set[str]:
        docs: set[str] = set()
        sop_dir = self.workspace / "sop"
        if not sop_dir.exists():
            return docs
        for p in sop_dir.glob("*.md"):
            if not p.is_file():
                continue
            if p.name.lower() == "readme.md":
                continue
            try:
                docs.add(str(p.resolve()))
            except Exception:
                docs.add(str(p))
        return docs

    def _editable_doc_paths(self) -> set[str]:
        docs: set[str] = set()

        for p in [self.workspace / "AGENTS.md", self.workspace / "SOUL.md", self.workspace / "USER.md"]:
            if p.exists() and p.is_file():
                docs.add(str(p))

        sop_dir = self.workspace / "sop"
        if sop_dir.exists():
            for p in sop_dir.glob("*.md"):
                if p.is_file():
                    docs.add(str(p))

        memory_dir = self.workspace / "memory"
        if memory_dir.exists():
            for p in memory_dir.glob("*.md"):
                if not p.is_file():
                    continue
                name = p.name.lower()
                if name.startswith("history") or ".bak" in name or "backup" in name:
                    continue
                docs.add(str(p))

        skills_dir = self.workspace / "skills"
        if skills_dir.exists():
            for d in skills_dir.iterdir():
                if not d.is_dir():
                    continue
                p = d / "SKILL.md"
                if p.exists() and p.is_file():
                    docs.add(str(p))

        for p in [
            self.workspace / "feeds.json",
            self.workspace / "source_scores.json",
            self.workspace / "llm_config.json",
            self.workspace / "schedule.json",
            self.workspace / "schedules.json",
            self.workspace / "skill_actions.json",
        ]:
            if p.exists() and p.is_file():
                docs.add(str(p))
        return docs
