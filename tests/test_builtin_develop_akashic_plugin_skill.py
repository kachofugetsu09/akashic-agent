import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

from agent.skills import SkillsLoader

REPO_ROOT = Path(__file__).parents[1]
SKILL_ROOT = REPO_ROOT / "skills" / "develop-akashic-plugin"


def test_develop_akashic_plugin_is_discoverable_builtin(tmp_path: Path) -> None:
    loader = SkillsLoader(tmp_path, builtin_skills_dir=REPO_ROOT / "skills")

    record = loader.load_skill_record("develop-akashic-plugin")

    assert record is not None
    assert record.source == "builtin"
    assert record.available is True
    assert record.always is False
    for trigger in (
        "创建",
        "编写",
        "验证插件",
        "验证 skill",
        "递归自验证",
    ):
        assert trigger in record.description


def test_develop_akashic_plugin_preserves_validation_contract(tmp_path: Path) -> None:
    loader = SkillsLoader(tmp_path, builtin_skills_dir=REPO_ROOT / "skills")
    body = loader.load_skill_body("develop-akashic-plugin")

    assert body is not None
    for contract in (
        "canonical source",
        "不要直接编辑 `~/.akashic-plugin/cache`",
        "不要指定 `--runtime`",
        "默认不沉淀语义记忆",
        "不能只问“你能否看到”",
        "message_push",
        "attached child",
        "safe candidate self-validation unavailable",
        "才告诉用户任务完成",
    ):
        assert contract in body


def test_develop_akashic_plugin_references_are_complete() -> None:
    body = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    authoring = (SKILL_ROOT / "references" / "plugin-authoring.md").read_text(
        encoding="utf-8"
    )
    validation = (SKILL_ROOT / "references" / "self-validation.md").read_text(
        encoding="utf-8"
    )
    diagnostics = (SKILL_ROOT / "references" / "runtime-diagnostics.md").read_text(
        encoding="utf-8"
    )

    assert "references/plugin-authoring.md" in body
    assert "references/self-validation.md" in body
    assert "references/runtime-diagnostics.md" in body
    for contract in (
        "@tool",
        "@on_prompt_render(priority=100)",
        "PromptSectionRender",
        ".venv/bin/python",
        "普通实例方法",
        "skill_roots()",
        "SkillsLoader",
        "plugin-doctor",
    ):
        assert contract in authoring
    for contract in (
        "成功前不要预先做全量诊断考古",
        "不要创建空 `requirements.txt`",
        "source test → commit → install",
    ):
        assert contract in body
    for contract in (
        "plugin-install",
        "不得添加 `--runtime latest`",
        "write_stdin",
        "plugin-revert",
        "validation_port_env",
        "semantic write set == 0",
        "非 read-only Tool/MCP 默认禁用",
        "目标渠道是否另写自己的 durable event",
    ):
        assert contract in validation
    for contract in (
        "mode=ro",
        "items_json",
        "llm_context_frame",
        "tool_chain",
        "runtime log unavailable: stderr is tty",
        "plugin_prompt_probe",
        "支持 `T await V → T 根据结果继续修改`",
    ):
        assert contract in diagnostics


def test_runtime_diagnostic_script_reads_turn_messages_and_reload(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    runtime_dir = workspace / "runtime"
    runtime_dir.mkdir(parents=True)
    with closing(sqlite3.connect(workspace / "sessions.db")) as sessions:
        sessions.executescript("""
            CREATE TABLE sessions (
                key TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_consolidated INTEGER NOT NULL DEFAULT 0,
                metadata TEXT,
                next_seq INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE turns (
                id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                status TEXT NOT NULL,
                input_json TEXT NOT NULL,
                items_json TEXT NOT NULL,
                usage_json TEXT,
                error_json TEXT,
                final_response TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            );
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_chain TEXT,
                extra TEXT,
                ts TEXT NOT NULL
            );
            """)
        sessions.execute(
            "INSERT INTO sessions (key, created_at, updated_at, metadata) VALUES (?, ?, ?, ?)",
            (
                "programmatic:probe",
                "2026-08-06T00:00:00+00:00",
                "2026-08-06T00:00:02+00:00",
                '{"skip_post_memory":true}',
            ),
        )
        sessions.execute(
            """
            INSERT INTO turns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "turn:probe",
                "programmatic:probe",
                "completed",
                '{"input":"probe","metadata":{}}',
                '[{"id":"item:1","type":"assistantMessage","data":{"content":"ok"}}]',
                '{"inputTokens":1}',
                None,
                "ok",
                "2026-08-06T00:00:00+00:00",
                "2026-08-06T00:00:01+00:00",
                "2026-08-06T00:00:02+00:00",
            ),
        )
        sessions.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "message:1",
                "programmatic:probe",
                0,
                "user",
                "probe",
                None,
                '{"llm_context_frame":"frame"}',
                "2026-08-06T00:00:01+00:00",
            ),
        )
        sessions.commit()
    with closing(sqlite3.connect(runtime_dir / "plugin-reloads.sqlite3")) as reloads:
        reloads.executescript("""
            CREATE TABLE reload_transactions (
                tx_id TEXT PRIMARY KEY,
                plugin_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                started_at TEXT NOT NULL,
                error TEXT NOT NULL
            );
            CREATE TABLE reload_events (
                sequence INTEGER PRIMARY KEY,
                tx_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """)
        reloads.execute(
            "INSERT INTO reload_transactions VALUES (?, ?, ?, ?, ?)",
            (
                "tx:probe",
                "probe@local",
                "complete",
                "2026-08-06T00:00:00+00:00",
                "secret token",
            ),
        )
        reloads.execute(
            "INSERT INTO reload_events VALUES (?, ?, ?, ?, ?)",
            (
                1,
                "tx:probe",
                "complete",
                '{"snapshot":"latest"}',
                "2026-08-06T00:00:02+00:00",
            ),
        )
        reloads.commit()

    completed = subprocess.run(
        [
            sys.executable,
            str(SKILL_ROOT / "scripts" / "inspect-runtime-trace.py"),
            "--workspace",
            str(workspace),
            "--turn-id",
            "turn:probe",
            "--plugin-id",
            "probe@local",
            "--include-content",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    assert report["turn"]["final_response"] == "ok"
    assert report["turn"]["items"][0]["type"] == "assistantMessage"
    assert report["messages"][0]["extra"]["llm_context_frame"] == "frame"
    assert report["plugin_reload"]["phase"] == "complete"
    assert report["plugin_reload"]["error"] == "secret token"
    assert report["plugin_reload"]["events"][0]["details"] == {"snapshot": "latest"}

    redacted = subprocess.run(
        [
            sys.executable,
            str(SKILL_ROOT / "scripts" / "inspect-runtime-trace.py"),
            "--workspace",
            str(workspace),
            "--turn-id",
            "turn:probe",
            "--plugin-id",
            "probe@local",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(redacted.stdout)
    assert "final_response" not in summary["turn"]
    assert summary["turn"]["final_response_summary"] == {
        "chars": 4,
        "type": "str",
    }
    assert "content" not in summary["messages"][0]
    assert summary["messages"][0]["content_summary"] == {
        "chars": 7,
        "type": "str",
    }
    assert "details" not in summary["plugin_reload"]["events"][0]
    assert "error" not in summary["plugin_reload"]
    assert summary["plugin_reload"]["error_summary"] == {
        "chars": 14,
        "type": "str",
    }


def test_plugin_system_routes_source_development_to_new_skill(tmp_path: Path) -> None:
    loader = SkillsLoader(tmp_path, builtin_skills_dir=REPO_ROOT / "skills")
    body = loader.load_skill_body("plugin-system")

    assert body is not None
    assert "先加载 `develop-akashic-plugin`" in body
