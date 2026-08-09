from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from agent.plugins.manager import ActivePluginInfo
from agent.plugins.skill_links import PluginSkillLinker
from agent.skills import SkillsLoader
from plugins.drift_flow.state import DriftStateStore


def _write_plugin_skill(
    plugin_root: Path,
    plugin_id: str,
    skill_name: str,
    *,
    body: str = "plugin skill body",
) -> Path:
    skill_dir = plugin_root / plugin_id / "skills" / skill_name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {skill_name}\n"
        "description: 插件技能\n"
        "---\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return plugin_root / plugin_id


def _write_plugin_drift_skill(
    plugin_root: Path,
    plugin_id: str,
    skill_name: str,
    *,
    body: str = "plugin drift skill body",
) -> Path:
    skill_dir = plugin_root / plugin_id / "drift" / "skills" / skill_name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {skill_name}\n"
        "description: 插件 Drift 技能\n"
        "---\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return plugin_root / plugin_id


def _plugin_info(
    plugin_id: str,
    plugin_dir: Path,
    manifest: dict[str, object] | None = None,
) -> ActivePluginInfo:
    return ActivePluginInfo(
        plugin_id=plugin_id,
        plugin_dir=plugin_dir,
        manifest=manifest or {},
        module_path=f"test_{plugin_id}",
        skill_roots=((plugin_dir / "skills",) if (plugin_dir / "skills").is_dir() else ()),
        drift_skill_roots=(
            (plugin_dir / "drift" / "skills",)
            if (plugin_dir / "drift" / "skills").is_dir()
            else ()
        ),
    )


def _memory_engine(name: str) -> object:
    return SimpleNamespace(describe=lambda: SimpleNamespace(name=name))


def test_plugin_skill_linker_creates_workspace_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    plugin_dir = _write_plugin_skill(plugin_root, "foo", "bar")

    result = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=None,
    ).sync([_plugin_info("foo", plugin_dir)])

    link = workspace / "skills" / "bar"
    assert result.expected == 1
    assert result.created == 1
    assert link.is_symlink()
    loader = SkillsLoader(workspace, builtin_skills_dir=tmp_path / "builtin")
    assert loader.load_skill_body("bar") == "plugin skill body"


def test_plugin_skill_linker_removes_stale_link(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    plugin_dir = _write_plugin_skill(plugin_root, "foo", "bar")
    linker = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=None,
    )
    linker.sync([_plugin_info("foo", plugin_dir)])

    result = linker.sync([])

    assert result.removed == 1
    assert not (workspace / "skills" / "bar").exists()


def test_plugin_skill_linker_preserves_unowned_broken_plugin_link(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    skills_dir = workspace / "skills"
    skills_dir.mkdir(parents=True)
    link = skills_dir / "gone:bar"
    link.symlink_to(plugin_root / "gone" / "skills" / "bar", target_is_directory=True)

    result = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=None,
    ).sync([])

    assert result.removed == 0
    assert link.is_symlink()


def test_plugin_skill_linker_rejects_user_skill_dir_without_deleting_it(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    plugin_dir = _write_plugin_skill(plugin_root, "foo", "bar")
    user_skill = workspace / "skills" / "bar"
    user_skill.mkdir(parents=True)
    (user_skill / "SKILL.md").write_text("user body", encoding="utf-8")

    with pytest.raises(RuntimeError, match="用户文件或目录冲突"):
        PluginSkillLinker(
            workspace=workspace,
            plugin_roots=[plugin_root],
            memory_engine=None,
        ).sync([_plugin_info("foo", plugin_dir)])

    assert user_skill.is_dir()
    assert not user_skill.is_symlink()
    assert (user_skill / "SKILL.md").read_text(encoding="utf-8") == "user body"


def test_plugin_skill_linker_rejects_user_symlink_without_replacing_it(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    plugin_dir = _write_plugin_skill(plugin_root, "foo", "bar")
    user_target = tmp_path / "personal" / "bar"
    user_target.mkdir(parents=True)
    (user_target / "SKILL.md").write_text("user body", encoding="utf-8")
    user_link = workspace / "skills" / "bar"
    user_link.parent.mkdir(parents=True)
    user_link.symlink_to(user_target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="用户软链接冲突"):
        PluginSkillLinker(
            workspace=workspace,
            plugin_roots=[plugin_root],
            memory_engine=None,
        ).sync([_plugin_info("foo", plugin_dir)])

    assert user_link.is_symlink()
    assert user_link.resolve() == user_target


def test_plugin_skill_linker_repairs_only_managed_plugin_symlink(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    old_plugin_dir = _write_plugin_skill(plugin_root, "old", "bar", body="old")
    linker = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=None,
    )
    linker.sync([_plugin_info("old", old_plugin_dir)])
    plugin_dir = _write_plugin_skill(plugin_root, "foo", "bar")
    result = linker.sync([_plugin_info("foo", plugin_dir)])

    link = workspace / "skills" / "bar"
    assert result.repaired == 1
    assert link.resolve() == plugin_dir / "skills" / "bar"


def test_plugin_skill_linker_recovers_crash_before_symlink_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    old_dir = _write_plugin_skill(plugin_root, "old", "bar", body="old")
    new_dir = _write_plugin_skill(plugin_root, "new", "bar", body="new")
    linker = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=None,
    )
    linker.sync([_plugin_info("old", old_dir)])
    link = workspace / "skills" / "bar"

    monkeypatch.setattr(
        linker,
        "_replace_link",
        lambda _link, _target: (_ for _ in ()).throw(SystemExit("crash")),
    )
    with pytest.raises(SystemExit, match="crash"):
        linker.sync([_plugin_info("new", new_dir)])

    recovered = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=None,
    )
    assert link.resolve() == old_dir / "skills" / "bar"
    result = recovered.sync([_plugin_info("new", new_dir)])
    assert result.repaired == 1
    assert link.resolve() == new_dir / "skills" / "bar"


def test_plugin_skill_linker_recovers_crash_after_symlink_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    old_dir = _write_plugin_skill(plugin_root, "old", "bar", body="old")
    new_dir = _write_plugin_skill(plugin_root, "new", "bar", body="new")
    linker = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=None,
    )
    linker.sync([_plugin_info("old", old_dir)])
    link = workspace / "skills" / "bar"

    monkeypatch.setattr(
        linker,
        "_commit_transition",
        lambda _key, _target: (_ for _ in ()).throw(SystemExit("crash")),
    )
    with pytest.raises(SystemExit, match="crash"):
        linker.sync([_plugin_info("new", new_dir)])
    assert link.resolve() == new_dir / "skills" / "bar"

    recovered = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=None,
    )
    result = recovered.sync([_plugin_info("new", new_dir)])
    assert result.repaired == 0
    assert link.resolve() == new_dir / "skills" / "bar"


def test_plugin_skill_linker_rolls_back_after_final_ownership_save_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    old_dir = _write_plugin_skill(plugin_root, "old", "bar", body="old")
    new_dir = _write_plugin_skill(plugin_root, "new", "bar", body="new")
    linker = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=None,
    )
    linker.sync([_plugin_info("old", old_dir)])
    link = workspace / "skills" / "bar"
    real_write = linker._write_ownership
    write_count = 0

    def fail_final_write(owned_links, pending_links) -> None:
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise OSError("simulated final ownership save failure")
        real_write(owned_links, pending_links)

    monkeypatch.setattr(linker, "_write_ownership", fail_final_write)
    with pytest.raises(OSError, match="final ownership save failure"):
        linker.sync([_plugin_info("new", new_dir)])
    assert link.resolve() == new_dir / "skills" / "bar"

    rollback = linker.sync([_plugin_info("old", old_dir)])
    assert rollback.repaired == 1
    assert link.resolve() == old_dir / "skills" / "bar"
    recovered = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=None,
    )
    assert recovered.sync([_plugin_info("old", old_dir)]).repaired == 0


def test_plugin_skill_linker_does_not_adopt_user_link_into_plugin_root(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    plugin_dir = _write_plugin_skill(plugin_root, "foo", "bar")
    user_link = workspace / "skills" / "bar"
    user_link.parent.mkdir(parents=True)
    user_link.symlink_to(plugin_dir / "skills" / "bar", target_is_directory=True)

    with pytest.raises(RuntimeError, match="用户软链接冲突"):
        PluginSkillLinker(
            workspace=workspace,
            plugin_roots=[plugin_root],
            memory_engine=None,
        ).sync([_plugin_info("foo", plugin_dir)])

    assert user_link.is_symlink()
    assert user_link.resolve() == plugin_dir / "skills" / "bar"


def test_plugin_skill_linker_does_not_interpret_runtime_policy(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    plugin_dir = _write_plugin_skill(plugin_root, "akasha", "memory")
    manifest: dict[str, object] = {
        "skills": {
            "enabled_when": {
                "kind": "memory_engine",
                "engine": "akasha",
            }
        }
    }
    plugin = _plugin_info("akasha", plugin_dir, manifest)

    disabled = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=_memory_engine("default"),
    ).sync([plugin])
    enabled = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=_memory_engine("akasha"),
    ).sync([plugin])
    removed = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=_memory_engine("default"),
    ).sync([plugin])

    assert disabled.expected == 1
    assert enabled.expected == 1
    assert removed.removed == 0
    assert (workspace / "skills" / "memory").is_symlink()


def test_aka_plugin_skill_is_exposed_with_bare_name(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    cache_root = tmp_path / "cache"
    plugin_dir = cache_root / "lab" / "feed" / "0.1.0"
    skill_dir = plugin_dir / "skills" / "feed-manage"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: feed-manage\n"
        "description: feed skill\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )
    plugin = ActivePluginInfo(
        plugin_id="feed@lab",
        plugin_dir=plugin_dir,
        manifest={},
        module_path="feed",
        skill_roots=(plugin_dir / "skills",),
    )

    result = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[cache_root],
        memory_engine=None,
    ).sync([plugin])

    assert result.expected == 1
    assert (workspace / "skills" / "feed-manage").is_symlink()
    assert not (workspace / "skills" / "feed@lab:feed-manage").exists()


def test_aka_plugin_skill_sync_preserves_unowned_old_prefixed_link(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    cache_root = tmp_path / "cache"
    plugin_dir = cache_root / "lab" / "feed" / "0.1.0"
    skill_dir = plugin_dir / "skills" / "feed-manage"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: feed-manage\n"
        "description: feed skill\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )
    old_link = workspace / "skills" / "feed@lab:feed-manage"
    old_link.parent.mkdir(parents=True)
    old_link.symlink_to(skill_dir, target_is_directory=True)
    plugin = ActivePluginInfo(
        plugin_id="feed@lab",
        plugin_dir=plugin_dir,
        manifest={},
        module_path="feed",
        skill_roots=(plugin_dir / "skills",),
    )

    result = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[cache_root],
        memory_engine=None,
    ).sync([plugin])

    assert result.created == 1
    assert result.removed == 0
    assert (workspace / "skills" / "feed-manage").is_symlink()
    assert old_link.is_symlink()


def test_aka_plugin_drift_skill_uses_bare_plugin_name(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    cache_root = tmp_path / "cache"
    plugin_dir = cache_root / "github" / "emotion" / "0.1.0"
    skill_dir = plugin_dir / "drift" / "skills" / "feedback-preference-context"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: feedback-preference-context\n"
        "description: drift skill\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )
    plugin = ActivePluginInfo(
        plugin_id="emotion@github",
        plugin_dir=plugin_dir,
        manifest={},
        module_path="emotion",
        drift_skill_roots=(plugin_dir / "drift" / "skills",),
    )

    result = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[cache_root],
        memory_engine=None,
    ).sync([plugin])

    assert result.expected == 1
    assert (workspace / "drift" / "skills" / "feedback-preference-context").is_symlink()
    assert not (workspace / "drift" / "skills" / "emotion:feedback-preference-context").exists()
    assert not (workspace / "drift" / "skills" / "emotion@github:feedback-preference-context").exists()


def test_plugin_drift_skill_linker_creates_workspace_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    plugin_dir = _write_plugin_drift_skill(plugin_root, "foo", "daily")

    result = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=None,
    ).sync([_plugin_info("foo", plugin_dir)])

    link = workspace / "drift" / "skills" / "daily"
    store = DriftStateStore(workspace / "drift")
    skills = store.scan_skills()
    skill_dir = store.skill_dir_for("daily")

    assert result.expected == 1
    assert result.created == 1
    assert link.is_symlink()
    assert {skill.name for skill in skills} == {"daily"}
    assert skill_dir is not None
    assert skill_dir == link
    assert "plugin drift skill body" in (skill_dir / "SKILL.md").read_text(encoding="utf-8")


def test_plugin_drift_skill_linker_removes_stale_link(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    plugin_dir = _write_plugin_drift_skill(plugin_root, "foo", "daily")
    linker = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=None,
    )
    linker.sync([_plugin_info("foo", plugin_dir)])

    result = linker.sync([])

    assert result.removed == 1
    assert not (workspace / "drift" / "skills" / "daily").exists()


def test_plugin_drift_skill_linker_rejects_user_skill_dir_without_deleting_it(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    plugin_dir = _write_plugin_drift_skill(plugin_root, "foo", "daily")
    user_skill = workspace / "drift" / "skills" / "daily"
    user_skill.mkdir(parents=True)
    (user_skill / "SKILL.md").write_text("user body", encoding="utf-8")

    with pytest.raises(RuntimeError, match="用户文件或目录冲突"):
        PluginSkillLinker(
            workspace=workspace,
            plugin_roots=[plugin_root],
            memory_engine=None,
        ).sync([_plugin_info("foo", plugin_dir)])

    assert user_skill.is_dir()
    assert not user_skill.is_symlink()
    assert (user_skill / "SKILL.md").read_text(encoding="utf-8") == "user body"


def test_plugin_drift_skill_linker_does_not_interpret_runtime_policy(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    plugin_dir = _write_plugin_drift_skill(plugin_root, "akasha", "daily")
    manifest: dict[str, object] = {
        "drift_skills": {
            "enabled_when": {
                "kind": "memory_engine",
                "engine": "akasha",
            }
        }
    }
    plugin = _plugin_info("akasha", plugin_dir, manifest)

    disabled = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=_memory_engine("default"),
    ).sync([plugin])
    enabled = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=_memory_engine("akasha"),
    ).sync([plugin])

    assert disabled.expected == 1
    assert enabled.expected == 1
    assert (workspace / "drift" / "skills" / "daily").is_symlink()


def test_default_memory_declared_drift_skill_is_linked_when_plugin_is_active(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = Path(__file__).parents[1] / "plugins"
    plugin_dir = plugin_root / "default_memory"
    plugin = _plugin_info("default_memory", plugin_dir)
    link = workspace / "drift" / "skills" / "audit-dirty-memories"
    enabled = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=_memory_engine("default"),
    ).sync([plugin])
    skills = DriftStateStore(workspace / "drift").scan_skills()

    assert enabled.expected == 1
    assert link.is_symlink()
    assert {skill.name for skill in skills} == {"audit-dirty-memories"}

    removed = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=_memory_engine("akasha"),
    ).sync([])

    assert removed.removed == 1
    assert not link.is_symlink()


def test_emotion_feedback_drift_skill_is_exposed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    cache_root = tmp_path / "cache"
    plugin_dir = cache_root / "github" / "emotion" / "0.1.0"
    skill_dir = plugin_dir / "drift" / "skills" / "feedback-preference-context"
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: feedback-preference-context\n"
        "description: 情绪反馈 drift skill\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )
    (scripts_dir / "sample_feedback_context.py").write_text(
        "print('ok')\n",
        encoding="utf-8",
    )
    plugin = ActivePluginInfo(
        plugin_id="emotion@github",
        plugin_dir=plugin_dir,
        manifest={},
        module_path="emotion",
        drift_skill_roots=(plugin_dir / "drift" / "skills",),
    )

    result = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[cache_root],
        memory_engine=None,
    ).sync([plugin])
    skills = DriftStateStore(workspace / "drift").scan_skills()
    linked_skill_dir = DriftStateStore(workspace / "drift").skill_dir_for(
        "feedback-preference-context"
    )

    assert result.expected >= 1
    assert (workspace / "drift" / "skills" / "feedback-preference-context").is_symlink()
    assert "feedback-preference-context" in {skill.name for skill in skills}
    assert linked_skill_dir is not None
    assert (
        linked_skill_dir / "scripts" / "sample_feedback_context.py"
    ).exists()


def test_default_memory_audit_script_uses_drift_journal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    drift_dir = workspace / "drift"
    memory_dir = workspace / "memory"
    memory_dir.mkdir(parents=True)
    DriftStateStore(drift_dir)
    memory_db = memory_dir / "memory2.db"
    conn = sqlite3.connect(memory_db)
    try:
        _ = conn.execute(
            """
            CREATE TABLE memory_items (
                id TEXT PRIMARY KEY,
                memory_type TEXT,
                summary TEXT,
                source_ref TEXT,
                happened_at TEXT,
                status TEXT
            )
            """
        )
        _ = conn.executemany(
            """
            INSERT INTO memory_items (
                id, memory_type, summary, source_ref, happened_at, status
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                ("m1", "fact", "用户喜欢极简实现", "telegram:1", "2026-07-01", "active"),
                ("m2", "fact", "用户正在调 Drift", "telegram:2", "2026-07-02", "active"),
                ("m3", "fact", "旧 post response", "telegram:3@post_response", "2026-07-03", "active"),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    script = (
        Path(__file__).parents[1]
        / "plugins"
        / "default_memory"
        / "drift"
        / "skills"
        / "audit-dirty-memories"
        / "scripts"
        / "sample_memory_for_audit.py"
    )
    sampled = _run_audit_script(script, "sample", "--drift-dir", str(drift_dir))
    item = cast(dict[str, object], sampled["item"])
    memory_id = str(item["id"])
    store = DriftStateStore(drift_dir)
    store.save_finish(
        skill_used="default_memory:audit-dirty-memories",
        status="completed",
        briefing=f"审计 memory_id={memory_id}，结果 clean",
        message_result="silent",
        scratchpad_update="下次继续随机抽样。",
        global_note_update=None,
        now_utc=datetime.now(timezone.utc),
        cursor_update={
            "next_action": "sample",
            "active_memory_id": None,
        },
        journal_append=[
            {
                "entry_type": "memory_audited",
                "key": memory_id,
                "payload": {"result": "clean"},
            }
        ],
    )
    sampled_again = _run_audit_script(script, "sample", "--drift-dir", str(drift_dir))
    journal = _load_audit_journal(drift_dir / "drift.db")
    cursor = _load_audit_cursor(drift_dir / "drift.db")

    assert sampled["found"] is True
    assert memory_id in {"m1", "m2"}
    required = cast(dict[str, object], sampled["journal_append_required"])
    assert required["entry_type"] == "memory_audited"
    if sampled_again["found"] is True:
        next_item = cast(dict[str, object], sampled_again["item"])
        assert str(next_item["id"]) != memory_id
    assert journal[memory_id]["result"] == "clean"
    assert cursor["next_action"] == "sample"
    assert not (
        drift_dir
        / "skill_state"
        / "default_memory:audit-dirty-memories"
        / "history.json"
    ).exists()


def _run_audit_script(script: Path, *args: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(script), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = json.loads(result.stdout)
    return cast(dict[str, object], loaded)


def _load_audit_journal(drift_db: Path) -> dict[str, dict[str, object]]:
    conn = sqlite3.connect(drift_db)
    try:
        rows = conn.execute(
            """
            SELECT key, payload_json
            FROM skill_journal
            WHERE skill_name = 'default_memory:audit-dirty-memories'
              AND entry_type = 'memory_audited'
            """
        ).fetchall()
    finally:
        conn.close()
    return {
        str(row[0]): cast(dict[str, object], json.loads(str(row[1] or "{}")))
        for row in rows
    }


def _load_audit_cursor(drift_db: Path) -> dict[str, object]:
    conn = sqlite3.connect(drift_db)
    try:
        row = conn.execute(
            """
            SELECT cursor_json
            FROM skill_continuum
            WHERE skill_name = 'default_memory:audit-dirty-memories'
            """
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    loaded = json.loads(str(row[0] or "{}"))
    return cast(dict[str, object], loaded)
