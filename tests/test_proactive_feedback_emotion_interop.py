"""通过正式 PluginManager 证明 PF 与 Emotion 的真实 Message 组合。"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web

from agent.config_models import Config
from agent.plugin_contracts import ContentPart, Input, Output
from agent.plugins.install import PluginInstallResult, install_git_plugin
from agent.plugins.model_control import RuntimeModelControl
from agent.plugins.snapshot import lease_runtime_snapshot
from bootstrap.init_workspace import init_workspace
from bootstrap.tools import build_core_runtime
from core.net.http import SharedHttpResources
from plugins.content.plugin import CONTENT
from session.log import SessionAttributes
from tests.fixtures.formal_plugins import install_formal_plugins


_PF_HEAD = "b0dd6dd1a14852e0e5df3c2459f60f1bc80c99f4"
_EMOTION_HEAD = "99e4acc1b656c63c818b741f533dfb4de9299ef6"
_BUILTIN_PLUGINS = (
    "content",
    "context",
    "drift",
    "models",
    "openai_compatible",
    "tools",
    "turn_projection",
)


def _plugin_roots() -> dict[str, Path]:
    raw = os.environ.get("AKASHIC_INTEROP_PLUGIN_ROOTS")
    if raw is None:
        pytest.skip("proactive source interop Gate supplies exact external roots")
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise RuntimeError("AKASHIC_INTEROP_PLUGIN_ROOTS 必须是对象")
    roots = {str(key): Path(str(value)) for key, value in decoded.items()}
    if set(roots) != {"proactive_feedback", "emotion"}:
        raise RuntimeError(f"interop plugin roots 不匹配: {sorted(roots)}")
    expected = {"proactive_feedback": _PF_HEAD, "emotion": _EMOTION_HEAD}
    for plugin_id, root in roots.items():
        actual = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
        ).strip()
        if actual != expected[plugin_id]:
            raise RuntimeError(
                f"{plugin_id} source SHA 与锁不一致: "
                f"expected={expected[plugin_id]} actual={actual}"
            )
    return roots


def _install_external_plugins(
    workspace: Path,
    plugin_home: Path,
    roots: dict[str, Path],
) -> dict[str, PluginInstallResult]:
    """通过正式安装入口发布两个外部 artifact，不把 checkout 交给 manager。"""

    installed: dict[str, PluginInstallResult] = {}
    for plugin_id in ("proactive_feedback", "emotion"):
        installed[plugin_id] = install_git_plugin(
            workspace=workspace,
            source=str(roots[plugin_id]),
            marketplace="interop",
            plugins_home=plugin_home,
        )
    return installed


async def _model_command(
    control: RuntimeModelControl, payload: dict[str, object]
) -> dict[str, object]:
    """通过 Models 插件的公开 RPC 配置真实 embedding provider。"""

    result = await control.invoke_rpc("models/command", payload)
    assert isinstance(result, dict)
    assert result.get("status") == 200, result
    body = result.get("body")
    assert isinstance(body, dict)
    return body


async def _eventually(predicate: Any, *, attempts: int = 300) -> None:
    """等待耐久消费者完成一次真实跨插件提交。"""

    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("PF/Emotion 互操作状态未在限定时间内完成")


def _count(path: Path, query: str) -> int:
    if not path.exists():
        return 0
    try:
        with closing(sqlite3.connect(path)) as connection:
            return int(connection.execute(query).fetchone()[0])
    except sqlite3.OperationalError:
        return 0


def _rows(
    path: Path,
    query: str,
    parameters: tuple[object, ...] = (),
) -> list[tuple[object, ...]]:
    """Read committed fixture rows for assertions that need owner identity."""

    with closing(sqlite3.connect(path)) as connection:
        return [tuple(row) for row in connection.execute(query, parameters)]


async def _embedding_server() -> tuple[web.AppRunner, str]:
    """提供真实 openai-compatible embedding HTTP 边界，返回确定向量。"""

    async def models(_request: web.Request) -> web.Response:
        return web.json_response({"data": [{"id": "fixture-embedding"}]})

    async def embeddings(request: web.Request) -> web.Response:
        payload = await request.json()
        texts = payload.get("input")
        if not isinstance(texts, list) or not texts:
            raise web.HTTPBadRequest(text="input must be a non-empty list")
        return web.json_response(
            {
                "data": [
                    {"index": index, "embedding": [1.0, 0.0]}
                    for index, _ in enumerate(texts)
                ],
                "usage": {"prompt_tokens": 1, "total_tokens": len(texts)},
            }
        )

    application = web.Application()
    application.router.add_get("/v1/models", models)
    application.router.add_post("/v1/embeddings", embeddings)
    runner = web.AppRunner(application)
    await runner.setup()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    await web.SockSite(runner, sock).start()
    return runner, f"http://127.0.0.1:{sock.getsockname()[1]}/v1"


async def _append_followup(
    core: Any,
    *,
    explicit_quote: bool,
    suffix: str,
    session_id: str = "wake:interop",
) -> None:
    """经正式 MessageLog 与当前 Content owner 追加完整 Wake follow-up。"""

    async with lease_runtime_snapshot(core.plugin_manager.snapshot_store) as snapshot:
        content = snapshot.composition_root.context.require(CONTENT)
        checks = {"text": content.check_text}
        log = core.message_log
        _ = log.ensure_session(session_id, SessionAttributes())
        proactive = log.writer(
            session_id,
            author="wake",
            source="wake",
            body_types=(Output,),
            content=checks,
        )
        proactive.append(
            f"p{suffix}",
            Output(
                (ContentPart("text", "主动提醒某个很长很长的主题"),),
                "complete",
            ),
        )
        user_text = (
            "被回复消息：主动提醒某个很长很长的主题\n\n"
            "【你当前新消息】我继续这个主题"
            if explicit_quote
            else "我继续这个主题"
        )
        user = log.writer(
            session_id,
            author="user",
            source="conversation",
            body_types=(Input,),
            content=checks,
        )
        user.append(
            f"u{suffix}",
            Input((ContentPart("text", user_text),)),
        )
        assistant = log.writer(
            session_id,
            author="assistant",
            source="conversation",
            body_types=(Output,),
            content=checks,
        )
        assistant.append(
            f"a{suffix}",
            Output((ContentPart("text", "我接着回答这个主题"),), "complete"),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_quote", (False, True))
async def test_installed_manager_message_append_reaches_pf_and_emotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    explicit_quote: bool,
) -> None:
    """正式归档加载后，真实 Message 自然抵达 PF 与 Emotion。"""

    roots = _plugin_roots()
    workspace = tmp_path / "workspace"
    _ = init_workspace(config_path=tmp_path / "config.toml", workspace=workspace)
    plugin_home, _ = install_formal_plugins(
        tmp_path,
        _BUILTIN_PLUGINS,
        marketplace="fixture",
    )
    installed = _install_external_plugins(workspace, plugin_home, roots)
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(plugin_home))

    runner, endpoint = await _embedding_server()
    http: SharedHttpResources | None = SharedHttpResources()
    core: Any | None = build_core_runtime(Config(), workspace, http, plugin_dirs=[])
    try:
        assert core is not None and http is not None
        await core.start()
        host = core.plugin_manager
        expected = {"proactive_feedback@interop", "emotion@interop"}
        assert expected <= set(host.current_snapshot.generations)
        for plugin_id in expected:
            generation = host.generation(plugin_id)
            assert generation is not None
            descriptor = host._archive.read_descriptor(generation.archive_ref)
            archive_root = host._archive.open(descriptor["code"])
            loaded = Path(generation.instance.module.__file__).resolve()
            assert loaded.is_relative_to(archive_root.resolve())
            source = roots[plugin_id.split("@", 1)[0]].resolve()
            assert not loaded.is_relative_to(source)

        await host.start_runtime()
        control = RuntimeModelControl(host.snapshot_store)
        await _model_command(
            control,
            {
                "type": "add_connection",
                "expected_revision": 0,
                "connection_id": "local",
                "name": "Local",
                "driver_id": "openai-compatible",
                "endpoint": endpoint,
                "auth_identity": "fixture",
                "credential": {"api_key": "fixture"},
            },
        )
        await _model_command(
            control,
            {
                "type": "add_model",
                "expected_revision": 1,
                "model_id": "fixture-embedding",
                "connection_id": "local",
                "kind": "embedding",
                "model": "fixture-embedding",
                "capabilities": {
                    "embedding_dimensions": 2,
                    "embedding_normalization": "unit",
                },
                "capability_sources": {},
            },
        )
        await _model_command(
            control,
            {
                "type": "set_default",
                "expected_revision": 2,
                "role": None,
                "model_id": "fixture-embedding",
            },
        )

        await _append_followup(core, explicit_quote=explicit_quote, suffix="1")
        pf_db = installed["proactive_feedback"].data_path / "proactive_feedback.db"
        emotion_db = workspace / "emotion" / "emotion.db"

        await _eventually(
            lambda: _count(
                pf_db,
                "SELECT count(*) FROM proactive_feedback_events",
            )
            == 1
        )
        pf_first = _rows(
            pf_db,
            "SELECT id, user_message_id, assistant_message_id, "
            "proactive_message_id, feedback_type, confidence, pa_score, pua_score "
            "FROM proactive_feedback_events ORDER BY id",
        )
        assert len(pf_first) == 1
        assert pf_first[0][0] == 1
        assert pf_first[0][1:4] == ("u1", "a1", "p1")
        expected_feedback_type = "explicit_quote" if explicit_quote else "topic_follow"
        assert pf_first[0][4] == expected_feedback_type
        assert pf_first[0][5] == ("gold" if explicit_quote else "high")
        if explicit_quote:
            assert pf_first[0][6:8] == (1.0, 1.0)
        else:
            assert pf_first[0][7] == pytest.approx(1.0)
        assert _count(
            pf_db,
            "SELECT count(*) FROM proactive_feedback_input_inbox "
            "WHERE processed_at IS NOT NULL",
        ) == 1

        # Emotion observes the same completed Turn directly from MessageCatalog.
        if explicit_quote:
            await _eventually(
                lambda: _count(
                    emotion_db,
                    "SELECT count(*) FROM emotion_events "
                    "WHERE source_event_id='emotion_explicit_quote:a1' "
                    "AND source_type='explicit_quote'",
                )
                == 1
            )
        else:
            assert _count(
                emotion_db,
                "SELECT count(*) FROM emotion_events "
                "WHERE source_plugin='emotion' AND source_type='explicit_quote'",
            ) == 0

        async def restart_manager() -> None:
            """Restart the formal manager so its real immediate Timer runs."""

            nonlocal core, http
            assert core is not None and http is not None
            await core.bus.aclose()
            await core.stop()
            core = None
            await http.aclose()
            http = SharedHttpResources()
            core = build_core_runtime(Config(), workspace, http, plugin_dirs=[])
            await core.start()
            await core.plugin_manager.start_runtime()

        # The first real restart consumes exactly PF row 1 through the
        # generation-owned immediate Timer; no lifecycle event is emitted here.
        await restart_manager()
        await _eventually(
            lambda: _count(
                emotion_db,
                "SELECT row_id FROM pf_history_cursor "
                "WHERE source='proactive_feedback'",
            )
            == 1,
            attempts=1400,
        )
        first_import = _rows(
            emotion_db,
            "SELECT source_plugin, source_event_id, source_type, "
            "valence_delta, dominance_delta "
            "FROM emotion_events WHERE source_event_id='proactive_feedback:1'",
        )
        assert len(first_import) == 1
        assert first_import[0][0] == "proactive_feedback"
        if explicit_quote:
            assert first_import[0][2:] == (
                "explicit_quote_already_applied",
                0.0,
                0.0,
            )
        else:
            assert first_import[0][2] == "topic_follow"
            assert isinstance(first_import[0][3], (int, float))
            assert isinstance(first_import[0][4], (int, float))
            assert first_import[0][3] > 0.0
            assert first_import[0][4] > 0.0
        assert _count(emotion_db, "SELECT count(*) FROM emotion_feedback_samples") == 1

        # Add another real completed Message after the first pull.  The running
        # history Timer is on its ordinary interval, so row 2 remains pending.
        await _append_followup(core, explicit_quote=explicit_quote, suffix="2")
        await _eventually(
            lambda: _count(
                pf_db,
                "SELECT count(*) FROM proactive_feedback_events",
            )
            == 2
        )
        if explicit_quote:
            await _eventually(
                lambda: _count(
                    emotion_db,
                    "SELECT count(*) FROM emotion_events "
                    "WHERE source_event_id='emotion_explicit_quote:a2' "
                    "AND source_type='explicit_quote'",
                )
                == 1
            )
        assert _count(
            emotion_db,
            "SELECT row_id FROM pf_history_cursor "
            "WHERE source='proactive_feedback'",
        ) == 1
        pf_rows = _rows(
            pf_db,
            "SELECT id, user_message_id, assistant_message_id, "
            "proactive_message_id FROM proactive_feedback_events ORDER BY id",
        )
        assert [row[0] for row in pf_rows] == [1, 2]
        assert [row[1:] for row in pf_rows] == [
            ("u1", "a1", "p1"),
            ("u2", "a2", "p2"),
        ]

        # A second real restart fires another immediate Timer and consumes only
        # the newly pending PF row 2.
        await restart_manager()
        await _eventually(
            lambda: _count(
                emotion_db,
                "SELECT row_id FROM pf_history_cursor "
                "WHERE source='proactive_feedback'",
            )
            == 2,
            attempts=1400,
        )
        imported = _rows(
            emotion_db,
            "SELECT source_plugin, source_event_id, source_type, "
            "valence_delta, dominance_delta "
            "FROM emotion_events "
            "WHERE source_event_id LIKE 'proactive_feedback:%' ORDER BY source_event_id",
        )
        assert [row[1] for row in imported] == [
            "proactive_feedback:1",
            "proactive_feedback:2",
        ]
        assert all(row[0] == "proactive_feedback" for row in imported)
        if explicit_quote:
            assert [row[2:] for row in imported] == [
                ("explicit_quote_already_applied", 0.0, 0.0),
                ("explicit_quote_already_applied", 0.0, 0.0),
            ]
            assert _count(
                emotion_db,
                "SELECT count(*) FROM emotion_events "
                "WHERE source_type='explicit_quote'",
            ) == 2
        else:
            assert [row[2] for row in imported] == ["topic_follow", "topic_follow"]
            for row in imported:
                assert isinstance(row[3], (int, float)) and isinstance(row[4], (int, float))
                assert row[3] > 0.0 and row[4] > 0.0
        state = _rows(
            emotion_db,
            "SELECT valence, dominance FROM emotion_state WHERE id=1",
        )
        assert len(state) == 1
        assert isinstance(state[0][0], (int, float))
        assert isinstance(state[0][1], (int, float))
        assert state[0][0] > 0.0
        assert state[0][1] > 0.0
        assert _count(emotion_db, "SELECT count(*) FROM emotion_feedback_samples") == 2
        sample_ids = _rows(
            emotion_db,
            "SELECT source_event_id FROM emotion_feedback_samples ORDER BY id",
        )
        if explicit_quote:
            assert sample_ids == [("emotion_explicit_quote:a1",), ("emotion_explicit_quote:a2",)]
        else:
            assert sample_ids == [("proactive_feedback:1",), ("proactive_feedback:2",)]

    finally:
        if core is not None:
            await core.bus.aclose()
            await core.stop()
        if http is not None:
            await http.aclose()
        await runner.cleanup()
