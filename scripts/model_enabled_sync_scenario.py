"""在一次性 SQLite 和真实 Models owner 上验证模型选择、删除与回退。

运行：python -m scripts.model_enabled_sync_scenario；受控 driver 不联网。
"""
from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agent.plugin_composition import (
    CapabilitySources, ChatModelSelection, DiscoveredModel, ModelCapabilities,
    ModelKind, ModelRequest,
)
from plugins.models.settings import AddModel, RemoveModel, SetDefaultModel, SyncModels
from plugins.models.state import ModelUnavailableError
from plugins.models.store import RevisionConflictError
from tests.support.material_models import MaterialModelDriver, material_models

CONNECTION = "material-fixture-connection"
CHAT = "material-fixture-chat"
EMBEDDING = "material-fixture-embedding"


async def scenario(workspace: Path) -> None:
    """沿设置、执行和账本边界核对选择生命周期。"""
    candidate = DiscoveredModel(
        kind=ModelKind.CHAT, model="extra-chat",
        capabilities=ModelCapabilities(context_window=8192),
        capability_sources=CapabilitySources(context_window="fixture"),
    )
    returned = [replace(candidate, kind=None)]
    original = MaterialModelDriver.definition

    async def discover(*_args):
        return tuple(returned)

    def definition(driver):
        base = original(driver)
        async def probe_embedding(descriptor, credential, model):
            result = await driver.probe_embedding(descriptor, credential,
                "fixture-embedding" if model == "extra-embedding" else model)
            return replace(result, model=model)
        return replace(base, discover=discover, probe_embedding=probe_embedding)

    with patch.object(MaterialModelDriver, "definition", definition):
        async with material_models(workspace) as models:
            def snapshot():
                return models.store.read_snapshot()

            async def sync():
                return await models.settings.apply(SyncModels(snapshot().revision, CONNECTION))

            async def add(model_id, *, discovery_owned=True):
                return await models.settings.apply(AddModel(
                    expected_revision=snapshot().revision, model_id=model_id,
                    connection_id=CONNECTION, kind=candidate.kind, model=candidate.model,
                    capabilities=candidate.capabilities, capability_sources=candidate.capability_sources,
                    discovery_owned=discovery_owned,
                ))

            async def remove(model_id):
                return await models.settings.apply(RemoveModel(snapshot().revision, model_id))

            # 1. 目录不是用户选择；只有显式采纳能增加模型配置。
            before = snapshot()
            await sync()
            assert snapshot() == before
            await add("extra")
            returned[0] = replace(candidate, kind=None, capabilities=ModelCapabilities(context_window=16384))
            await sync()
            assert snapshot().models["extra"].capabilities.context_window == 16384
            returned[:] = [replace(candidate, kind=None, model="unselected-chat")]
            await sync()
            assert "extra" in snapshot().models and not snapshot().models["extra"].enabled
            returned[:] = [replace(candidate, kind=None)]
            await sync()
            assert snapshot().models["extra"].enabled
            # 未声明用途的目录只刷新已验证的选择，不重建向量空间。
            embedding = snapshot().models[EMBEDDING]
            await models.settings.apply(AddModel(
                expected_revision=snapshot().revision, model_id="extra-space",
                connection_id=CONNECTION, kind=ModelKind.EMBEDDING, model="extra-embedding",
                capabilities=embedding.capabilities, capability_sources=embedding.capability_sources,
                discovery_owned=True,
            ))
            space = models.embeddings.describe(model_id="extra-space")
            returned.append(DiscoveredModel(kind=None, model="extra-embedding",
                capabilities=ModelCapabilities(), capability_sources=CapabilitySources()))
            await sync()
            refreshed = models.embeddings.describe(model_id="extra-space")
            assert (refreshed.identity, refreshed.dimensions) == (space.identity, space.dimensions)
            assert snapshot().models[EMBEDDING] == embedding, "manual definitions stay unchanged"
            await remove("extra-space")
            returned.pop()
            print("PASS untyped catalog refreshes selected capabilities and availability, preserves verified spaces/manual definitions, and never adopts new models")

            # 2. 删除有 CAS、真实备份；在途绑定及调用账不依赖被删配置行。
            async with models.chat_models.execution(model_id="extra") as execution:
                old = execution.chat("agent")
                old_revision = snapshot().revision
                await remove("extra")
                try:
                    await models.settings.apply(RemoveModel(old_revision, CHAT))
                except RevisionConflictError:
                    pass
                else:
                    raise AssertionError("stale delete was accepted")
                response = await old.complete(ModelRequest(messages=({"role": "user", "content": "hello"},)))
                assert response.content == "fixture:extra"
                assert response.call_record_id is not None
                record = models.store.read_call(response.call_record_id)
                assert record["state"] == "success"
                assert models.store.read_call_stats(response.call_record_id).model == candidate.model
            calls = models.store.read_calls("", 100)
            await sync()
            assert "extra" not in snapshot().models
            assert models.store.read_calls("", 100) == calls
            assert CONNECTION in snapshot().connections
            restored = False
            for backup in (workspace / "model-backups").rglob("*.sqlite3"):
                with closing(sqlite3.connect(backup)) as connection:
                    assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
                    assert not connection.execute("PRAGMA foreign_key_check").fetchall()
                    restored |= connection.execute("SELECT 1 FROM model_definitions WHERE id='extra'").fetchone() is not None
            assert restored, "a pre-delete recovery copy must contain the removed model"
            print("PASS delete CAS/backup; frozen execution and durable call receipt survive; sync cannot resurrect")

            # 3. 旧会话特选回到 default，而不是 agent 特选；删除 default 简单择一。
            await add("replacement")
            await models.settings.apply(SetDefaultModel(snapshot().revision, "agent", "replacement"))
            selected = models.state.validate_chat_selection(ChatModelSelection("extra", "high"))
            assert selected == ChatModelSelection(CHAT, None)
            async with models.chat_models.execution(model_id="extra", reasoning_effort="high") as execution:
                assert execution.chat("agent").descriptor.model_id == CHAT
            await remove(CHAT)
            assert snapshot().role_bindings["default"] == "replacement"
            await remove("replacement")
            assert not snapshot().role_bindings
            try:
                async with models.chat_models.execution(model_id="extra"):
                    raise AssertionError("execution without any selected chat model was accepted")
            except ModelUnavailableError:
                pass
            print("PASS removed session preference uses default; deleted default selects a remaining model; empty selection fails clearly")

            # 4. 向量空间没有聊天式回退，删除配置不重写已有向量或调用记录。
            descriptor = models.embeddings.describe(model_id=EMBEDDING)
            await remove(EMBEDDING)
            assert snapshot().default_embedding_model_id is None
            try:
                models.embeddings.describe(model_id=descriptor.model_id)
            except ModelUnavailableError:
                pass
            else:
                raise AssertionError("removed embedding identity silently rebound")
            assert models.store.read_calls("", 100) == calls
            with closing(sqlite3.connect(models.store.path)) as connection:
                assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
                assert not connection.execute("PRAGMA foreign_key_check").fetchall()
            print("PASS removed embedding binding cannot switch spaces; registry integrity and ledger remain intact")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="model-selection-") as directory:
        asyncio.run(scenario(Path(directory)))
    print("PASS temporary workspace cleaned")
