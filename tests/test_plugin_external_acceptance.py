from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

import docker.debug.plugin_external_acceptance as external_acceptance
from agent.plugin_composition import ServiceKey
from docker.debug.plugin_external_acceptance import (
    _exercise_core_bootstrap,
    _exercise_business_composition,
    _ensure_empty_directory,
    _invoke_capability,
    _load_distribution,
    _write_bootstrap_config,
)


def _git_commit(source: Path, message: str = "fixture") -> None:
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=external-acceptance",
            "-c",
            "user.email=external-acceptance@example.invalid",
            "-c",
            "commit.gpgSign=false",
            "commit",
            "--quiet",
            "-m",
            message,
        ],
        check=True,
    )


def _write_plugin_source(
    root: Path, *, name: str, version: str, module: str
) -> Path:
    root.mkdir(parents=True)
    (root / "plugin.py").write_text(module, encoding="utf-8")
    (root / "akashic.plugin.toml").write_text(
        "\n".join(
            (
                "schema_version = 1",
                f'name = "{name}"',
                f'version = "{version}"',
                "api_version = 3",
                'entrypoint = "plugin.py"',
                "",
            )
        ),
        encoding="utf-8",
    )
    _git_commit(root, f"{name}-{version}")
    return root


def _copy_content_source(root: Path) -> Path:
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "plugins" / "content",
        root,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    _git_commit(root, "content-business-subset")
    return root


_MESSAGE_ROUNDTRIP_PLUGIN = '''
from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.messages import (
    MESSAGE_CATALOG,
    MESSAGE_WRITERS,
)
from agent.plugin_contracts import ContentPart, Input

CONTENT = ServiceKey("content.v2")
ROUNDTRIP = ServiceKey("acceptance.messages.roundtrip.v1")
api_version = 3
name = "message_roundtrip"
version = "1.0.0"
inject = (MESSAGE_CATALOG, MESSAGE_WRITERS, CONTENT)


async def apply(ctx: Context, config: object) -> None:
    _ = config
    catalog = ctx.require(MESSAGE_CATALOG)
    writers = ctx.require(MESSAGE_WRITERS)
    content = ctx.require(CONTENT)

    def roundtrip(request):
        if set(request) != {"session_id", "message_id", "text"}:
            raise ValueError("roundtrip request fields 无效")
        session_id = request["session_id"]
        message_id = request["message_id"]
        text = request["text"]
        if not all(isinstance(value, str) and value for value in (session_id, message_id, text)):
            raise TypeError("roundtrip request 必须是非空字符串")
        open_writer = writers.bind(
            ctx,
            author="external-acceptance",
            source="acceptance",
            body_types=(Input,),
            content={"text": content.check_text},
        )
        written = open_writer(session_id).append(
            message_id,
            Input((ContentPart("text", text),)),
        )
        read = catalog.reader(session_id).get(written.message_id)
        if read is None or len(read.body.parts) != 1 or read.body.parts[0].kind != "text":
            raise RuntimeError("roundtrip Message reader 未读回唯一 text")
        return {
            "session_id": session_id,
            "message_id": written.message_id,
            "written_text": text,
            "read_text": read.body.parts[0].value,
        }

    await ctx.provide(ROUNDTRIP, roundtrip)
'''


def _provider_plugin(value: str, version: str) -> str:
    return f'''
from agent.plugin_composition import Context, ServiceKey

PROVIDER = ServiceKey("acceptance.provider.v1")
api_version = 3
name = "provider"
version = "{version}"
inject = ()


async def apply(ctx: Context, config: object) -> None:
    _ = config

    def provide(request):
        if not isinstance(request, dict) or request.get("kind") != "replacement":
            raise ValueError("provider request 无效")
        return {{"value": "{value}"}}

    await ctx.provide(PROVIDER, provide)
'''


_CONSUMER_PLUGIN = '''
from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.messages import (
    MESSAGE_CATALOG,
    MESSAGE_WRITERS,
)
from agent.plugin_contracts import ContentPart, Input

CONTENT = ServiceKey("content.v2")
PROVIDER = ServiceKey("acceptance.provider.v1")
CONSUMER = ServiceKey("acceptance.consumer.roundtrip.v1")
api_version = 3
name = "consumer"
version = "1.0.0"
inject = (PROVIDER, MESSAGE_CATALOG, MESSAGE_WRITERS, CONTENT)


async def apply(ctx: Context, config: object) -> None:
    _ = config
    provider = ctx.require(PROVIDER)
    catalog = ctx.require(MESSAGE_CATALOG)
    writers = ctx.require(MESSAGE_WRITERS)
    content = ctx.require(CONTENT)

    def consume(request):
        if set(request) != {"session_id", "message_id", "text", "kind"}:
            raise ValueError("consumer request fields 无效")
        session_id = request["session_id"]
        message_id = request["message_id"]
        text = request["text"]
        provider_value = provider({"kind": request["kind"]})["value"]
        rendered = f"{provider_value}:{text}"
        open_writer = writers.bind(
            ctx,
            author="external-acceptance-consumer",
            source="acceptance",
            body_types=(Input,),
            content={"text": content.check_text},
        )
        written = open_writer(session_id).append(
            message_id,
            Input((ContentPart("text", rendered),)),
        )
        read = catalog.reader(session_id).get(written.message_id)
        if read is None or len(read.body.parts) != 1 or read.body.parts[0].kind != "text":
            raise RuntimeError("consumer Message reader 未读回唯一 text")
        return {
            "session_id": session_id,
            "message_id": written.message_id,
            "provider": provider_value,
            "written_text": rendered,
            "read_text": read.body.parts[0].value,
        }

    await ctx.provide(CONSUMER, consume)
'''


def _roundtrip_spec(
    *, message_id: str, text: str, provider: str | None = None,
    service: str = "acceptance.messages.roundtrip.v1",
) -> dict:
    value = {
        "session_id": "acceptance:composition",
        "message_id": message_id,
        "provider": provider,
        "written_text": text if provider is None else f"{provider}:{text}",
        "read_text": text if provider is None else f"{provider}:{text}",
    }
    if provider is None:
        value.pop("provider")
    return {
        "service": service,
        "entrypoint": "external_acceptance.message_roundtrip",
        "input": {
            "session_id": "acceptance:composition",
            "message_id": message_id,
            "text": text,
            **({"kind": "replacement"} if provider is not None else {}),
        },
        "expect": {"type": "dict", "value": value},
    }


def test_external_acceptance_rejects_nonempty_runtime_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "old-state").write_text("must not be reused", encoding="utf-8")

    with pytest.raises(ValueError, match="必须为空"):
        _ensure_empty_directory(workspace, "workspace")


def test_bootstrap_workspace_seeds_legal_context_material_grants(tmp_path: Path) -> None:
    _write_bootstrap_config(tmp_path, marketplace="acceptance")

    config = (tmp_path / "plugin-data/context-acceptance/config.local.toml").read_text(
        encoding="utf-8"
    )
    assert 'default_prompt = "prompt@acceptance"' in config
    assert 'markdown_memory = "markdown_memory@acceptance"' in config
    assert 'skills = "standard_tools@acceptance"' in config
    assert 'summary_source = ["compaction", "compaction@acceptance"]' in config


def test_capability_enumeration_does_not_count_as_call() -> None:
    key = ServiceKey("message.display:model.facts")

    class Root:
        def provided_services(self, *, plugin_ids):
            _ = plugin_ids
            return {key: object()}

    spec = {
        "service": key.name,
        "entrypoint": "plugins.models.projection.display_facts",
        "input": {
            "kind": "model.facts",
            "value": {
                "call_record_id": "acceptance-call",
                "tool_ids": {},
                "thinking": None,
                "continuation": None,
            },
        },
        "expect": {"type": "dict", "value": {}},
    }

    with pytest.raises(TypeError, match="不可调用"):
        asyncio.run(_invoke_capability(root=Root(), plugin_id="models@test", spec=spec))


def test_capability_oracle_calls_declared_input_and_checks_output() -> None:
    key = ServiceKey("message.display:model.facts")
    seen = []

    def display(part):
        seen.append(part)
        return {"call_record_id": part.value["call_record_id"], "thinking": part.value["thinking"]}

    class Root:
        def provided_services(self, *, plugin_ids):
            _ = plugin_ids
            return {key: display}

    spec = {
        "service": key.name,
        "entrypoint": "plugins.models.projection.display_facts",
        "input": {
            "kind": "model.facts",
            "value": {
                "call_record_id": "acceptance-call",
                "tool_ids": {},
                "thinking": None,
                "continuation": None,
            },
        },
        "expect": {
            "type": "dict",
            "value": {"call_record_id": "acceptance-call", "thinking": None},
        },
    }

    evidence = asyncio.run(
        _invoke_capability(root=Root(), plugin_id="models@test", spec=spec)
    )
    assert evidence["call_executed"] is True
    assert evidence["status"] == "passed"
    assert len(seen) == 1
    assert seen[0].kind == "model.facts"


def test_distribution_report_requires_external_bundle_files(tmp_path: Path) -> None:
    path = tmp_path / "distribution.json"
    path.write_text(
        json.dumps(
            {
                "source_commit": "0" * 40,
                "core": {"file": "core.tar"},
                "plugins": [{"name": "example", "file": "example.bundle"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="bundle 缺失"):
        _load_distribution(path)


def test_core_probe_records_real_start_and_stop_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class Manager:
        current_snapshot = object()

    class Core:
        plugin_manager = Manager()

    class Runtime:
        _started = True
        _shutdown = False
        core = Core()
        channel_host = object()
        app_server = object()
        dashboard_task = None
        chat_task = None
        mobile_gateway_task = None
        plugin_watcher_task = None

        async def shutdown(self) -> None:
            calls.append("stop")
            self._shutdown = True
            self.core.plugin_manager.current_snapshot = None

    async def start(**kwargs):
        calls.append("start")
        return (
            Runtime(),
            {
                "checks": {
                    "bootstrap_start_returned": True,
                    "runtime_started": True,
                    "core_runtime_created": True,
                    "stable_snapshot_published": True,
                    "channel_host_started": True,
                    "app_server_started": True,
                    "checkout_invisible": True,
                    "core_modules_from_artifact": True,
                },
                "status": "passed",
            },
            {"AKASHIC_PLUGIN_HOME": None, "AKASHIC_WORKSPACE": None},
        )

    monkeypatch.setattr(external_acceptance, "_validate_core_root", lambda root, repo: root)
    monkeypatch.setattr(external_acceptance, "_prepare_runtime", lambda **kwargs: {})
    monkeypatch.setattr(external_acceptance, "_start_app_runtime", start)

    result = asyncio.run(
        _exercise_core_bootstrap(
            repo_root=tmp_path / "repo",
            core_root=tmp_path / "core",
            workspace=tmp_path / "workspace",
            plugins_home=tmp_path / "plugins-home",
        )
    )

    assert result["status"] == "passed"
    assert calls == ["start", "stop"]
    assert result["bootstrap"]["checks"]["stable_snapshot_drained"] is True


@pytest.mark.asyncio
async def test_business_composition_writes_reads_and_replaces_provider_from_new_generation(
    tmp_path: Path,
) -> None:
    """Use real installed artifacts and MessageLog for two legal external subsets."""

    sources = tmp_path / "sources"
    sources.mkdir()
    content = _copy_content_source(sources / "content")
    message_roundtrip = _write_plugin_source(
        sources / "message-roundtrip",
        name="message_roundtrip",
        version="1.0.0",
        module=_MESSAGE_ROUNDTRIP_PLUGIN,
    )
    provider_old = _write_plugin_source(
        sources / "provider-old",
        name="provider",
        version="1.0.0",
        module=_provider_plugin("original", "1.0.0"),
    )
    provider_new = _write_plugin_source(
        sources / "provider-new",
        name="provider",
        version="2.0.0",
        module=_provider_plugin("replacement", "2.0.0"),
    )
    consumer = _write_plugin_source(
        sources / "consumer",
        name="consumer",
        version="1.0.0",
        module=_CONSUMER_PLUGIN,
    )

    result = await _exercise_business_composition(
        jobs=[
            {"label": "content", "source": str(content)},
            {
                "label": "message_roundtrip",
                "source": str(message_roundtrip),
                "capability_spec": _roundtrip_spec(
                    message_id="roundtrip-1",
                    text="content-and-message",
                ),
            },
            {"label": "provider", "source": str(provider_old)},
            {
                "label": "consumer",
                "source": str(consumer),
                "capability_spec": _roundtrip_spec(
                    message_id="replacement-before",
                    text="consumer-before",
                    provider="original",
                    service="acceptance.consumer.roundtrip.v1",
                ),
            },
        ],
        repo_root=Path(__file__).resolve().parents[1],
        marketplace="acceptance",
        workspace=tmp_path / "workspace",
        plugins_home=tmp_path / "plugins-home",
        replacement={
            "plugin": "provider",
            "consumer": "consumer",
            "source": str(provider_new),
            "before": _roundtrip_spec(
                message_id="replacement-old-lease",
                text="old-lease",
                provider="original",
                service="acceptance.consumer.roundtrip.v1",
            ),
            "after": _roundtrip_spec(
                message_id="replacement-new-lease",
                text="new-lease",
                provider="replacement",
                service="acceptance.consumer.roundtrip.v1",
            ),
            "remove_original_source": True,
        },
    )

    assert result["status"] == "passed", result
    assert result["checks"] == {
        "composition_loaded": True,
        "all_reports_passed": True,
        "business_calls_executed": True,
        "durable_message_readback": True,
        "replacement_verified": True,
    }
    assert all(row["status"] == "passed" for row in result["reports"]), [
        (row.get("plugin"), row.get("checks"), row.get("checkout_modules_visible"))
        for row in result["reports"]
    ]
    replacement = result["replacement"]
    assert replacement is not None
    assert replacement["status"] == "passed"
    assert replacement["old_consumer_call"]["message_readback"]["text_parts"] == (
        "original:old-lease",
    )
    assert replacement["new_consumer_call"]["message_readback"]["text_parts"] == (
        "replacement:new-lease",
    )
    assert replacement["old_generation_id"] != replacement["new_generation_id"]
    assert replacement["checks"]["consumer_read_under_new_snapshot"] is True
    assert replacement["checks"]["replacement_module_from_new_artifact"] is True
    assert replacement["checks"]["replacement_module_not_old_artifact"] is True
    assert replacement["checks"]["original_source_not_required"] is True
    assert provider_old.exists()
    assert not list(provider_old.parent.glob(provider_old.name + ".before-acceptance-*"))


@pytest.mark.asyncio
async def test_legal_subsets_run_separately_and_accept_differently_named_provider(tmp_path: Path):
    """两个独立安装根分别运行，另一家 provider 不需要原包名字或源码。"""
    sources = tmp_path / "sources"
    sources.mkdir()
    content = _copy_content_source(sources / "content")
    roundtrip = _write_plugin_source(
        sources / "message-roundtrip", name="message_roundtrip", version="1.0.0",
        module=_MESSAGE_ROUNDTRIP_PLUGIN,
    )
    first = await _exercise_business_composition(
        jobs=[
            {"label": "content", "source": str(content)},
            {"label": "message_roundtrip", "source": str(roundtrip),
             "capability_spec": _roundtrip_spec(message_id="subset-one", text="standalone")},
        ],
        repo_root=Path(__file__).resolve().parents[1], marketplace="acceptance",
        workspace=tmp_path / "first-workspace", plugins_home=tmp_path / "first-home",
    )
    assert first["status"] == "passed", first
    assert {row["plugin_id"] for row in first["reports"]} == {
        "content@acceptance", "message_roundtrip@acceptance",
    }

    alternate = _write_plugin_source(
        sources / "alternate", name="alternate", version="1.0.0",
        module=_provider_plugin("independent", "1.0.0").replace('name = "provider"', 'name = "alternate"'),
    )
    consumer = _write_plugin_source(
        sources / "consumer", name="consumer", version="1.0.0", module=_CONSUMER_PLUGIN,
    )
    second = await _exercise_business_composition(
        jobs=[
            {"label": "content", "source": str(content)},
            {"label": "alternate", "source": str(alternate)},
            {"label": "consumer", "source": str(consumer), "capability_spec": _roundtrip_spec(
                message_id="subset-two", text="same-consumer", provider="independent",
                service="acceptance.consumer.roundtrip.v1",
            )},
        ],
        repo_root=Path(__file__).resolve().parents[1], marketplace="acceptance",
        workspace=tmp_path / "second-workspace", plugins_home=tmp_path / "second-home",
    )
    assert second["status"] == "passed", second
    assert {row["plugin_id"] for row in second["reports"]} == {
        "content@acceptance", "alternate@acceptance", "consumer@acceptance",
    }
    call = next(row["capability_call"] for row in second["reports"] if row["plugin_id"] == "consumer@acceptance")
    assert call["message_readback"]["text_parts"] == ("independent:same-consumer",)
