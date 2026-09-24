from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
import pytest
from docker.debug.plugin_external_acceptance import _exercise_business_composition

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
    compile(module, str(root / "plugin.py"), "exec")
    root.mkdir(parents=True)
    (root / "plugin.py").write_text(module, encoding="utf-8")
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


async def apply(ctx: Context) -> None:
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


async def apply(ctx: Context) -> None:

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


async def apply(ctx: Context) -> None:
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

    assert result["status"] == "passed", {
        "failed_checks": [key for key, ok in result["checks"].items() if not ok],
        "replacement_failed": (
            None if result["replacement"] is None else [
                key for key, ok in result["replacement"]["checks"].items() if not ok
            ]
        ),
        "errors": [row.get("error") for row in result["reports"] if row["status"] != "passed"],
    }
    assert result["checks"] == {
        "composition_loaded": True,
        "all_reports_passed": True,
        "business_calls_executed": True,
        "durable_message_readback": True,
        "live_root_closed": True,
        "message_log_closed": True,
        "event_bus_closed": True,
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
    assert replacement["checks"]["install_waited_for_old_consumer"] is True
    assert replacement["checks"]["consumer_reactivated_after_provider_change"] is True
    assert replacement["old_consumer_call"]["live_root_id"] == (
        replacement["new_consumer_call"]["live_root_id"]
    )
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
