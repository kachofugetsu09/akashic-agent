"""plugin_update 只消费公开安装结果，并保留历史通知的只读身份。"""
from __future__ import annotations

import asyncio
import json

import pytest

from agent.plugin_composition import ServiceKey
from agent.plugin_composition.messages import MESSAGE_WRITERS, OWNER_STATE
from agent.plugin_composition.plugin_updates import UpdateStatus
from plugins.delivery.records import DeliveryRecords, delivery_key
from plugins.plugin_update.inputs import CONTENT, DELIVERY
from plugins.plugin_update.plugin import result_message_id
from plugins.plugin_update.tool import InstallInput, decode_request, receipt, update_id
from session.message import ContentPart, Output


REPORT_HEALTH_CONTROL = ServiceKey("test.report-health-control")
DELIVERY_CONTROL = ServiceKey("test.delivery-control")


def test_historical_owner_request_drops_only_retired_validation_fields():
    """历史 OWNER_STATE 不再触发验证，未知字段仍交给严格 schema 拒绝。"""
    request = decode_request({
        "install": {
            "source": "https://example.invalid/plugin.git",
            "marketplace": "lab",
            "ref": "main",
            "sparse": ["plugin.py"],
            "validation_prompt": "retired",
            "validation_tools": ["retired_tool"],
            "excluded_materials": ["retired-material"],
        },
        "session_id": "akashic:room",
        "sink": None,
    })
    assert request.install.source.endswith("plugin.git")
    assert request.install.sparse == ["plugin.py"]

    with pytest.raises(ValueError):
        InstallInput.model_validate({
            "source": "source",
            "marketplace": "lab",
            "validation_prompt": "new request must reject this",
        })

    with pytest.raises(ValueError):
        decode_request({
            "install": {
                "source": "source",
                "marketplace": "lab",
                "future_field": True,
            },
            "session_id": "akashic:room",
            "sink": None,
        })


def test_terminal_result_ids_are_distinct_and_do_not_reuse_history():
    """失败后重试 active 使用新结果 ID，旧 complete/problem 消息保持只读。"""
    active = UpdateStatus(
        "u-active", "probe@lab", "input-active", "selected", "g-active",
        "input-active", "ACTIVE", "active", "",
    )
    failed = UpdateStatus(
        "u-failed", "probe@lab", "input-failed", "selected", None,
        None, None, "failed", "load failed",
    )
    assert result_message_id("plugin-update:key", active) == "plugin-update:key:result-active"
    assert result_message_id("plugin-update:key", failed) == "plugin-update:key:result-failed"
    assert result_message_id("plugin-update:key", active) not in {
        "plugin-update:key:complete", "plugin-update:key:problem",
    }
    assert receipt(active).outcome == "success"
    assert receipt(failed).outcome == "error"


@pytest.mark.asyncio
async def test_real_plugin_update_watcher_reports_through_delivery(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
):
    """Drive failed -> retry active reporting with durable send receipts."""
    from tests.test_default_reply import application
    from tests.test_plugin_install import _commit, _write_v3_plugin
    from agent.plugins.install import install_git_plugin
    from agent.plugin_composition.bindings import BINDINGS
    from plugins.plugin_update.inputs import DELIVERY_SENDERS
    from plugins.plugin_update.tool import InstallPlugin
    from session.log import SessionAttributes

    def extra_sources(sources):
        _write_v3_plugin(
            sources / "report_health_control",
            name="report_health_control",
            module_source=(
                'import asyncio\n'
                'from agent.plugin_composition import ServiceKey\n'
                'CONTROL = ServiceKey("test.report-health-control")\n'
                'api_version = 3\nname = "report_health_control"\nversion = "1.0.0"\n'
                'async def apply(ctx):\n'
                '    await ctx.provide(CONTROL, {"fail": False})\n'
            ),
        )
        initial_target = tmp_path / "initial-target"
        _write_v3_plugin(
            initial_target,
            name="report_target",
            module_source=(
                'from agent.plugin_composition import ServiceKey\n'
                'TARGET = ServiceKey("test.report-target")\n'
                'api_version = 3\nname = "report_target"\nversion = "1.0.0"\n'
                'async def apply(ctx):\n'
                '    await ctx.provide(TARGET, "old")\n'
            ),
        )
        _commit(initial_target)
        install_git_plugin(
            workspace=tmp_path / "workspace", source=str(initial_target),
            marketplace="builtin", plugins_home=tmp_path / "home",
        )
        _write_v3_plugin(
            sources / "report_consumer",
            name="report_consumer",
            module_source=(
                'from agent.plugin_composition import ServiceKey\n'
                'TARGET = ServiceKey("test.report-target")\n'
                'CONTROL = ServiceKey("test.report-health-control")\n'
                'api_version = 3\nname = "report_consumer"\nversion = "1.0.0"\n'
                'inject = (TARGET, CONTROL)\n'
                'async def apply(ctx):\n'
                '    health = await ctx.health("report-required")\n'
                '    if ctx.require(CONTROL)["fail"]:\n'
                '        health.degrade("controlled report health failure")\n'
                '    _ = ctx.require(TARGET)\n'
            ),
        )
        _write_v3_plugin(
            sources / "delivery_control",
            name="delivery_control",
            module_source=(
                'import asyncio\n'
                'from agent.plugin_composition import ServiceKey\n'
                'CONTROL = ServiceKey("test.delivery-control")\n'
                'api_version = 3\nname = "delivery_control"\nversion = "1.0.0"\n'
                'async def apply(ctx):\n'
                '    await ctx.provide(CONTROL, {\n'
                '        "started": asyncio.Event(), "release": asyncio.Event(),\n'
                '        "finished": asyncio.Event(),\n'
                '    })\n'
            ),
        )
        (sources / "test_sender" / "plugin.py").write_text(
            '''
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
from typing import Literal
from agent.plugin_composition import ServiceKey
SENDERS = ServiceKey("delivery.senders.v1")
CONTROL = ServiceKey("test.delivery-control")
api_version = 3
name = "test_sender"
version = "1.0.0"
inject = (SENDERS, CONTROL)

@dataclass(frozen=True)
class SendResult:
    status: Literal["delivered", "rejected", "failed"]
    provider_ids: tuple[str, ...] = ()
    error: str | None = None

async def apply(ctx):
    control = ctx.require(CONTROL)
    class Sender:
        idempotent = True
        async def send(self, key, address, message):
            control["started"].set()
            await control["release"].wait()
            path = ctx.data_root / "sent.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as file:
                file.write(json.dumps([key, address, message.message_id, "report-sender"]) + "\\n")
            control["finished"].set()
            return SendResult(status="delivered", provider_ids=("report-sender",))
        async def query(self, key, address):
            return None
    @asynccontextmanager
    async def open():
        yield Sender()
    await ctx.require(SENDERS).register(ctx, name="test", idempotent=True, open=open)
'''
        )

    candidate = tmp_path / "candidate"
    _write_v3_plugin(
        candidate,
        name="report_target",
        module_source=(
            'from agent.plugin_composition import ServiceKey\n'
            'TARGET = ServiceKey("test.report-target")\n'
            'api_version = 3\nname = "report_target"\nversion = "2.0.0"\n'
            'async def apply(ctx):\n    await ctx.provide(TARGET, "new")\n'
        ),
    )
    _commit(candidate)

    send_calls: list[tuple[str, str]] = []
    replay_processed = asyncio.Event()
    initial_send_count = 0

    async with application(
        tmp_path, replying=False, updates=True, extra_sources=extra_sources,
    ) as (log, host):
        log.ensure_session("updates", SessionAttributes())
        generation = next(
            item for item in host._active_generations.values()
            if item.plugin_id == "plugin_update"
        )
        plugin_context = generation.fiber.context
        identity = update_id("real-report")
        health_control = host.live_root.context.require(REPORT_HEALTH_CONTROL)
        delivery_control = host.live_root.context.require(DELIVERY_CONTROL)
        historical_complete = identity + ":complete"
        historical_problem = identity + ":problem"
        async with plugin_context.runtime_scope():
            actual_delivery = plugin_context.require(DELIVERY).open(plugin_context)
            delivery_type = type(actual_delivery)
            original_send = delivery_type.send

            async def observe_send(delivery, message_id: str, sink: str):
                result = await original_send(delivery, message_id, sink)
                send_calls.append((message_id, sink))
                if len(send_calls) > initial_send_count and message_id.endswith(
                    (":result-failed", ":result-active")
                ):
                    replay_processed.set()
                return result

            # Patch only the archived runtime class returned by this real scope.
            monkeypatch.setattr(delivery_type, "send", observe_send)
            writers = plugin_context.require(MESSAGE_WRITERS)
            content = plugin_context.require(CONTENT)
            history_writer = writers.bind(
                plugin_context, author="history", source="history",
                body_types=(Output,), content={"text": content.check_text},
            )("updates")
            try:
                history_writer.append(
                    historical_complete,
                    Output((ContentPart("text", "old complete"),), "complete"),
                )
                history_writer.append(
                    historical_problem,
                    Output((ContentPart("text", "old problem"),), "complete"),
                )
            finally:
                history_writer.expire()
            health_control["fail"] = True
            bindings = plugin_context.require(BINDINGS)
            senders = plugin_context.require(DELIVERY_SENDERS).bind_all(bindings)
            tool = InstallPlugin(plugin_context, senders)
            result = await tool.invoke(
                "real-report",
                {
                    "install": {
                        "source": str(candidate),
                        "marketplace": "builtin",
                        "ref": "",
                        "sparse": [],
                    },
                    "session_id": "updates",
                    "sink": {"name": "test", "binding_id": senders["test"], "address": "room"},
                },
            )
            request_facts = plugin_context.require(OWNER_STATE).open(plugin_context).read(identity)
            delivery_owner = plugin_context.require_runtime_owner(
                DELIVERY, plugin_context.require(DELIVERY),
            )
        assert result.outcome == "success"
        assert request_facts is not None
        operation = host._operation
        assert operation is not None
        await asyncio.wait_for(delivery_control["started"].wait(), 5)
        delivery_control["release"].set()
        await asyncio.wait_for(asyncio.gather(operation.task, return_exceptions=True), 5)
        failed = host.read_update(identity)
        assert failed.state == "failed"
        failed_message = log.reader("updates").get(identity + ":result-failed")
        assert failed_message is not None and isinstance(failed_message.body, Output)
        assert failed_message.body.parts[0].value.startswith("插件 report_target@builtin 更新失败：")
        await asyncio.wait_for(delivery_control["finished"].wait(), 5)
        initial_send_count = len(send_calls)

        health_control["fail"] = False
        delivery_control["started"].clear()
        delivery_control["finished"].clear()
        delivery_control["release"] = asyncio.Event()
        retry = asyncio.create_task(host.retry_runtime_recovery("report_target@builtin"))
        await asyncio.wait_for(delivery_control["started"].wait(), 5)
        delivery_control["release"].set()
        await asyncio.wait_for(retry, 5)
        active = host.read_update(identity)
        assert active.state == "active"
        active_message = log.reader("updates").get(identity + ":result-active")
        assert active_message is not None and isinstance(active_message.body, Output)
        assert active_message.body.parts[0].value == "插件 report_target@builtin 已激活。"
        assert failed_message.body != active_message.body
        await asyncio.wait_for(delivery_control["finished"].wait(), 5)
        initial_send_count = len(send_calls)
        replay_processed.clear()

        assert log.reader("updates").get(historical_complete).body.parts[0].value == "old complete"
        assert log.reader("updates").get(historical_problem).body.parts[0].value == "old problem"
        async with plugin_context.runtime_scope():
            assert plugin_context.require(OWNER_STATE).open(plugin_context).read(identity) == request_facts
        delivery_generation = host.generation("delivery")
        assert delivery_generation is not None and delivery_generation.fiber is not None
        delivery_context = delivery_generation.fiber.context
        async with delivery_context.runtime_scope():
            records = DeliveryRecords(
                delivery_context.require(OWNER_STATE).open(delivery_context),
                delivery_owner,
            )
            failed_delivery = records.read(identity + ":result-failed", "test")[1]
            active_delivery = records.read(identity + ":result-active", "test")[1]
        assert failed_delivery.phase == "delivered"
        assert failed_delivery.receipt is not None
        assert failed_delivery.receipt.provider_ids == ("report-sender",)
        assert active_delivery.phase == "delivered"
        assert active_delivery.receipt is not None
        assert active_delivery.receipt.provider_ids == ("report-sender",)

        sent = next(tmp_path.rglob("sent.jsonl"))
        sent_records = [json.loads(line) for line in sent.read_text().splitlines()]
        assert len(sent_records) == 2
        by_message_id = {record[2]: record for record in sent_records}
        for message_id, delivery_record in (
            (failed_message.message_id, failed_delivery),
            (active_message.message_id, active_delivery),
        ):
            sent_record = by_message_id[message_id]
            assert sent_record[0] == delivery_key(message_id, "test")
            assert sent_record[1] == "room"
            assert sent_record[2] == message_id
            assert delivery_record.receipt is not None
            assert sent_record[3] in delivery_record.receipt.provider_ids

        # A duplicate wake and a watcher restart reuse the same delivered receipts.
        host._notify_updates()
        await host.retry_runtime_recovery("plugin_update")
        await asyncio.wait_for(replay_processed.wait(), 5)
        replayed_ids = {
            message_id for message_id, _sink in send_calls[initial_send_count:]
        }
        assert replayed_ids == {active_message.message_id}
        assert len(sent.read_text().splitlines()) == 2
