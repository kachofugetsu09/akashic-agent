from pathlib import Path
import shutil

def sources(path):
    shutil.copytree(Path(__file__).resolve().parents[2] / "plugins/delivery", path / "delivery",
                    ignore=shutil.ignore_patterns("__pycache__"))
    target = path / "test_sender"
    target.mkdir()
    (target / "plugin.py").write_text('''
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
from typing import Literal
from agent.plugin_composition import RUNTIME_STARTED, ServiceKey
api_version = 3
name = "test_sender"
version = "1.0.0"
inject = (ServiceKey("delivery.senders.v1"), ServiceKey("delivery.v1"))

@dataclass(frozen=True)
class SendResult:
    status: Literal["delivered", "rejected", "failed"]
    provider_ids: tuple[str, ...] = ()
    error: str | None = None

async def apply(ctx):
    async def start(_event):
        path = ctx.data_root / "receiver-starts"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as file:
            file.write("started\\n")
    await ctx.on(RUNTIME_STARTED, start)
    class Sender:
        idempotent = True
        async def send(self, key, address, message):
            path = ctx.data_root / "sent.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as file:
                file.write(json.dumps([key, address, message.message_id, "original-A"]) + "\\n")
            return SendResult(status="delivered", provider_ids=("original-A",))
        async def query(self, key, address):
            path = ctx.data_root / "sent.jsonl"
            if not path.exists():
                return None
            for line in path.read_text().splitlines():
                entry = json.loads(line)
                if entry[0] == key and entry[1] == address:
                    return SendResult(status="delivered", provider_ids=("original-A",))
            return None
    @asynccontextmanager
    async def open():
        if (ctx.data_root / "credential-revoked").exists():
            raise PermissionError("original credential revoked")
        yield Sender()
    await ctx.require(inject[0]).register(ctx, name="test", idempotent=True, open=open)
    await ctx.provide(ServiceKey("fixture.delivery"), lambda: ctx.require(ServiceKey("delivery.v1")).open(ctx))
''')
