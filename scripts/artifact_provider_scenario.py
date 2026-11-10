"""两个独立附件 provider 经正式安装链服务同一个不重启的消费者。"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def repository(source: Path, destination: Path) -> None:
    """只复制真实源码到临时 Git 仓库，正式安装不引用开发目录。"""
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__"))
    subprocess.run(["git", "init", "-q", "--initial-branch=source", str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "add", "."], check=True)
    subprocess.run(["git", "-C", str(destination), "-c", "user.name=Scenario", "-c",
                    "user.email=scenario@example.invalid", "commit", "-qm", "provider"], check=True)


async def run(directory: Path) -> dict[str, object]:
    """实际导入文件、热换 provider、保留引用读取、重开及卸载。"""
    from agent.plugin_composition import ServiceKey
    from agent.plugins.manager import PluginManager
    from agent.plugins.selection import PluginSelection

    workspace, home, sources = directory / "workspace", directory / "home", directory / "sources"
    workspace.mkdir()
    PluginSelection(workspace).initialize()
    os.environ.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(home), AKASHIC_PLUGIN_DISTRIBUTION="")
    ledger, alternate = directory / "ledger", directory / "alternate"
    repository(ROOT / "plugins/ledger", ledger)
    repository(ROOT / "examples/artifact_provider", alternate)
    consumer = sources / "consumer"
    consumer.mkdir(parents=True)
    (consumer / "plugin.py").write_text('''from agent.plugin_composition import ServiceKey
from plugins.ledger.contract import ARTIFACT_IMPORT, ARTIFACT_READ, AttachmentKind
api_version = 3
name = "consumer"
version = "1.0.0"
async def apply(ctx):
    with (ctx.data_root / "applies").open("a") as stream:
        stream.write("apply\\n")
    async def copy(path):
        with ctx.borrow(ARTIFACT_IMPORT) as imports:
            if imports is None:
                raise LookupError("附件导入 provider 缺席")
            return await imports.import_source(path, AttachmentKind.FILE)
    async def read(ref):
        with ctx.borrow(ARTIFACT_READ) as files:
            if files is None:
                raise LookupError("附件读取 provider 缺席")
            lease = await files.acquire(ref)
            try:
                whole = await lease.read_bytes(max_bytes=ref.size_bytes)
                part = await lease.read_chunk(offset=1, max_bytes=3)
                return whole, part
            finally:
                await lease.aclose()
    await ctx.provide(ServiceKey("scenario.import"), ctx.entrypoint(copy))
    await ctx.provide(ServiceKey("scenario.read"), ctx.entrypoint(read))
''')
    original = directory / "message.txt"
    original.write_bytes("a real attachment\n".encode())
    def build():
        return PluginManager([sources], workspace=workspace, installed_cache_root=home / "cache")
    host = build()
    try:
        await host.load_all()
        await host.install(source=str(ledger), marketplace="lab", ref_name="", sparse_paths=[], update_id="ledger")
        await host.wait_idle()
        root = host.live_root
        assert root is not None
        copy, read = root.context.require(ServiceKey("scenario.import")), root.context.require(ServiceKey("scenario.read"))
        ref = await copy(str(original))
        expected = await read(ref)
        observer = host.generation("consumer").fiber
        await host.uninstall("ledger@lab")
        await host.wait_idle()
        await host.install(source=str(alternate), marketplace="lab", ref_name="", sparse_paths=[], update_id="alternate")
        await host.wait_idle()
        assert host.read_update("alternate").state == "active"
        other = await copy(str(original))
        assert await read(other) == expected
        # 第二实现按公共内容身份读回原引用，不解释旧 SQLite 或原 artifact ID。
        assert await read(ref) == expected
        assert host.generation("consumer").fiber is observer
        assert (workspace / "plugin-data/consumer-builtin/applies").read_text() == "apply\n"
        original.unlink()
    finally:
        await host.terminate_all()
    host = build()
    try:
        await host.load_all()
        root = host.live_root
        assert root is not None
        read = root.context.require(ServiceKey("scenario.read"))
        assert await read(ref) == expected
        await host.uninstall("artifact_provider@lab")
        await host.wait_idle()
        try:
            await read(ref)
        except LookupError:
            pass
        else:
            raise AssertionError("卸载后仍在使用旧 provider")
    finally:
        await host.terminate_all()
    return {"ports": ["ledger.artifact_import.v1", "ledger.artifact_read.v1"],
            "independent_blob_provider": True, "consumer_apply_delta": 0,
            "same_bytes_and_chunk": True, "old_reference": True, "restart": True, "unload": True}


if __name__ == "__main__":
    directory = Path(tempfile.mkdtemp(prefix="akashic-artifact-provider-"))
    print(json.dumps({"evidence": str(directory)}), flush=True)
    print(json.dumps(asyncio.run(run(directory))), flush=True)
