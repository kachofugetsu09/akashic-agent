"""真实附件与回复组合验收；使用临时 workspace，不调用外部模型。"""

import asyncio
from datetime import UTC, datetime
import hashlib
import io
from pathlib import Path
import random
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from agent.plugin_composition import ServiceKey
from agent.plugin_composition.artifacts import ArtifactRead
from agent.plugin_composition.channels import CHANNEL_INPUT_V2 as CHANNEL_INPUT, ChannelInboundMessage
from agent.plugin_contracts import ContentPart, Control
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from plugins.models.content import ContentOwner, load_artifacts
from session.artifact_store import ArtifactStore
from session.artifacts import AttachmentKind
from session.message import Output
from plugins.content.plugin import check_artifact, check_text
from tests.test_default_reply import application, live_root


def image_bytes(*, padded=False, noise=False):
    image = (Image.frombytes("RGB", (1024, 1024), random.Random(3).randbytes(1024 * 1024 * 3))
             if noise else Image.new("RGB", (2, 2), "red"))
    data = io.BytesIO()
    image.save(data, format="PNG")
    raw = data.getvalue()
    return raw + bytes(15 * 1024 * 1024 - len(raw)) if padded else raw


async def resource_scenario(path):
    """实际 ArtifactRead 核验字节、重复引用计费、未读取旧图与明确失败。"""
    (path / "workspace").mkdir(exist_ok=True)
    metadata = ArtifactStore(path / "sessions.db")
    store = ChannelAttachmentArtifactStore(workspace=path / "workspace", metadata_store=metadata)
    acquired = []

    async def acquire(ref):
        acquired.append(ref.artifact_id)
        return await store.acquire(ref)

    async def load(refs, current):
        return await load_artifacts(ArtifactRead(acquire), refs, accepts_images=True,
                                    current_artifact_ids=frozenset(current))

    try:
        refs = [await store.import_bytes(image_bytes(padded=True), kind=AttachmentKind.IMAGE,
                                        filename=f"{i}.png", media_type="image/png") for i in range(3)]
        view = await load(refs, (refs[-1].artifact_id,))
        assert acquired == [refs[-1].artifact_id, refs[-2].artifact_id]
        assert view[refs[-1].artifact_id][-1]["type"] == "image_url"
        assert "本次未提供图片内容" in str(view[refs[0].artifact_id])
        acquired.clear()
        try:
            await load(refs, (ref.artifact_id for ref in refs))
        except ValueError as error:
            assert "当前输入图片" in str(error)
        else:
            raise AssertionError("Current images exceeding the resource budget must fail")
        noisy = await store.import_bytes(image_bytes(noise=True), kind=AttachmentKind.IMAGE,
                                         filename="noise.png", media_type="image/png")
        acquired.clear()
        view = await load((noisy,) * 5, ())
        assert acquired == [noisy.artifact_id]
        assert len(view[noisy.artifact_id]) == 1
        assert "资源预算已用完" in str(view[noisy.artifact_id])
        try:
            await load((noisy,) * 5, (noisy.artifact_id,))
        except ValueError:
            pass
        else:
            raise AssertionError("Repeated current image blocks must be charged")
        for ref in (*refs, noisy):
            lease = await store.acquire(ref)
            try:
                assert hashlib.sha256(await lease.read_bytes(max_bytes=20 * 1024 * 1024)).hexdigest() == ref.sha256
            finally:
                await lease.aclose()
    finally:
        metadata.close()


async def reply_scenario(path):
    """真实旧 model.facts 后续仍重投影历史图，工具原图沿同一组合进入请求。"""
    def provider(sources):
        module = sources / "test_provider/plugin.py"
        module.write_text(module.read_text().replace(
            "ModelCapabilities(context_window=10000)",
            'ModelCapabilities(context_window=10000, input_modalities=("text", "image"))',
        ).replace('return Result("success", (ContentPart("text", "written"),))',
                  f'return Result("success", (ContentPart("artifact_ref", Path({str(path / "tool-ref")!r}).read_text()),))'))

    async with application(path, replying=True, extra_sources=provider) as (log, host):
        metadata = ArtifactStore(path / "sessions.db")
        store = ChannelAttachmentArtifactStore(workspace=path / "workspace", metadata_store=metadata)
        try:
            ref = await store.import_bytes(image_bytes(), kind=AttachmentKind.IMAGE,
                                           filename="photo.png", media_type="image/png")
        finally:
            metadata.close()
        (path / "tool-ref").write_text(ref.artifact_id)
        requests = None

        async def send(identity, attachments=()):
            nonlocal requests
            async with live_root(host) as root:
                accepted = await root.context.require(CHANNEL_INPUT)(
                    "test:room", identity, ChannelInboundMessage(
                        "test", "user", "room", "Describe the image.", datetime.now(UTC), {},
                        attachments=attachments,
                    ),
                )
                requests = root.context.require(ServiceKey("fixture.calls"))
            async def terminal():
                async for _ in log.catalog().follow():
                    for row in log.reader("test:room").snapshot():
                        if row.seq > accepted.seq and (isinstance(row.body, Control) or
                                isinstance(row.body, Output) and row.body.finish == "complete"):
                            assert isinstance(row.body, Output), row.body
                            return row
            return await asyncio.wait_for(terminal(), 5)

        # 1. 模拟旧策略产生一次真实持久 model.facts，而不是伪造 replay。
        original = ContentOwner.load_artifacts
        async def old_policy(reader, refs, **kwargs):
            return await original(reader, refs, accepts_images=False,
                                  current_artifact_ids=kwargs["current_artifact_ids"])
        ContentOwner.load_artifacts = staticmethod(old_policy)
        try:
            await send("before", (ref,))
        finally:
            ContentOwner.load_artifacts = staticmethod(original)
        before = log.reader("test:room").snapshot()
        assert any(part.kind == "model.facts" for row in before if isinstance(row.body, Output)
                   for part in row.body.parts if isinstance(part, ContentPart))
        # 2. 不再上传的文字后续也能看到保留的原 Input 图和 ToolResult 图。
        await send("after")
        assert requests is not None and len(requests) == 3
        images = [part for row in requests[-1].messages if isinstance(row["content"], (list, tuple))
                  for part in row["content"] if part["type"] == "image_url"]
        assert len(images) == 2
        assert log.reader("test:room").snapshot()[:len(before)] == before


async def summary_scenario(path):
    """资源标记不拦截 Context 的真实摘要与后续业务请求。"""
    def provider(sources):
        module = sources / "test_provider/plugin.py"
        module.write_text(module.read_text().replace(
            "ModelCapabilities(context_window=10000)",
            'ModelCapabilities(context_window=10000, input_modalities=("text", "image"))',
        ))
    async with application(path, replying=True, compaction=True, output_tokens=128,
                           extra_sources=provider) as (log, host):
        metadata = ArtifactStore(path / "sessions.db")
        store = ChannelAttachmentArtifactStore(workspace=path / "workspace", metadata_store=metadata)
        writer = log.writer("test:room", author="scheduler", source="scheduler", body_types=(Output,),
                            content={"text": check_text, "artifact_ref": check_artifact})
        try:
            for index in range(5):
                ref = await store.import_bytes(image_bytes(padded=True), kind=AttachmentKind.IMAGE,
                                               filename=f"old-{index}.png", media_type="image/png")
                writer.append(f"old-{index}", Output((ContentPart("text", "past " * 3000),
                              ContentPart("artifact_ref", ref.artifact_id)), "complete"))
            current = await store.import_bytes(image_bytes(), kind=AttachmentKind.IMAGE,
                                               filename="current.png", media_type="image/png")
        finally:
            writer.expire()
            metadata.close()
        before = log.reader("test:room").snapshot()
        async with live_root(host) as root:
            accepted = await root.context.require(CHANNEL_INPUT)("test:room", "now",
                ChannelInboundMessage("test", "user", "room", "Describe this new photo.",
                                      datetime.now(UTC), {}, attachments=(current,)))
            requests = root.context.require(ServiceKey("fixture.calls"))
        async def terminal():
            async for _ in log.catalog().follow():
                for row in log.reader("test:room").snapshot():
                    if row.seq > accepted.seq and (isinstance(row.body, Control) or
                            isinstance(row.body, Output) and row.body.finish == "complete"):
                        assert isinstance(row.body, Output), row.body
                        return
        await asyncio.wait_for(terminal(), 10)
        rows = log.reader("test:room").snapshot()
        assert rows[:len(before)] == before
        assert any(part.kind == "context.summary" for row in rows if isinstance(row.body, Output)
                   for part in row.body.parts if isinstance(part, ContentPart))
        assert any("[Source messages]" in str(request.messages) for request in requests)
        assert current.artifact_id in str(requests[-1].messages)
        assert "image_url" in str(requests[-1].messages)


async def main():
    with tempfile.TemporaryDirectory(prefix="akashic-image-view-") as directory:
        path = Path(directory)
        (path / "resources").mkdir()
        (path / "reply").mkdir()
        await resource_scenario(path / "resources")
        await reply_scenario(path / "reply")
        (path / "summary").mkdir()
        await summary_scenario(path / "summary")
    print("PASS: bounded current/history image view, repeated references, real facts continuation, tool image, immutable originals")


if __name__ == "__main__":
    asyncio.run(main())
