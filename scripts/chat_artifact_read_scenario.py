"""通过真实附件存储和 HTTP 路由验证 Unicode 文件名可读。"""
from __future__ import annotations

import asyncio
from email.message import Message
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugins.akashic_clients.chat_api import create_chat_app
from plugins.akashic_clients.web_chat import WebChatChannel
from plugins.ledger.artifact_store import ArtifactStore
from plugins.ledger.attachments import ChannelAttachmentArtifactStore
from plugins.ledger.contract import AttachmentKind


async def run(workspace: Path) -> None:
    """发布隔离附件，经 HTTP 读回原字节并核对浏览器下载文件名。"""
    # 1. 使用真实附件 owner，读取路由不获得写入或改名权限。
    metadata = ArtifactStore(workspace / "artifacts.db")
    files = ChannelAttachmentArtifactStore(workspace=workspace, metadata_store=metadata)
    app = create_chat_app(workspace=workspace, channel=WebChatChannel(), artifact_store=files)
    content = json.dumps({"schema": "akashic.visual.v1", "title": "斜率与直线",
                          "summary": "拖动参数", "html": "<h2>中文交互图</h2>"}, ensure_ascii=False).encode()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://scenario") as client:
            # 2. 中文、emoji 和响应头特殊字符均应保留可读的文件名。
            for filename in ("斜率与直线.akashic-visual.json", "图表 📈.json", "chart.json", "chart 100%;.json"):
                ref = await files.import_bytes(content, kind=AttachmentKind.FILE,
                                               filename=filename, media_type="application/json")
                response = await client.get(f"/api/chat/artifacts/{ref.artifact_id}")
                assert response.status_code == 200, response.text
                assert response.content == content
                header = Message()
                header["Content-Disposition"] = response.headers["content-disposition"]
                assert header.get_content_disposition() == "inline"
                assert header.get_filename() == filename
                assert files.resolve_refs((ref.artifact_id,)) == (ref,)
            assert (await client.get("/api/chat/artifacts/missing")).status_code == 404
    finally:
        metadata.close()
    print(json.dumps({"http_read": "passed", "unicode_filename": "preserved", "artifact_bytes": "unchanged"}))


if __name__ == "__main__":
    with TemporaryDirectory(prefix="chat-artifact-read-") as directory:
        asyncio.run(run(Path(directory)))
