"""在一次性目录中核对目录工具的输出、翻页、内存和真实 UDS 行为。"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import tracemalloc
from unittest.mock import patch


def page(text: str) -> tuple[list[str], str | None]:
    """读取公开文本页面及提示里的 JSON 游标。"""
    body, marker, hint = text.partition("\n\n[还有条目")
    cursor = None
    if marker:
        cursor, _ = json.JSONDecoder().raw_decode(hint.split("after=", 1)[1])
    return body.splitlines(), cursor


async def run(args: argparse.Namespace) -> dict:
    """调用真实文件工具及 Bridge，只写本次创建的临时目录。"""
    sys.path.insert(0, str(args.source))
    from plugins.host_execution.bridge import filesystem
    from plugins.host_execution.contract import FileError, FileImage

    report = {"source": str(args.source), "checks": []}
    with tempfile.TemporaryDirectory(prefix="list-dir-check-") as temporary:
        root = Path(temporary)
        large = root / "large"
        large.mkdir()
        # 1. 同一大目录用来复现旧版无界输出和核对新版实际工作集。
        for index in range(args.entries):
            (large / (f"{index:06d}-" + "x" * 150)).touch()
        operation = filesystem.ListDirOperation(enable_bridge=False)
        tracemalloc.start()
        first = await operation.execute(str(large))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert isinstance(first, str), first
        report.update(entries=args.entries, output_bytes=len(first.encode()), peak_bytes=peak)
        if args.baseline:
            assert len(first.encode()) > 10_000, "基线没有复现无界目录输出"
            report["checks"].append("baseline_unbounded_output")
            return report
        assert len(first.encode()) <= 10_000
        assert peak < 2 * 1024 * 1024, f"目录工作集过大：{peak}"
        rows, cursor = page(first)
        assert 0 < len(rows) <= 500 and cursor is not None
        report["checks"].append("large_directory_bytes_and_memory")

        # 2. 在静止目录逐页读取，证明没有重复或遗漏，也不靠扩大 limit 续读。
        stable = root / "stable"
        stable.mkdir()
        expected = [f"item-{index:04d}" for index in range(1203)]
        for name in expected:
            (stable / name).touch()
        seen = []
        after = None
        while True:
            result = await operation.execute(str(stable), limit=71, after=after)
            assert isinstance(result, str), result
            entries, next_after = page(result)
            assert len(entries) <= 71 and len(result.encode()) <= 10_000
            seen.extend(line.removeprefix("📄 ") for line in entries)
            if next_after is None:
                break
            assert next_after != after
            after = next_after
        assert seen == expected
        report["checks"].append("complete_stable_directory_paging")

        small = root / "small"
        small.mkdir()
        (small / "b.txt").touch()
        (small / "a-dir").mkdir()
        assert await operation.execute(str(small)) == "📁 a-dir\n📄 b.txt"
        links = root / "links"
        links.mkdir()
        (links / "directory-link").symlink_to(small, target_is_directory=True)
        (links / "missing-link").symlink_to(root / "absent")
        assert await operation.execute(str(links)) == "📁 directory-link\n📄 missing-link"
        quoted = 'c-"\\-中文'
        (small / quoted).touch()
        output = await operation.execute(str(small), limit=2)
        assert isinstance(output, str)
        assert page(output)[1] == "b.txt"
        last = await operation.execute(str(small), after="b.txt")
        assert last == f"📄 {quoted}"
        # 删除旧游标并插入两侧名字：游标是名称边界，不是必须存在的文件句柄。
        (small / "b.txt").unlink()
        (small / "a-new").touch()
        (small / "d-new").touch()
        changed = await operation.execute(str(small), after="b.txt")
        assert changed == f"📄 {quoted}\n📄 d-new"
        quoted_page = await operation.execute(str(small), limit=1, after="b.txt")
        assert isinstance(quoted_page, str) and page(quoted_page)[1] == quoted
        assert await operation.execute(str(small), after=page(quoted_page)[1]) == "📄 d-new"
        utf8 = root / "utf8"
        utf8.mkdir()
        for index in range(200):
            (utf8 / (f"{index:03d}-" + "中文" * 40)).touch()
        utf8_page = await operation.execute(str(utf8))
        assert isinstance(utf8_page, str) and len(utf8_page.encode()) <= 10_000
        assert page(utf8_page)[1] is not None
        for limit in (0, -1, 501, True, 1.5):
            # 故意越过类型提示，核对真实输入边界拒绝浮点数和布尔值。
            result = await operation.execute(str(small), limit=limit)  # pyright: ignore[reportArgumentType]
            assert isinstance(result, FileError)
        missing = await operation.execute(str(root / "absent"))
        assert isinstance(missing, FileError)
        denied = await filesystem.ListDirOperation(
            allowed_dir=small, enable_bridge=False
        ).execute(str(large))
        assert isinstance(denied, FileError)
        report["checks"].append("small_results_errors_and_directory_changes")

        # 3. 在真实线程入口设置屏障；取消必须等物理枚举排空。
        entered = threading.Event()
        release = threading.Event()
        original = filesystem.os.scandir

        def blocked(path):
            entered.set()
            assert release.wait(5), "枚举屏障未释放"
            return original(path)

        with patch.object(filesystem.os, "scandir", blocked):
            task = asyncio.create_task(operation.execute(str(small)))
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                task.cancel()
                checkpoint = asyncio.get_running_loop().create_future()
                asyncio.get_running_loop().call_soon(checkpoint.set_result, None)
                await checkpoint
                assert not task.done(), "取消提前释放了仍在运行的枚举"
            finally:
                release.set()
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("取消未传播")
        report["checks"].append("cancel_drains_physical_enumeration")

        # 4. 真实 Protobuf UDS、认证与 manager admission，不替换 RPC 或业务 handler。
        import grpc
        from plugins.host_execution.bridge import transport
        from plugins.host_execution.bridge.factory import HostBridgeRpcError
        from plugins.host_execution.bridge.client import HostBridgeShellProcessManager
        from plugins.host_execution.bridge.server import HostBridgeService

        class ReplyLossService(HostBridgeService):
            """完成真实目录读取后只丢一次 RPC 响应，不替换目录业务。"""
            list_calls = 0
            drop_next_list = False

            async def FileTool(self, request, context):
                if request.WhichOneof('operation') == 'list':
                    self.list_calls += 1
                reply = await super().FileTool(request, context)
                if request.WhichOneof('operation') == 'list' and self.drop_next_list:
                    self.drop_next_list = False
                    await context.abort(grpc.StatusCode.UNAVAILABLE, '场景：目录读取后响应丢失')
                return reply

        commit = subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
        ).strip()
        digest = "b" * 64
        socket = root / "bridge.sock"
        service = ReplyLossService(
            "scenario-token", 60, root / "artifacts", release_commit=commit,
            toolchain_digest=digest, runtime_checkout=args.source,
            bridge_python=Path(sys.executable),
        )
        server = transport.Server(service)
        await server.start(socket)
        client = HostBridgeShellProcessManager(
            socket, "scenario-boot", "scenario-token", commit, digest
        )
        try:
            await client.claim_boot()
            # 图片由真实本地读取和 UDS 返回相同字节；空缺路径仍是明确错误。
            from PIL import Image
            from io import BytesIO
            image_path = root / "image.png"
            Image.new("RGB", (8, 8), "red").save(image_path)
            reader = filesystem.ReadFileOperation(enable_bridge=False)
            try:
                local_image = await reader.execute(str(image_path))
                remote_image = await client.execute_file_tool(
                    "read_file", allowed_dir=root, arguments={"path": str(image_path)},
                )
                assert isinstance(remote_image, FileImage) and remote_image == local_image
                assert remote_image.mime_type == "image/png"
                with Image.open(BytesIO(remote_image.data)) as decoded, Image.open(image_path) as original:
                    assert decoded.size == original.size and decoded.tobytes() == original.tobytes()
            finally:
                await reader.aclose()
            report["checks"].append("image_bytes_match_local_and_bridge")
            for parameters in ({}, {"limit": 71, "after": "000010-" + "x" * 150}):
                local = await operation.execute(str(large), **parameters)
                remote = await client.execute_file_tool(
                    "list_dir", allowed_dir=large, arguments={"path": str(large), **parameters}
                )
                assert remote == local
            for limit in (0, 501):
                error = await client.execute_file_tool(
                    "list_dir", allowed_dir=large,
                    arguments={"path": str(large), "limit": limit},
                )
                assert isinstance(error, FileError)
            for limit in (True, False):
                local = await operation.execute(str(large), limit=limit)
                assert isinstance(local, FileError)
                try:
                    await client.execute_file_tool(
                        "list_dir", allowed_dir=large,
                        arguments={"path": str(large), "limit": limit},
                    )
                except ValueError as error:
                    assert "limit" in str(error)
                else:
                    raise AssertionError("Python bool 被 protobuf 静默转换成了整数")
            report["checks"].append("real_uds_pages_and_invalid_limits")
            # 5. 服务端读完但响应丢失；客户端不自动重发，重接后显式读取同一页。
            before = service.list_calls
            service.drop_next_list = True
            try:
                await client.execute_file_tool('list_dir', allowed_dir=large, arguments={'path': str(large)})
            except HostBridgeRpcError as error:
                assert error.code is grpc.StatusCode.UNAVAILABLE
            else:
                raise AssertionError('目录响应丢失被伪装成成功')
            assert service.list_calls == before + 1, '响应丢失触发了隐含重发'
            await client.close_transport()
            client = HostBridgeShellProcessManager(socket, 'scenario-boot', 'scenario-token', commit, digest)
            await client.claim_boot()
            repeated = await client.execute_file_tool(
                'list_dir', allowed_dir=large, arguments={'path': str(large)})
            assert repeated == first and service.list_calls == before + 2
            report['checks'].append('lost_directory_reply_no_automatic_replay_and_explicit_reconnect')
        finally:
            try:
                cleanup = await client.shutdown()
                assert not cleanup.failures, cleanup
            finally:
                try:
                    await service.shutdown()
                    assert not service._managers
                finally:
                    await server.stop()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--entries", type=int, default=50_000)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    args.source = args.source.resolve()
    if args.entries < 1000:
        parser.error("--entries 必须至少为 1000")
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
