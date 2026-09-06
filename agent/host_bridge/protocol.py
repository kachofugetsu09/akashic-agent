from __future__ import annotations

import base64
import binascii
from collections.abc import Iterable

from google.protobuf.message import Message

from agent.host_bridge import host_bridge_pb2 as pb
from agent.tools.base import ToolResult
from agent.tools.unified_exec import (
    ExecutionCleanupFailure,
    ExecutionCleanupReport,
    ExecutionResult,
)

# UDS 已有访问控制；关闭重试，避免重放可能已执行的命令或输入。
CHANNEL_OPTIONS = (
    ("grpc.max_receive_message_length", 16 * 1024 * 1024),
    ("grpc.max_send_message_length", 16 * 1024 * 1024),
    ("grpc.enable_retries", 0),
)


def require_fields(message: Message, *names: str) -> None:
    for name in names:
        if not message.HasField(name):
            raise ValueError(f"Host Bridge {message.DESCRIPTOR.name}.{name} 缺失")


def require_text(value: str, name: str) -> None:
    if not value:
        raise ValueError(f"Host Bridge {name} 必须非空")


def require_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"Host Bridge {name} 必须大于零")


def require_nonnegative(value: int, name: str) -> None:
    if value < 0:
        raise ValueError(f"Host Bridge {name} 不能为负数")


def require_names(values: Iterable[str], name: str) -> None:
    for value in values:
        require_text(value, name)


def encode_execution(result: ExecutionResult) -> pb.ExecutionReply:
    """把已有执行结果直接编码为字节和互斥的运行/退出字段。"""
    reply = pb.ExecutionReply(
        output=result.output,
        wall_time_ms=result.wall_time_ms,
        original_token_count=result.original_token_count,
        output_omitted_bytes=result.output_omitted_bytes,
        output_path=result.output_path,
        finish_reason=result.finish_reason,
    )
    if result.execution_id is not None:
        if result.exit_code is not None:
            raise RuntimeError("execution 同时包含句柄和退出码")
        reply.execution_id = result.execution_id
    elif result.exit_code is not None:
        reply.exit_code = result.exit_code
    else:
        raise RuntimeError("execution 缺少句柄和退出码")
    return reply


def decode_execution(reply: pb.ExecutionReply) -> ExecutionResult:
    """在远端响应边界拒绝缺失结果，保留空输出和退出码零。"""
    # 1. 校验响应的存在性及值域，不用 protobuf 默认值补齐坏响应。
    require_fields(
        reply, "output", "wall_time_ms", "original_token_count", "output_omitted_bytes"
    )
    require_nonnegative(reply.wall_time_ms, "wall_time_ms")
    require_nonnegative(reply.original_token_count, "original_token_count")
    require_nonnegative(reply.output_omitted_bytes, "output_omitted_bytes")
    require_text(reply.finish_reason, "finish_reason")
    state = reply.WhichOneof("result")
    if state is None:
        raise ValueError("Host Bridge execution 缺少运行或退出结果")
    if state == "execution_id":
        require_positive(reply.execution_id, "execution_id")
    if reply.HasField("output_path"):
        require_text(reply.output_path, "output_path")
    # 2. 只转回已有领域结果，协议不持有执行状态。
    return ExecutionResult(
        output=reply.output,
        wall_time_ms=reply.wall_time_ms,
        original_token_count=reply.original_token_count,
        output_omitted_bytes=reply.output_omitted_bytes,
        execution_id=reply.execution_id if state == "execution_id" else None,
        exit_code=reply.exit_code if state == "exit_code" else None,
        output_path=reply.output_path if reply.HasField("output_path") else None,
        finish_reason=reply.finish_reason,
    )


def encode_cleanup(report: ExecutionCleanupReport) -> pb.CleanupReply:
    return pb.CleanupReply(
        attempted=report.attempted_execution_ids,
        cleaned=report.cleaned_execution_ids,
        failures=[
            pb.CleanupFailure(
                execution_id=f.execution_id, error_type=f.error_type, message=f.message
            )
            for f in report.failures
        ],
    )


def decode_cleanup(reply: pb.CleanupReply) -> ExecutionCleanupReport:
    """校验清理报告，保留所有尚未确认回收的 execution。"""
    for execution_id in (*reply.attempted, *reply.cleaned):
        require_positive(execution_id, "cleanup execution_id")
    failures = []
    for failure in reply.failures:
        require_fields(failure, "execution_id")
        require_positive(failure.execution_id, "execution_id")
        require_text(failure.error_type, "error_type")
        require_text(failure.message, "message")
        failures.append(
            ExecutionCleanupFailure(
                failure.execution_id, failure.error_type, failure.message
            )
        )
    return ExecutionCleanupReport(
        tuple(reply.attempted), tuple(reply.cleaned), tuple(failures)
    )


def encode_file_result(result: str | ToolResult) -> pb.FileReply:
    """只转换文件工具已拥有的文本或单张图片，不接受动态扩展载荷。"""
    # 1. 文本包括工具业务错误，保持其原有返回语义。
    if isinstance(result, str):
        return pb.FileReply(text=result)
    if (
        result.mobile_attention is not None
        or result.runtime_provenance
        or len(result.content_blocks) != 1
    ):
        raise RuntimeError("Host Bridge 文件结果不是单张图片")
    block = result.content_blocks[0]
    if set(block) != {"type", "image_url"} or block["type"] != "image_url":
        raise RuntimeError("Host Bridge 文件结果不是 image_url")
    image = block["image_url"]
    if (
        not isinstance(image, dict)
        or set(image) != {"url", "detail"}
        or image["detail"] != "high"
    ):
        raise RuntimeError("Host Bridge 文件图片字段不符合合同")
    url = image["url"]
    if (
        not isinstance(url, str)
        or not url.startswith("data:image/")
        or ";base64," not in url
    ):
        raise RuntimeError("Host Bridge 文件图片必须是 data URI")
    # 2. 去掉展示层的 Base64；协议只携带原字节和 MIME。
    header, encoded = url.split(";base64,", 1)
    mime = header[5:]
    if mime not in {"image/png", "image/jpeg", "image/gif", "image/webp"}:
        raise RuntimeError("Host Bridge 文件图片 MIME 不受支持")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError("Host Bridge 文件图片 Base64 损坏") from exc
    if not data:
        raise RuntimeError("Host Bridge 文件图片为空")
    return pb.FileReply(
        image=pb.FileImage(text=result.text, mime_type=mime, data=data, detail="high")
    )


def decode_file_result(reply: pb.FileReply) -> str | ToolResult:
    """在 Core 重建文件工具结果，模型图片能力仍由原工具判断。"""
    kind = reply.WhichOneof("result")
    if kind == "text":
        return reply.text
    if kind != "image":
        raise ValueError("Host Bridge 文件响应缺少结果")
    image = reply.image
    require_fields(image, "text", "data")
    if (
        image.mime_type not in {"image/png", "image/jpeg", "image/gif", "image/webp"}
        or image.detail != "high"
        or not image.data
    ):
        raise ValueError("Host Bridge 文件图片响应不符合合同")
    uri = (
        f"data:{image.mime_type};base64,{base64.b64encode(image.data).decode('ascii')}"
    )
    return ToolResult(
        text=image.text,
        content_blocks=[
            {"type": "image_url", "image_url": {"url": uri, "detail": image.detail}}
        ],
    )
