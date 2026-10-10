from __future__ import annotations

from collections.abc import Iterable

from google.protobuf.message import Message

from . import host_bridge_pb2 as pb
from plugins.host_execution.contract import FileError, FileImage, FileResult
from plugins.host_execution.contract import (
    ExecutionCleanupFailure, ExecutionCleanupReport, ExecutionResult
)

# 宿主只接纳执行级展示与诊断字段；Core 的其余环境不会改变宿主身份。
EXECUTION_ENV_NAMES = (
    "AKASHIC_PLUGIN_ROLLOUT_OWNER_TURN", "AKASHIC_CALL_CONTEXT", "NO_COLOR",
    "TERM", "COLORTERM", "PAGER", "GIT_PAGER", "GH_PAGER",
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


def encode_file_result(result: FileResult) -> pb.FileReply:
    """协议直接携带物理结果；模型内容块由工具消费方构造。"""
    if isinstance(result, str):
        return pb.FileReply(text=result)
    if isinstance(result, FileError):
        return pb.FileReply(error=pb.FileError(text=result.text, is_error=True))
    return pb.FileReply(image=pb.FileImage(
        text=result.text, mime_type=result.mime_type, data=result.data, detail="high",
    ))


def decode_file_result(reply: pb.FileReply) -> FileResult:
    """在 RPC 边界校验文件结果，不构造模型内容块。"""
    kind = reply.WhichOneof("result")
    if kind == "text":
        return reply.text
    if kind == "error":
        error = reply.error
        require_fields(error, "text", "is_error")
        if not error.is_error:
            raise ValueError("Host Bridge 文件错误结果必须标记 is_error")
        return FileError(error.text)
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
    return FileImage(image.text, image.mime_type, image.data)
