from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RequestContext(_message.Message):
    __slots__ = ("boot_id", "manager_id", "request_id", "expected_release_commit", "expected_toolchain_digest", "session_ref", "turn_id")
    BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    MANAGER_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_RELEASE_COMMIT_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_TOOLCHAIN_DIGEST_FIELD_NUMBER: _ClassVar[int]
    SESSION_REF_FIELD_NUMBER: _ClassVar[int]
    TURN_ID_FIELD_NUMBER: _ClassVar[int]
    boot_id: str
    manager_id: str
    request_id: str
    expected_release_commit: str
    expected_toolchain_digest: str
    session_ref: str
    turn_id: str
    def __init__(self, boot_id: _Optional[str] = ..., manager_id: _Optional[str] = ..., request_id: _Optional[str] = ..., expected_release_commit: _Optional[str] = ..., expected_toolchain_digest: _Optional[str] = ..., session_ref: _Optional[str] = ..., turn_id: _Optional[str] = ...) -> None: ...

class ContextRequest(_message.Message):
    __slots__ = ("context",)
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    context: RequestContext
    def __init__(self, context: _Optional[_Union[RequestContext, _Mapping]] = ...) -> None: ...

class IdentityReply(_message.Message):
    __slots__ = ("release_commit", "toolchain_digest", "capabilities")
    RELEASE_COMMIT_FIELD_NUMBER: _ClassVar[int]
    TOOLCHAIN_DIGEST_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    release_commit: str
    toolchain_digest: str
    capabilities: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, release_commit: _Optional[str] = ..., toolchain_digest: _Optional[str] = ..., capabilities: _Optional[_Iterable[str]] = ...) -> None: ...

class ClaimBootReply(_message.Message):
    __slots__ = ("owner_boot_id", "previous_boot_id", "cleaned_manager_count", "cleaned_execution_count")
    OWNER_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    PREVIOUS_BOOT_ID_FIELD_NUMBER: _ClassVar[int]
    CLEANED_MANAGER_COUNT_FIELD_NUMBER: _ClassVar[int]
    CLEANED_EXECUTION_COUNT_FIELD_NUMBER: _ClassVar[int]
    owner_boot_id: str
    previous_boot_id: str
    cleaned_manager_count: int
    cleaned_execution_count: int
    def __init__(self, owner_boot_id: _Optional[str] = ..., previous_boot_id: _Optional[str] = ..., cleaned_manager_count: _Optional[int] = ..., cleaned_execution_count: _Optional[int] = ...) -> None: ...

class HeartbeatReply(_message.Message):
    __slots__ = ("alive",)
    ALIVE_FIELD_NUMBER: _ClassVar[int]
    alive: bool
    def __init__(self, alive: bool = ...) -> None: ...

class ExecRequest(_message.Message):
    __slots__ = ("context", "command", "argv", "cwd", "env", "tty", "yield_time_ms", "max_output_tokens", "hard_timeout_s", "owner_session_key", "shell_snapshot")
    class EnvEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    ARGV_FIELD_NUMBER: _ClassVar[int]
    CWD_FIELD_NUMBER: _ClassVar[int]
    ENV_FIELD_NUMBER: _ClassVar[int]
    TTY_FIELD_NUMBER: _ClassVar[int]
    YIELD_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    MAX_OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    HARD_TIMEOUT_S_FIELD_NUMBER: _ClassVar[int]
    OWNER_SESSION_KEY_FIELD_NUMBER: _ClassVar[int]
    SHELL_SNAPSHOT_FIELD_NUMBER: _ClassVar[int]
    context: RequestContext
    command: str
    argv: _containers.RepeatedScalarFieldContainer[str]
    cwd: str
    env: _containers.ScalarMap[str, str]
    tty: bool
    yield_time_ms: int
    max_output_tokens: int
    hard_timeout_s: int
    owner_session_key: str
    shell_snapshot: bool
    def __init__(self, context: _Optional[_Union[RequestContext, _Mapping]] = ..., command: _Optional[str] = ..., argv: _Optional[_Iterable[str]] = ..., cwd: _Optional[str] = ..., env: _Optional[_Mapping[str, str]] = ..., tty: bool = ..., yield_time_ms: _Optional[int] = ..., max_output_tokens: _Optional[int] = ..., hard_timeout_s: _Optional[int] = ..., owner_session_key: _Optional[str] = ..., shell_snapshot: bool = ...) -> None: ...

class WriteStdinRequest(_message.Message):
    __slots__ = ("context", "execution_id", "chars", "yield_time_ms", "max_output_tokens", "owner_session_key")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    CHARS_FIELD_NUMBER: _ClassVar[int]
    YIELD_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    MAX_OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    OWNER_SESSION_KEY_FIELD_NUMBER: _ClassVar[int]
    context: RequestContext
    execution_id: int
    chars: str
    yield_time_ms: int
    max_output_tokens: int
    owner_session_key: str
    def __init__(self, context: _Optional[_Union[RequestContext, _Mapping]] = ..., execution_id: _Optional[int] = ..., chars: _Optional[str] = ..., yield_time_ms: _Optional[int] = ..., max_output_tokens: _Optional[int] = ..., owner_session_key: _Optional[str] = ...) -> None: ...

class ExecutionReply(_message.Message):
    __slots__ = ("output", "wall_time_ms", "original_token_count", "output_omitted_bytes", "execution_id", "exit_code", "unknown_execution_id", "output_path", "finish_reason")
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    WALL_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    ORIGINAL_TOKEN_COUNT_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_OMITTED_BYTES_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    UNKNOWN_EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_PATH_FIELD_NUMBER: _ClassVar[int]
    FINISH_REASON_FIELD_NUMBER: _ClassVar[int]
    output: bytes
    wall_time_ms: int
    original_token_count: int
    output_omitted_bytes: int
    execution_id: int
    exit_code: int
    unknown_execution_id: int
    output_path: str
    finish_reason: str
    def __init__(self, output: _Optional[bytes] = ..., wall_time_ms: _Optional[int] = ..., original_token_count: _Optional[int] = ..., output_omitted_bytes: _Optional[int] = ..., execution_id: _Optional[int] = ..., exit_code: _Optional[int] = ..., unknown_execution_id: _Optional[int] = ..., output_path: _Optional[str] = ..., finish_reason: _Optional[str] = ...) -> None: ...

class StopRequest(_message.Message):
    __slots__ = ("context", "execution_id", "owner_session_key")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    OWNER_SESSION_KEY_FIELD_NUMBER: _ClassVar[int]
    context: RequestContext
    execution_id: int
    owner_session_key: str
    def __init__(self, context: _Optional[_Union[RequestContext, _Mapping]] = ..., execution_id: _Optional[int] = ..., owner_session_key: _Optional[str] = ...) -> None: ...

class StopReply(_message.Message):
    __slots__ = ("stopped",)
    STOPPED_FIELD_NUMBER: _ClassVar[int]
    stopped: bool
    def __init__(self, stopped: bool = ...) -> None: ...

class OwnerRequest(_message.Message):
    __slots__ = ("context", "owner_session_key")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    OWNER_SESSION_KEY_FIELD_NUMBER: _ClassVar[int]
    context: RequestContext
    owner_session_key: str
    def __init__(self, context: _Optional[_Union[RequestContext, _Mapping]] = ..., owner_session_key: _Optional[str] = ...) -> None: ...

class CleanupFailure(_message.Message):
    __slots__ = ("execution_id", "error_type", "message")
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    ERROR_TYPE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    execution_id: int
    error_type: str
    message: str
    def __init__(self, execution_id: _Optional[int] = ..., error_type: _Optional[str] = ..., message: _Optional[str] = ...) -> None: ...

class CleanupReply(_message.Message):
    __slots__ = ("attempted", "cleaned", "failures")
    ATTEMPTED_FIELD_NUMBER: _ClassVar[int]
    CLEANED_FIELD_NUMBER: _ClassVar[int]
    FAILURES_FIELD_NUMBER: _ClassVar[int]
    attempted: _containers.RepeatedScalarFieldContainer[int]
    cleaned: _containers.RepeatedScalarFieldContainer[int]
    failures: _containers.RepeatedCompositeFieldContainer[CleanupFailure]
    def __init__(self, attempted: _Optional[_Iterable[int]] = ..., cleaned: _Optional[_Iterable[int]] = ..., failures: _Optional[_Iterable[_Union[CleanupFailure, _Mapping]]] = ...) -> None: ...

class ActiveExecutionsReply(_message.Message):
    __slots__ = ("execution_ids",)
    EXECUTION_IDS_FIELD_NUMBER: _ClassVar[int]
    execution_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, execution_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class ExecutionState(_message.Message):
    __slots__ = ("execution_id", "settled", "exit_code", "finish_reason")
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    SETTLED_FIELD_NUMBER: _ClassVar[int]
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    FINISH_REASON_FIELD_NUMBER: _ClassVar[int]
    execution_id: int
    settled: bool
    exit_code: int
    finish_reason: str
    def __init__(self, execution_id: _Optional[int] = ..., settled: bool = ..., exit_code: _Optional[int] = ..., finish_reason: _Optional[str] = ...) -> None: ...

class ExecutionStatesReply(_message.Message):
    __slots__ = ("states",)
    STATES_FIELD_NUMBER: _ClassVar[int]
    states: _containers.RepeatedCompositeFieldContainer[ExecutionState]
    def __init__(self, states: _Optional[_Iterable[_Union[ExecutionState, _Mapping]]] = ...) -> None: ...

class FileRequest(_message.Message):
    __slots__ = ("context", "allowed_dir", "read", "write", "edit", "list", "path_info")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    ALLOWED_DIR_FIELD_NUMBER: _ClassVar[int]
    READ_FIELD_NUMBER: _ClassVar[int]
    WRITE_FIELD_NUMBER: _ClassVar[int]
    EDIT_FIELD_NUMBER: _ClassVar[int]
    LIST_FIELD_NUMBER: _ClassVar[int]
    PATH_INFO_FIELD_NUMBER: _ClassVar[int]
    context: RequestContext
    allowed_dir: str
    read: ReadFile
    write: WriteFile
    edit: EditFile
    list: ListDir
    path_info: PathInfo
    def __init__(self, context: _Optional[_Union[RequestContext, _Mapping]] = ..., allowed_dir: _Optional[str] = ..., read: _Optional[_Union[ReadFile, _Mapping]] = ..., write: _Optional[_Union[WriteFile, _Mapping]] = ..., edit: _Optional[_Union[EditFile, _Mapping]] = ..., list: _Optional[_Union[ListDir, _Mapping]] = ..., path_info: _Optional[_Union[PathInfo, _Mapping]] = ...) -> None: ...

class ReadFile(_message.Message):
    __slots__ = ("path", "offset", "limit")
    PATH_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    path: str
    offset: int
    limit: int
    def __init__(self, path: _Optional[str] = ..., offset: _Optional[int] = ..., limit: _Optional[int] = ...) -> None: ...

class WriteFile(_message.Message):
    __slots__ = ("path", "content", "required_dir")
    PATH_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_DIR_FIELD_NUMBER: _ClassVar[int]
    path: str
    content: str
    required_dir: str
    def __init__(self, path: _Optional[str] = ..., content: _Optional[str] = ..., required_dir: _Optional[str] = ...) -> None: ...

class EditFile(_message.Message):
    __slots__ = ("path", "old_text", "new_text", "replace_all")
    PATH_FIELD_NUMBER: _ClassVar[int]
    OLD_TEXT_FIELD_NUMBER: _ClassVar[int]
    NEW_TEXT_FIELD_NUMBER: _ClassVar[int]
    REPLACE_ALL_FIELD_NUMBER: _ClassVar[int]
    path: str
    old_text: str
    new_text: str
    replace_all: bool
    def __init__(self, path: _Optional[str] = ..., old_text: _Optional[str] = ..., new_text: _Optional[str] = ..., replace_all: bool = ...) -> None: ...

class ListDir(_message.Message):
    __slots__ = ("path", "limit", "after")
    PATH_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    AFTER_FIELD_NUMBER: _ClassVar[int]
    path: str
    limit: int
    after: str
    def __init__(self, path: _Optional[str] = ..., limit: _Optional[int] = ..., after: _Optional[str] = ...) -> None: ...

class PathInfo(_message.Message):
    __slots__ = ("action", "path", "base_dir", "after", "limit", "max_bytes")
    ACTION_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    BASE_DIR_FIELD_NUMBER: _ClassVar[int]
    AFTER_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    MAX_BYTES_FIELD_NUMBER: _ClassVar[int]
    action: str
    path: str
    base_dir: str
    after: str
    limit: int
    max_bytes: int
    def __init__(self, action: _Optional[str] = ..., path: _Optional[str] = ..., base_dir: _Optional[str] = ..., after: _Optional[str] = ..., limit: _Optional[int] = ..., max_bytes: _Optional[int] = ...) -> None: ...

class FileReply(_message.Message):
    __slots__ = ("text", "image", "error")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    text: str
    image: FileImage
    error: FileError
    def __init__(self, text: _Optional[str] = ..., image: _Optional[_Union[FileImage, _Mapping]] = ..., error: _Optional[_Union[FileError, _Mapping]] = ...) -> None: ...

class FileError(_message.Message):
    __slots__ = ("text", "is_error")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    IS_ERROR_FIELD_NUMBER: _ClassVar[int]
    text: str
    is_error: bool
    def __init__(self, text: _Optional[str] = ..., is_error: bool = ...) -> None: ...

class FileImage(_message.Message):
    __slots__ = ("text", "mime_type", "data", "detail")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    MIME_TYPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    text: str
    mime_type: str
    data: bytes
    detail: str
    def __init__(self, text: _Optional[str] = ..., mime_type: _Optional[str] = ..., data: _Optional[bytes] = ..., detail: _Optional[str] = ...) -> None: ...

class SkillRequirementsRequest(_message.Message):
    __slots__ = ("context", "bins", "env")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    BINS_FIELD_NUMBER: _ClassVar[int]
    ENV_FIELD_NUMBER: _ClassVar[int]
    context: RequestContext
    bins: _containers.RepeatedScalarFieldContainer[str]
    env: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, context: _Optional[_Union[RequestContext, _Mapping]] = ..., bins: _Optional[_Iterable[str]] = ..., env: _Optional[_Iterable[str]] = ...) -> None: ...

class RequirementNames(_message.Message):
    __slots__ = ("bins", "env")
    BINS_FIELD_NUMBER: _ClassVar[int]
    ENV_FIELD_NUMBER: _ClassVar[int]
    bins: _containers.RepeatedScalarFieldContainer[str]
    env: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, bins: _Optional[_Iterable[str]] = ..., env: _Optional[_Iterable[str]] = ...) -> None: ...

class SkillRequirementsReply(_message.Message):
    __slots__ = ("available", "missing")
    AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    MISSING_FIELD_NUMBER: _ClassVar[int]
    available: RequirementNames
    missing: RequirementNames
    def __init__(self, available: _Optional[_Union[RequirementNames, _Mapping]] = ..., missing: _Optional[_Union[RequirementNames, _Mapping]] = ...) -> None: ...

class TerminalRequest(_message.Message):
    __slots__ = ("context", "owner_session_key", "open", "write", "resize", "read", "close", "list")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    OWNER_SESSION_KEY_FIELD_NUMBER: _ClassVar[int]
    OPEN_FIELD_NUMBER: _ClassVar[int]
    WRITE_FIELD_NUMBER: _ClassVar[int]
    RESIZE_FIELD_NUMBER: _ClassVar[int]
    READ_FIELD_NUMBER: _ClassVar[int]
    CLOSE_FIELD_NUMBER: _ClassVar[int]
    LIST_FIELD_NUMBER: _ClassVar[int]
    context: RequestContext
    owner_session_key: str
    open: TerminalOpen
    write: TerminalWrite
    resize: TerminalResize
    read: TerminalRead
    close: int
    list: bool
    def __init__(self, context: _Optional[_Union[RequestContext, _Mapping]] = ..., owner_session_key: _Optional[str] = ..., open: _Optional[_Union[TerminalOpen, _Mapping]] = ..., write: _Optional[_Union[TerminalWrite, _Mapping]] = ..., resize: _Optional[_Union[TerminalResize, _Mapping]] = ..., read: _Optional[_Union[TerminalRead, _Mapping]] = ..., close: _Optional[int] = ..., list: bool = ...) -> None: ...

class TerminalOpen(_message.Message):
    __slots__ = ("cwd", "cols", "rows")
    CWD_FIELD_NUMBER: _ClassVar[int]
    COLS_FIELD_NUMBER: _ClassVar[int]
    ROWS_FIELD_NUMBER: _ClassVar[int]
    cwd: str
    cols: int
    rows: int
    def __init__(self, cwd: _Optional[str] = ..., cols: _Optional[int] = ..., rows: _Optional[int] = ...) -> None: ...

class TerminalWrite(_message.Message):
    __slots__ = ("terminal_id", "data")
    TERMINAL_ID_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    terminal_id: int
    data: bytes
    def __init__(self, terminal_id: _Optional[int] = ..., data: _Optional[bytes] = ...) -> None: ...

class TerminalResize(_message.Message):
    __slots__ = ("terminal_id", "cols", "rows")
    TERMINAL_ID_FIELD_NUMBER: _ClassVar[int]
    COLS_FIELD_NUMBER: _ClassVar[int]
    ROWS_FIELD_NUMBER: _ClassVar[int]
    terminal_id: int
    cols: int
    rows: int
    def __init__(self, terminal_id: _Optional[int] = ..., cols: _Optional[int] = ..., rows: _Optional[int] = ...) -> None: ...

class TerminalRead(_message.Message):
    __slots__ = ("terminal_id", "offset", "wait_ms")
    TERMINAL_ID_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    WAIT_MS_FIELD_NUMBER: _ClassVar[int]
    terminal_id: int
    offset: int
    wait_ms: int
    def __init__(self, terminal_id: _Optional[int] = ..., offset: _Optional[int] = ..., wait_ms: _Optional[int] = ...) -> None: ...

class TerminalInfo(_message.Message):
    __slots__ = ("terminal_id", "shell", "cwd", "exit_code", "exited")
    TERMINAL_ID_FIELD_NUMBER: _ClassVar[int]
    SHELL_FIELD_NUMBER: _ClassVar[int]
    CWD_FIELD_NUMBER: _ClassVar[int]
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    EXITED_FIELD_NUMBER: _ClassVar[int]
    terminal_id: int
    shell: str
    cwd: str
    exit_code: int
    exited: bool
    def __init__(self, terminal_id: _Optional[int] = ..., shell: _Optional[str] = ..., cwd: _Optional[str] = ..., exit_code: _Optional[int] = ..., exited: bool = ...) -> None: ...

class TerminalOutput(_message.Message):
    __slots__ = ("data", "offset", "exit_code", "exited")
    DATA_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    EXITED_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    offset: int
    exit_code: int
    exited: bool
    def __init__(self, data: _Optional[bytes] = ..., offset: _Optional[int] = ..., exit_code: _Optional[int] = ..., exited: bool = ...) -> None: ...

class TerminalList(_message.Message):
    __slots__ = ("terminals",)
    TERMINALS_FIELD_NUMBER: _ClassVar[int]
    terminals: _containers.RepeatedCompositeFieldContainer[TerminalInfo]
    def __init__(self, terminals: _Optional[_Iterable[_Union[TerminalInfo, _Mapping]]] = ...) -> None: ...

class TerminalReply(_message.Message):
    __slots__ = ("info", "done", "output", "list", "unknown_terminal_id")
    INFO_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    LIST_FIELD_NUMBER: _ClassVar[int]
    UNKNOWN_TERMINAL_ID_FIELD_NUMBER: _ClassVar[int]
    info: TerminalInfo
    done: bool
    output: TerminalOutput
    list: TerminalList
    unknown_terminal_id: int
    def __init__(self, info: _Optional[_Union[TerminalInfo, _Mapping]] = ..., done: bool = ..., output: _Optional[_Union[TerminalOutput, _Mapping]] = ..., list: _Optional[_Union[TerminalList, _Mapping]] = ..., unknown_terminal_id: _Optional[int] = ...) -> None: ...
