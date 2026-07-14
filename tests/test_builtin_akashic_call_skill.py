import json
import queue
import socket
import subprocess
import sys
import threading
from pathlib import Path
from typing import Protocol, cast

from agent.skills import SkillsLoader


REPO_ROOT = Path(__file__).parents[1]
SKILL_ROOT = REPO_ROOT / "skills" / "akashic-call"


class _FrameStream(Protocol):
    def readline(self) -> bytes: ...

    def write(self, data: bytes, /) -> int: ...

    def flush(self) -> None: ...


def test_akashic_call_is_discoverable_builtin(tmp_path: Path) -> None:
    loader = SkillsLoader(tmp_path, builtin_skills_dir=REPO_ROOT / "skills")

    record = loader.load_skill_record("akashic-call")

    assert record is not None
    assert record.source == "builtin"
    assert record.available is True
    assert record.always is False
    assert record.when_to_use
    for trigger in (
        "调用 akashic",
        "程序化调用 Akashic",
        "从 Codex 调用 Akashic",
        "外部自动化调用",
        "复用 Akashic session/thread",
    ):
        assert trigger in record.description


def test_akashic_call_content_preserves_runtime_boundaries(tmp_path: Path) -> None:
    loader = SkillsLoader(tmp_path, builtin_skills_dir=REPO_ROOT / "skills")
    body = loader.load_skill_body("akashic-call")

    assert body is not None
    for contract in (
        "固定模型、固定 workspace",
        "`Thread` 是持久\nsession",
        "禁止同步执行同 workspace",
        "形成自死锁",
        "不同 workspace、不同 runtime endpoint",
        "禁止使用 `--last`",
        "不会自动发送到 Telegram",
    ):
        assert contract in body


def test_akashic_call_examples_are_complete_and_referenced(tmp_path: Path) -> None:
    loader = SkillsLoader(tmp_path, builtin_skills_dir=REPO_ROOT / "skills")
    body = loader.load_skill_body("akashic-call")
    guide = (SKILL_ROOT / "references" / "external-caller.md").read_text(encoding="utf-8")
    raw_client = SKILL_ROOT / "examples" / "raw_jsonrpc_uds.py"

    assert body is not None
    assert "references/external-caller.md" in body
    assert "examples/raw_jsonrpc_uds.py" in body
    assert raw_client.is_file()
    for command in (
        "exec \\",
        '--thread "$AKASHIC_THREAD_ID"',
        "Akashic.connect(endpoint)",
        "thread_resume(os.environ[\"AKASHIC_THREAD_ID\"])",
        '"method":"initialize"',
        '"method":"thread/resume"',
        '"method":"turn/start"',
    ):
        assert command in guide
    assert "自动化不得用“最近一次会话”" in guide
    assert "Akashic 首次 turn 执行失败" in guide
    assert "Akashic JSONL 中缺少 threadId" in guide
    assert 'printf \'%s\\n\' "$AKASHIC_THREAD_ID" > "$AKASHIC_THREAD_FILE"' in guide
    assert "--timeout 600" in guide
    compile(raw_client.read_text(encoding="utf-8"), str(raw_client), "exec")


def _read_frame(stream: _FrameStream) -> dict[str, object]:
    payload = json.loads(stream.readline())
    if not isinstance(payload, dict):
        raise ValueError("request frame must be an object")
    return cast(dict[str, object], payload)


def _write_frame(stream: _FrameStream, payload: dict[str, object]) -> None:
    _ = stream.write(json.dumps(payload, separators=(",", ":")).encode() + b"\n")
    stream.flush()


def test_raw_client_buffers_terminal_arriving_before_turn_response(tmp_path: Path) -> None:
    endpoint = tmp_path / "fake-akashic.sock"
    ready = threading.Event()
    failures: queue.SimpleQueue[BaseException] = queue.SimpleQueue()

    def serve() -> None:
        """模拟在 turn/start response 前发出终态的合法服务端。"""

        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(str(endpoint))
                listener.listen(1)
                ready.set()
                connection, _ = listener.accept()
                with connection, connection.makefile("rwb") as stream:
                    initialize = _read_frame(stream)
                    _write_frame(stream, {"jsonrpc": "2.0", "id": initialize["id"], "result": {}})
                    assert _read_frame(stream)["method"] == "initialized"

                    resume = _read_frame(stream)
                    _write_frame(
                        stream,
                        {
                            "jsonrpc": "2.0",
                            "id": resume["id"],
                            "result": {"id": "programmatic:test"},
                        },
                    )

                    start = _read_frame(stream)
                    _write_frame(
                        stream,
                        {
                            "jsonrpc": "2.0",
                            "method": "turn/completed",
                            "params": {
                                "threadId": "programmatic:test",
                                "turnId": "turn:fast",
                                "turn": {
                                    "id": "turn:fast",
                                    "status": "completed",
                                    "finalResponse": "ok",
                                },
                            },
                        },
                    )
                    _write_frame(
                        stream,
                        {
                            "jsonrpc": "2.0",
                            "id": start["id"],
                            "result": {"id": "turn:fast"},
                        },
                    )
        except BaseException as exc:
            failures.put(exc)

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    assert ready.wait(timeout=2)

    completed = subprocess.run(
        [
            sys.executable,
            str(SKILL_ROOT / "examples" / "raw_jsonrpc_uds.py"),
            str(endpoint),
            "--thread",
            "programmatic:test",
            "--timeout",
            "2",
            "fast turn",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    server.join(timeout=2)

    if not failures.empty():
        raise failures.get()
    assert not server.is_alive()
    assert completed.returncode == 0, completed.stderr
    assert '"method": "turn/completed"' in completed.stdout
