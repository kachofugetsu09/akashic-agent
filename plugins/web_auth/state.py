"""门禁状态：单主人账号、设备会话与本机一次性票据，全部在插件声明的 workspace 目录。"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import time
from collections.abc import Iterator

USERNAME = "akashic"
COOKIE = "akashic_session"
REMEMBER_S = 90 * 86400
BROWSER_SESSION_S = 86400
TICKET_S = 120
_ALPHABET = "23456789abcdefghjkmnpqrstuvwxyz"


@dataclass(frozen=True)
class Session:
    digest: str
    remember: bool
    expires: float


def generate_password() -> str:
    # 三组四位、去掉易混字符；约 59 bit，抄写和手输都不易出错。
    return "-".join("".join(secrets.choice(_ALPHABET) for _ in range(4)) for _ in range(3))


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32).hex()


class OwnerAuth:
    """插件进程与本机命令进程经同一把文件锁读写；写入一律原子替换。"""

    def __init__(self, root: Path) -> None:
        self.root = root

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        with (self.root / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _read(self, name: str) -> dict[str, object]:
        path = self.root / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _write(self, name: str, content: str) -> None:
        # 先写 0600 临时文件再原子替换；读者只会看到完整内容。
        temporary = self.root / f".{name}.{secrets.token_hex(4)}"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temporary, self.root / name)

    # 首次启动创建账号与本机密钥；返回新生成的初始密码，已有账号时返回 None。
    def ensure(self) -> str | None:
        with self._locked():
            if not (self.root / "secret").exists():
                self._write("secret", secrets.token_urlsafe(32))
            if (self.root / "account.json").exists():
                return None
            password = generate_password()
            self._store_password(password, must_change=True)
            self._write("initial-password", password + "\n")
            return password

    def _store_password(self, password: str, *, must_change: bool) -> None:
        salt = secrets.token_bytes(16)
        self._write("account.json", json.dumps({"username": USERNAME, "salt": salt.hex(),
            "hash": _hash_password(password, salt), "mustChange": must_change}))

    def local_secret(self) -> str:
        return (self.root / "secret").read_text(encoding="utf-8").strip()

    def must_change(self) -> bool:
        return bool(self._read("account.json").get("mustChange", True))

    def check_password(self, username: str, password: str) -> bool:
        account = self._read("account.json")
        if not account:
            return False
        expected = str(account["hash"])
        actual = _hash_password(password, bytes.fromhex(str(account["salt"])))
        return secrets.compare_digest(username.encode(), USERNAME.encode()) & secrets.compare_digest(actual, expected)

    # 设置主人自己的密码：初始密码文件删除，除 keep 这个会话之外全部退出。
    def set_password(self, password: str, *, keep: str | None) -> None:
        if len(password) < 10:
            raise ValueError("密码至少 10 个字符")
        with self._locked():
            self._store_password(password, must_change=False)
            (self.root / "initial-password").unlink(missing_ok=True)
            sessions = self._read("sessions.json")
            kept = None if keep is None else _digest(keep)
            self._write("sessions.json", json.dumps({k: v for k, v in sessions.items() if k == kept}))

    # 忘记密码：换成临时密码并让所有设备退出，登录后必须再改。
    def reset_password(self) -> str:
        password = generate_password()
        with self._locked():
            self._store_password(password, must_change=True)
            self._write("sessions.json", "{}")
        return password

    def issue_ticket(self) -> str:
        ticket = secrets.token_urlsafe(24)
        with self._locked():
            tickets = {k: v for k, v in self._read("tickets.json").items() if float(str(v)) > time.time()}
            tickets[_digest(ticket)] = time.time() + TICKET_S
            self._write("tickets.json", json.dumps(tickets))
        return ticket

    def consume_ticket(self, ticket: str) -> bool:
        with self._locked():
            tickets = self._read("tickets.json")
            expires = tickets.pop(_digest(ticket), None)
            self._write("tickets.json", json.dumps(tickets))
        return expires is not None and float(str(expires)) > time.time()

    def create_session(self, *, remember: bool, label: str) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self._locked():
            sessions = self._read("sessions.json")
            sessions[_digest(token)] = {"remember": remember, "label": label[:200],
                                        "created": now}
            self._write("sessions.json", json.dumps(sessions))
        return token

    # 会话从签发起固定有效期，与浏览器 cookie 一致；不随使用续期。
    def session(self, token: str) -> Session | None:
        key = _digest(token)
        entry = self._read("sessions.json").get(key)
        if not isinstance(entry, dict):
            return None
        remember = bool(entry["remember"])
        expires = float(entry["created"]) + (REMEMBER_S if remember else BROWSER_SESSION_S)
        if time.time() >= expires:
            self.revoke(token)
            return None
        return Session(key, remember, expires)

    def revoke(self, token: str) -> None:
        with self._locked():
            sessions = self._read("sessions.json")
            sessions.pop(_digest(token), None)
            self._write("sessions.json", json.dumps(sessions))
