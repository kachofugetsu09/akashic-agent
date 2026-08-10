from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO, cast

import qrcode

from scripts.akashic_release.doctor import read_environment

RequestJson = Callable[[str, str, Mapping[str, object] | None], dict[str, object]]


def pair_mobile(
    environment_file: Path,
    *,
    input_fn: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
    request_json: RequestJson | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, object]:
    """Create, render, confirm, and consume one mobile pairing offer."""

    # 1. 只访问当前 release 的 loopback WebChat 管理入口
    environment = read_environment(environment_file)
    base_url = _loopback_base_url(environment)
    request = request_json or _request_json
    offer = request("POST", f"{base_url}/api/chat/mobile-pairing", None)
    pairing_id = _text(offer, "pairing_id")
    expires_at = _timestamp(offer, "expires_at")

    # 2. 使用锁定依赖生成终端二维码，不打印 secret 原文
    _print_qr(offer, output)
    print(f"二维码有效期至 {expires_at.isoformat()}（约 8 分钟）。", file=output)
    print("请用 Akashic Android 扫码，随后核对 6 位确认码。", file=output)

    # 3. 等待已验签设备 claim，并要求 operator 输入相同确认码
    claim = _wait_for_claim(
        request=request,
        base_url=base_url,
        pairing_id=pairing_id,
        expires_at=expires_at,
        sleep=sleep,
        now=now,
    )
    confirmation_code = _text(claim, "confirmation_code")
    device_name = _text(claim, "device_name")
    print(f"等待连接的设备：{device_name}", file=output)
    print(f"服务端确认码：{confirmation_code}", file=output)
    entered = input_fn("确认手机显示相同数字后，输入这 6 位确认码：").strip()
    if entered != confirmation_code:
        raise RuntimeError("确认码未匹配，未批准设备")

    # 4. 由既有 pairing owner 原子消费 secret 并登记设备
    device = request(
        "POST",
        f"{base_url}/api/chat/mobile-pairing/{pairing_id}/approve",
        {"confirmation_code": confirmation_code},
    )
    return {
        "status": "paired",
        "deviceId": _text(device, "device_id"),
        "displayName": _text(device, "display_name"),
    }


def _loopback_base_url(environment: Mapping[str, str]) -> str:
    raw_port = environment.get("AKASHIC_PUBLISHED_WEB_PORT", "2236")
    try:
        port = int(raw_port)
    except ValueError as error:
        raise RuntimeError("AKASHIC_PUBLISHED_WEB_PORT 必须为端口整数") from error
    if not 1 <= port <= 65535:
        raise RuntimeError("AKASHIC_PUBLISHED_WEB_PORT 超出有效范围")
    return f"http://127.0.0.1:{port}"


def _request_json(
    method: str,
    url: str,
    payload: Mapping[str, object] | None,
) -> dict[str, object]:
    """Call the loopback pairing API and require one JSON object response."""

    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {} if body is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            value = json.load(response)
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        raise RuntimeError(f"手机配对 API 调用失败: {method} {url}") from error
    if not isinstance(value, dict):
        raise RuntimeError("手机配对 API 必须返回 JSON object")
    return cast(dict[str, object], value)


def _print_qr(offer: Mapping[str, object], output: TextIO) -> None:
    encoded = json.dumps(offer, ensure_ascii=False, separators=(",", ":"))
    qr_module = cast(Any, qrcode)
    qr = qr_module.QRCode(
        error_correction=qr_module.constants.ERROR_CORRECT_M,
        border=2,
    )
    qr.add_data(encoded)
    qr.make(fit=True)
    qr.print_ascii(out=output, tty=False, invert=True)


def _wait_for_claim(
    *,
    request: RequestJson,
    base_url: str,
    pairing_id: str,
    expires_at: datetime,
    sleep: Callable[[float], None],
    now: Callable[[], datetime],
) -> dict[str, object]:
    """Poll until a verified device claim arrives or the offer expires."""

    path = f"{base_url}/api/chat/mobile-pairing/{pairing_id}"
    while now() < expires_at:
        status = request("GET", path, None)
        state = _text(status, "status")
        if state == "waiting_for_desktop_confirmation":
            return status
        if state != "waiting_for_phone":
            raise RuntimeError(f"未知配对状态: {state}")
        sleep(1.25)
    raise RuntimeError("二维码已过期，未批准设备")


def _text(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise RuntimeError(f"手机配对响应缺少 {key}")
    return item


def _timestamp(value: Mapping[str, object], key: str) -> datetime:
    raw = _text(value, key)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError(f"手机配对响应的 {key} 非法") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeError(f"手机配对响应的 {key} 必须含时区")
    return parsed
