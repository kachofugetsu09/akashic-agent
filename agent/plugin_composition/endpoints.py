"""插件声明的监听端点；文件仅是供进程外壳读取的派生视图。"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile


@dataclass(frozen=True, slots=True)
class Endpoint:
    name: str
    protocol: str
    address: str
    routes: tuple[str, ...]
    owner: str
    generation_id: str

    def __post_init__(self) -> None:
        for value in (self.name, self.protocol, self.address):
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError("端点名称、协议和地址必须是非空字符串")
        if not isinstance(self.routes, tuple) or len(set(self.routes)) != len(self.routes):
            raise ValueError("端点 routes 必须是无重复的 tuple")
        for route in self.routes:
            if (not isinstance(route, str) or not route.startswith("/")
                    or "?" in route or "#" in route or "//" in route
                    or (route != "/" and route.endswith("/"))):
                raise ValueError("端点 route 必须是规范路径前缀")


EndpointPublisher = Callable[[tuple[Endpoint, ...]], None]


def save_endpoint_plan(path: Path, generation_id: str, endpoints: tuple[Endpoint, ...]) -> None:
    """先完整写派生视图，再以单次替换提交；替换后没有可能失败的同步步骤。"""
    # 1. Root 调用者尚未更改自己的登记，写失败时原计划仍可读取。
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".endpoints-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"generation_id": generation_id, "endpoints": [asdict(item) for item in endpoints]}, stream)
        # 2. 派生视图无需掉电持久保证；不在替换成功后追加 fsync 失败窗口。
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def load_endpoint_plan(path: Path) -> tuple[Endpoint, ...]:
    """在进程外读取已发布计划；缺席可恢复，损坏必须显式报错。"""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ()
    if (not isinstance(payload, dict) or set(payload) != {"generation_id", "endpoints"}
            or not isinstance(payload["generation_id"], str) or not payload["generation_id"]
            or not isinstance(payload["endpoints"], list)):
        raise ValueError("endpoint plan 结构无效")
    fields = {"name", "protocol", "address", "routes", "owner", "generation_id"}
    endpoints: list[Endpoint] = []
    prefixes: set[str] = set()
    for item in payload["endpoints"]:
        if (not isinstance(item, dict) or set(item) != fields
                or any(not isinstance(item[key], str) or not item[key] for key in fields - {"routes"})
                or not isinstance(item["routes"], list)):
            raise ValueError("endpoint plan 登记无效")
        endpoint = Endpoint(item["name"], item["protocol"], item["address"],
                            tuple(item["routes"]), item["owner"], item["generation_id"])
        if prefixes.intersection(endpoint.routes):
            raise ValueError("endpoint plan 路由前缀重复")
        prefixes.update(endpoint.routes)
        endpoints.append(endpoint)
    return tuple(endpoints)
