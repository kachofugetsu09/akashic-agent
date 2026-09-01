from pathlib import Path

import pytest

from agent.plugins.static_manifest import load_static_plugin_manifest


def _source(*, workload_ref: str = "worker", digest: str | None = None) -> str:
    image_digest = digest or "a" * 64
    return f"""
schema_version = 1
name = "fixture"
version = "1.0.0"
api_version = 3
entrypoint = "plugin.py"

[[workload]]
name = "worker"
image = "example.invalid/worker@sha256:{image_digest}"
command = ["serve"]

[[workload.ports]]
name = "gateway"
number = 8080

[[workload.data]]
name = "state"
target = "/data"
writable = true

[workload.health]
port = "gateway"
path = "/health"
timeout_seconds = 30

[workload.limits]
memory_mb = 128
cpu_count = 1.0
pids = 64

[[mcp]]
name = "fixture"
command = ["mcp.py"]
required_tools = ["read"]
candidate_read_only_tools = ["read"]

[[mcp.workload_env]]
env = "WORKER_URL"
workload = "{workload_ref}"
port = "gateway"
"""


def _plugin(tmp_path: Path, source: str) -> Path:
    root = tmp_path / "fixture"
    root.mkdir()
    (root / "plugin.py").write_text("", encoding="utf-8")
    (root / "mcp.py").write_text("", encoding="utf-8")
    (root / "akashic.plugin.toml").write_text(source, encoding="utf-8")
    return root


def test_static_workload_and_mcp_binding_enter_identity(tmp_path: Path) -> None:
    root = _plugin(tmp_path, _source())

    manifest = load_static_plugin_manifest(root)

    assert manifest.workloads[0].ports == (("gateway", 8080),)
    assert manifest.workloads[0].data == (("state", "/data", True),)
    assert manifest.workloads[0].user_namespaces is False
    assert manifest.mcp_servers[0].workload_env == (
        ("WORKER_URL", "worker", "gateway"),
    )
    original = manifest.identity_digest
    path = root / "akashic.plugin.toml"
    path.write_text(
        _source().replace("memory_mb = 128", "memory_mb = 256"), encoding="utf-8"
    )
    assert load_static_plugin_manifest(root).identity_digest != original


@pytest.mark.parametrize(
    ("memory", "cpu", "pids", "expected"),
    [
        ("0", "1.0", "64", (0, 1.0, 64)),
        ("128", "0.0", "64", (128, 0.0, 64)),
        ("128", "1.0", "0", (128, 1.0, 0)),
        ("0", "0.0", "0", (0, 0.0, 0)),
    ],
)
def test_static_workload_limits_can_be_unlimited_independently(
    tmp_path: Path,
    memory: str,
    cpu: str,
    pids: str,
    expected: tuple[int, float, int],
) -> None:
    source = (
        _source()
        .replace("memory_mb = 128", f"memory_mb = {memory}")
        .replace("cpu_count = 1.0", f"cpu_count = {cpu}")
        .replace("pids = 64", f"pids = {pids}")
    )
    root = _plugin(tmp_path, source)

    manifest = load_static_plugin_manifest(root)

    assert manifest.workloads[0].limits == expected


def test_static_workload_user_namespaces_enter_identity(tmp_path: Path) -> None:
    root = _plugin(tmp_path, _source())
    original = load_static_plugin_manifest(root).identity_digest
    path = root / "akashic.plugin.toml"

    path.write_text(
        _source().replace(
            'command = ["serve"]',
            'command = ["serve"]\nuser_namespaces = true',
        ),
        encoding="utf-8",
    )

    manifest = load_static_plugin_manifest(root)
    assert manifest.workloads[0].user_namespaces is True
    assert manifest.identity_digest != original


def test_static_workload_rejects_unpinned_image(tmp_path: Path) -> None:
    root = _plugin(tmp_path, _source(digest="latest"))

    with pytest.raises(ValueError, match="sha256 digest"):
        load_static_plugin_manifest(root)


def test_static_workload_loopback_port_enters_identity(tmp_path: Path) -> None:
    source = _source().replace("number = 8080", "number = 8080\nloopback = 18080")
    root = _plugin(tmp_path, source)

    manifest = load_static_plugin_manifest(root)

    assert manifest.workloads[0].loopback_ports == (("gateway", 18080),)
    original = manifest.identity_digest
    (root / "akashic.plugin.toml").write_text(
        source.replace("loopback = 18080", "loopback = 18081"),
        encoding="utf-8",
    )
    assert load_static_plugin_manifest(root).identity_digest != original


def test_static_mcp_rejects_unknown_workload_port(tmp_path: Path) -> None:
    root = _plugin(tmp_path, _source(workload_ref="missing"))

    with pytest.raises(ValueError, match="未声明的 Workload"):
        load_static_plugin_manifest(root)
