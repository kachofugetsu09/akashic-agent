"""发布制品只能含选定宿主路径与各插件自己的源码。"""
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from agent.plugins.install import install_git_plugin
from scripts.build_plugin_distribution import build


def test_distribution_installs_isolated_git_sources_and_refuses_overwrite(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("one", "two"):
        root = source / "plugins" / name
        root.mkdir(parents=True)
        (root / "plugin.py").write_text(f'api_version = 3\nname = "{name}"\nversion = "1"\ndef apply(ctx, config): pass\n')
        (root / "akashic.plugin.toml").write_text(f'schema_version = 1\napi_version = 3\nname = "{name}"\nversion = "1"\nentrypoint = "plugin.py"\n')
    (source / "main.py").write_text('print("core")\n')
    (source / "private.txt").write_text("must not ship")
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "-c", "commit.gpgSign=false", "commit", "-m", "source"], check=True, capture_output=True)
    output = tmp_path / "release"
    report = build(source, "HEAD", output)
    with tarfile.open(fileobj=io.BytesIO((output / "core.tar").read_bytes())) as archive:
        assert archive.getnames() == ["main.py"]
    assert {row["name"] for row in report["plugins"]} == {"one", "two"}
    for row in report["plugins"]:
        installed = install_git_plugin(workspace=tmp_path / "workspace", plugins_home=tmp_path / "home",
            source=str(output / row["file"]), marketplace="distribution")
        assert installed.source_revision == row["source_revision"]
        assert not (installed.installed_path / "plugins").exists()
        assert not (installed.installed_path / "private.txt").exists()
        provenance = json.loads((installed.installed_path / ".akashic-source.json").read_text())
        assert provenance == {"commit": report["source_commit"], "path": row["source_path"]}
    with pytest.raises(FileExistsError):
        build(source, "HEAD", output)
