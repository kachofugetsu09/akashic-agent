"""验证 stage_plugin_package 的发布制品边界：排除前端源码与 node_modules，保留运行资产并精准覆盖本轮生成产物。"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from scripts.build_plugin_distribution import stage_plugin_package


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        },
    )
    return proc.stdout.strip()


def test_stage_plugin_package_excludes_web_and_node_modules_and_overlays_generated(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")

    plugin = repo / "plugins" / "demo"
    (plugin / "web" / "src").mkdir(parents=True)
    (plugin / "web" / "node_modules" / "pkg").mkdir(parents=True)
    (plugin / "skills" / "demo" / "web").mkdir(parents=True)
    (plugin / "plugin.py").write_text('api_version = 3\nname = "demo"\nversion = "1.0.0"\ndesc = "d"\n', encoding="utf-8")
    (plugin / "web" / "src" / "index.tsx").write_text("export const x = 1;\n", encoding="utf-8")
    (plugin / "web" / "node_modules" / "pkg" / "index.js").write_text("module.exports = 1;\n", encoding="utf-8")
    (plugin / "skills" / "demo" / "web" / "guide.md").write_text("# skill web guide\n", encoding="utf-8")
    (plugin / "web_module.js").write_text("/* hand-written */\n", encoding="utf-8")
    (plugin / "frame.html").write_text("OLD_FRAME\n", encoding="utf-8")

    _git(repo, "add", "-A", "-f")
    _git(repo, "commit", "-q", "-m", "init")
    commit = _git(repo, "rev-parse", "HEAD")

    generated_root = tmp_path / "generated"
    gen_demo = generated_root / "demo"
    (gen_demo / "static" / "chunks").mkdir(parents=True)
    (gen_demo / "runner_ui.js").write_text("NEW_RUNNER_JS\n", encoding="utf-8")
    (gen_demo / "runner_ui.css").write_text("NEW_RUNNER_CSS\n", encoding="utf-8")
    (gen_demo / "frame.html").write_text("NEW_FRAME_BYTES\n", encoding="utf-8")
    (gen_demo / "static" / "chunks" / "a.js").write_text("CHUNK_A\n", encoding="utf-8")
    (gen_demo / ".akashic-generated-assets.json").write_text(json.dumps(["frame.html"]), encoding="utf-8")

    pkg1 = tmp_path / "pkg1"
    pkg2 = tmp_path / "pkg2"
    pkg1.mkdir()
    pkg2.mkdir()
    stage_plugin_package(repo, commit, "plugins/demo", pkg1, generated_root)
    stage_plugin_package(repo, commit, "plugins/demo", pkg2, generated_root)

    # 1. 排除插件根 web/、node_modules 与构建期清单；保留 skills/.../web/ 与手写 web_module.js。
    assert not (pkg1 / "web").exists()
    assert not (pkg1 / ".akashic-generated-assets.json").exists()
    assert (pkg1 / "skills" / "demo" / "web" / "guide.md").read_text(encoding="utf-8") == "# skill web guide\n"
    assert (pkg1 / "web_module.js").read_text(encoding="utf-8") == "/* hand-written */\n"

    # 2. 本轮构建出的 runner_ui.*、static/chunks/* 与清单声明的 frame.html 必须按新字节覆盖。
    assert (pkg1 / "runner_ui.js").read_text(encoding="utf-8") == "NEW_RUNNER_JS\n"
    assert (pkg1 / "runner_ui.css").read_text(encoding="utf-8") == "NEW_RUNNER_CSS\n"
    assert (pkg1 / "frame.html").read_text(encoding="utf-8") == "NEW_FRAME_BYTES\n"
    assert (pkg1 / "static" / "chunks" / "a.js").read_text(encoding="utf-8") == "CHUNK_A\n"

    # 3. 同一提交两次 staging 逐文件完全一致。
    files1 = {p.relative_to(pkg1).as_posix(): p.read_bytes() for p in sorted(pkg1.rglob("*")) if p.is_file()}
    files2 = {p.relative_to(pkg2).as_posix(): p.read_bytes() for p in sorted(pkg2.rglob("*")) if p.is_file()}
    assert files1 == files2


def test_stage_plugin_package_rejects_dangling_symlink_into_excluded_web(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")

    plugin = repo / "plugins" / "bad"
    (plugin / "web").mkdir(parents=True)
    (plugin / "plugin.py").write_text('api_version = 3\nname = "bad"\nversion = "1.0.0"\ndesc = "b"\n', encoding="utf-8")
    (plugin / "web" / "secret.js").write_text("1\n", encoding="utf-8")
    (plugin / "broken.js").symlink_to("web/secret.js")

    _git(repo, "add", "-A", "-f")
    _git(repo, "commit", "-q", "-m", "init")
    commit = _git(repo, "rev-parse", "HEAD")

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    with pytest.raises(ValueError, match="悬空符号链接"):
        stage_plugin_package(repo, commit, "plugins/bad", pkg, None)


def test_bundle_plugin_keeps_gitignored_generated_assets(tmp_path: Path) -> None:
    """即使插件内带 .gitignore 忽略编译产物，正式 .bundle 也必须包含 staging 覆盖后的生成资产。"""
    from scripts.build_plugin_distribution import _bundle_plugin

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    plugin = repo / "plugins" / "ignored_gen"
    (plugin / "web").mkdir(parents=True)
    (plugin / "plugin.py").write_text('api_version = 3\nname = "ignored_gen"\nversion = "1.0.0"\ndesc = "g"\n', encoding="utf-8")
    (plugin / ".gitignore").write_text("plugin_ui.js\nstatic/\n", encoding="utf-8")
    (plugin / "web" / "index.tsx").write_text("export const x = 1;\n", encoding="utf-8")
    _git(repo, "add", "-A", "-f")
    _git(repo, "commit", "-q", "-m", "init")
    commit = _git(repo, "rev-parse", "HEAD")

    generated_root = tmp_path / "generated"
    gen = generated_root / "ignored_gen"
    (gen / "static" / "chunks").mkdir(parents=True)
    (gen / "plugin_ui.js").write_text("export default { slots: {} };\n", encoding="utf-8")
    (gen / "static" / "chunks" / "c1.js").write_text("export const c = 1;\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    info = _bundle_plugin(repo, commit, "2026-11-27T00:00:00Z", "plugins/ignored_gen", out_dir, set(), generated_root)
    bundles = list(out_dir.rglob("*.bundle"))
    assert len(bundles) == 1 and info

    cloned = tmp_path / "cloned"
    _git(tmp_path, "clone", "-q", str(bundles[0]), str(cloned))
    assert (cloned / "plugin_ui.js").read_text(encoding="utf-8") == "export default { slots: {} };\n"
    assert (cloned / "static" / "chunks" / "c1.js").read_text(encoding="utf-8") == "export const c = 1;\n"
    assert not (cloned / "web").exists()
