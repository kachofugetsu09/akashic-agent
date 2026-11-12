/**
 * 服务端文件浏览与右上角工作台开关的 Monkey Test：
 * 起真实 Python FastAPI dashboard + PathInfoOperation/FileByteAccess 后端，注入随机网络抖动与乱序延迟，
 * 在 Chromium 中用固定种子随机驱动：右上角「文件/工作台」开关、主会话/项目会话切换、多根预设与路径收藏、
 * 手输目录或文件绝对路径、面包屑与上一级越界回退、隐藏项、200+ 项游标分页、筛选、大文件头尾切片、
 * 二进制与套接字保护、行号选择引用及多宽度（320–1365px）缩放，每步核对 DOM/几何/Worker/竞态不变量。
 */
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const repo = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const scratch = mkdtempSync(resolve(tmpdir(), "akashic-file-browser-monkey-"));
const evidence = resolve(process.env.EVIDENCE_DIR ?? scratch);
mkdirSync(evidence, { recursive: true });

const backendPy = resolve(scratch, "backend.py");
writeFileSync(backendPy, `
import asyncio, base64, json, os, socket as sock_mod, sys
from pathlib import Path
from fastapi import FastAPI
import uvicorn

repo = Path(sys.argv[1])
root = Path(sys.argv[2])
sys.path[:0] = [str(repo)]

from plugins.host_execution.files import Files
from plugins.host_execution.contract import FILES
from plugins.working_directory.contract import WORKING_DIRECTORY, DirectorySnapshot
from plugins.markdown_reader import dashboard

PNG_DATA = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAACXBIWXMAAAsTAAALEwEAmpwYAAAA"
    "B3RJTUUH6gsMCh8w5q3zNAAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUH"
    "AAAAQElEQVR42u3PQREAAAgDINc/9K3h4Qc0J6Snp6enp6enp6enp6enp6enp6enp6enp6enp6en"
    "p6enp6enp6fXAwt5AAH/8k6GAAAAAElFTkSuQmCC"
)

ws = root / "workspace"
data_root = ws / "plugin-data" / "markdown_reader@release"
mem = ws / "memory"
repo_dir = root / "project-repo"
home_dir = root / "home"
for d in (data_root, mem / "topics", ws / "assets", ws / "watch", ws / "logs", ws / "paged", repo_dir / "src", home_dir):
    d.mkdir(parents=True, exist_ok=True)
os.environ["HOME"] = str(home_dir)

(mem / "MEMORY.md").write_text("# 长期记忆\\n\\n见 [VEDA](./VEDA.md) 与 ![avatar](../assets/avatar.png)\\n", encoding="utf-8")
(mem / "VEDA.md").write_text("# Veda\\n\\n服务端人格真源。\\n", encoding="utf-8")
(mem / "topics/server.md").write_text("# Server Topic\\n", encoding="utf-8")
(ws / "assets/avatar.png").write_bytes(PNG_DATA)
(ws / "bundle.patch.toml").write_text("[rows.markdown_reader]\\nplugin = 'markdown_reader@release'\\n", encoding="utf-8")
(ws / ".hidden-secret").write_text("DOTFILE=1\\n", encoding="utf-8")
(ws / "watch/health.sh").write_text("#!/usr/bin/env bash\\nset -e\\necho ok\\n", encoding="utf-8")
(ws / "sessions.db").write_bytes(b"SQLite format 3\\x00" + b"\\x01\\x02\\x03\\x00" * 1024)
(ws / "logs/huge.jsonl").write_text(
    "\\n".join(json.dumps({"seq": i, "msg": f"服务端日志行 #{i:04d}"}) for i in range(1, 3800)) + "\\n",
    encoding="utf-8",
)
for i in range(1, 226):
    (ws / "paged" / f"item-{i:03d}.txt").write_text(f"item {i}\\n", encoding="utf-8")

os.mkfifo(str(ws / "akashic.fifo"))
(ws / "broken.lnk").symlink_to(ws / "missing-target")

(repo_dir / "README.md").write_text("# Project Repo\\n\\nSee [app](./src/app.ts)\\n", encoding="utf-8")
(repo_dir / "src/app.ts").write_text(
    "\\n".join(f"export const line{i} = {i};" for i in range(1, 45)) + "\\n", encoding="utf-8"
)
(home_dir / "memo.md").write_text("# Home Memo\\n", encoding="utf-8")

files_service = Files()

class StubDirectories:
    def snapshot(self, session_id: str) -> DirectorySnapshot:
        if session_id == "proj":
            return DirectorySnapshot(str(repo_dir.resolve()), 1)
        return DirectorySnapshot(None, None)

    async def resolve_target(self, session_id: str | None, path: str, *, legacy_base: str | None = None):
        snap = self.snapshot(session_id or "")
        base = snap.path or legacy_base
        async with files_service.paths() as reader:
            info = await reader.read("resolve", path, base_dir=base)
        if info.status != "available":
            raise ValueError(f"unavailable: {info.status}")
        return info.path, snap.path

class StubContext:
    def __init__(self):
        self.data_root = data_root
    def require(self, key):
        if key == FILES:
            return files_service
        if key == WORKING_DIRECTORY:
            return StubDirectories()
        raise KeyError(key)

app = FastAPI()
dashboard.register(app, StubContext())

async def main():
    sock = sock_mod.socket(sock_mod.AF_INET, sock_mod.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    print(json.dumps({
        "port": port,
        "workspace": str(ws.resolve()),
        "repo": str(repo_dir.resolve()),
        "home": str(home_dir.resolve()),
    }), flush=True)
    cfg = uvicorn.Config(app, fd=sock.fileno(), log_level="error")
    await uvicorn.Server(cfg).serve()

asyncio.run(main())
`);

const pyBin = process.env.PYTHON_BIN || "python3";
const backendProc = spawn(pyBin, [backendPy, repo, resolve(scratch, "fs")], { stdio: ["ignore", "pipe", "inherit"] });
const backendInfo = await new Promise((resolveReady, rejectReady) => {
  let buf = "";
  backendProc.stdout.on("data", (chunk) => {
    buf += chunk.toString("utf8");
    const nl = buf.indexOf("\n");
    if (nl >= 0) resolveReady(JSON.parse(buf.slice(0, nl)));
  });
  backendProc.once("exit", (code) => rejectReady(new Error(`backend exited early: ${code}`)));
});

const random = (seed) => {
  let value = seed >>> 0;
  return () => (value = (value * 1664525 + 1013904223) >>> 0) / 2 ** 32;
};

let jitterRng = random(1337);
let inFlightApi = 0;

// 静态模块与带随机延迟（模拟弱网/乱序到达）的后端代理同源托管。
const htmlPage = `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="/plugins/conversation_ui/web_module.css">
<link rel="stylesheet" href="/plugins/markdown_reader/web_module.css">
<style>
  :root {
    --ak-paper-canvas: #f7f4ee; --ak-paper-sheet: #fbf9f5; --ak-paper-quiet: #efeae0; --ak-paper-raised: #e7e0d3;
    --ak-ink-primary: #1f1d1a; --ak-ink-secondary: #5c574f; --ak-ink-muted: #857f75;
    --ak-rule-subtle: rgba(31,29,26,0.1); --ak-rule-default: rgba(31,29,26,0.18);
    --ak-rule-strong: rgba(31,29,26,0.34); --ak-rule-focus: #2c5282;
    --ak-radius-xs: 4px; --ak-radius-sm: 8px; --ak-radius-md: 12px; --ak-radius-lg: 16px; --ak-radius-full: 999px;
    --ak-elevation-raised: 0 1px 2px rgba(0,0,0,0.08); --ak-elevation-float: 0 6px 20px rgba(0,0,0,0.12);
    --ak-color-action-primary: #274464; --ak-color-on-action-primary: #ffffff;
    --ak-color-status-warning: #b7791f; --ak-color-status-error: #b83227; --ak-color-shadow: #000000;
    --ak-type-ui: system-ui, sans-serif; --ak-type-reading: Georgia, serif; --ak-type-technical: ui-monospace, monospace;
  }
  html, body { margin: 0; height: 100%; background: var(--ak-paper-canvas); color: var(--ak-ink-primary); font-family: var(--ak-type-ui); }
  #app { height: 100vh; display: flex; flex-direction: column; }
  #mount { flex: 1; min-height: 0; display: flex; }
  #mount > * { flex: 1; min-width: 0; }
  .mock-chat { display: flex; flex-direction: column; height: 100%; min-width: 0; }
  .mock-header { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 8px 12px; border-bottom: 1px solid var(--ak-rule-subtle); }
  .mock-actions { display: flex; gap: 4px; }
  .mock-actions button { min-width: 36px; min-height: 36px; border: 1px solid var(--ak-rule-subtle); border-radius: 8px; background: var(--ak-paper-sheet); cursor: pointer; }
  .mock-actions button[aria-pressed="true"] { background: var(--ak-paper-raised); border-color: var(--ak-rule-strong); }
</style>
<script type="importmap">
{
  "imports": {
    "@akashic/chat-app-v1": "/chat-stub.js"
  }
}
</script>
</head>
<body>
<div id="app"><div id="mount"></div></div>
<script type="module">
  window.workerStats = { live: 0, max: 0, created: 0 };
  const NativeWorker = window.Worker;
  window.Worker = class extends NativeWorker {
    constructor(...args) {
      super(...args);
      window.workerStats.created += 1;
      window.workerStats.live += 1;
      window.workerStats.max = Math.max(window.workerStats.max, window.workerStats.live);
    }
    terminate() {
      if (!this.__terminated) {
        this.__terminated = true;
        window.workerStats.live -= 1;
      }
      super.terminate();
    }
  };

  const { activate: activateConv } = await import("/plugins/conversation_ui/web_module.js");
  const { activate: activateFiles } = await import("/plugins/markdown_reader/web_module.js");

  const entries = [];
  let convEntry = null;
  const ctx = {
    http: { request: (url, init) => fetch(url, init) },
    data: {},
    ui: {
      inject(mountId, fn) {
        if (mountId === "conversation.tools.v2") return fn({ register: (e) => { entries.push(e); return () => {}; } });
        if (mountId === "shell.pages.v1") return fn({ register: (e) => { convEntry = e; return () => {}; } });
        return () => {};
      }
    }
  };
  activateFiles(ctx);
  activateConv(ctx);
  const view = {
    child(id) {
      return {
        entries: id === "conversation.tools.v2" ? entries : [],
        render(entryId, host, props) {
          const found = entries.find((item) => item.id === entryId);
          return found.render(host, { child: () => ({ entries: [], render: () => () => {}, style: () => () => {} }) }, props);
        },
        style: () => () => {},
      };
    }
  };
  convEntry.render(document.getElementById("mount"), view, { railActions: [] });
</script>
</body>
</html>`;

const chatStubJs = `
export function mountChat(host, bridge) {
  window.__bridge = bridge;
  window.__quotes = [];
  window.__intents = [];
  window.__headerActions = [];
  window.__session = "main";
  const wrap = document.createElement("div");
  wrap.className = "mock-chat";
  const header = document.createElement("header");
  header.className = "mock-header conversation-heading";
  const title = document.createElement("strong");
  title.id = "session-label";
  title.textContent = "Session: main";
  const actionsBox = document.createElement("div");
  actionsBox.className = "mock-actions";
  header.append(title, actionsBox);
  const textarea = document.createElement("textarea");
  textarea.style.margin = "12px";
  textarea.rows = 4;
  wrap.append(header, textarea);
  host.replaceChildren(wrap);

  function paintActions() {
    actionsBox.replaceChildren();
    for (const item of window.__headerActions) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.dataset.headerAction = item.id;
      btn.setAttribute("aria-label", item.hint || item.label);
      if (item.active !== undefined) btn.setAttribute("aria-pressed", String(item.active));
      btn.innerHTML = item.iconSvg;
      btn.onclick = () => bridge.activateHeaderAction(item.id);
      actionsBox.append(btn);
    }
  }

  window.__setSession = (next) => {
    window.__session = next;
    title.textContent = "Session: " + (next || "(none)");
    bridge.reportSession(next);
  };
  window.__openIntent = (name, payload) => bridge.requestIntent(name, window.__session, payload);
  queueMicrotask(() => bridge.reportSession("main"));

  return {
    setRailActions() {},
    setHeaderActions(items) { window.__headerActions = items; paintActions(); },
    setShortcuts() {},
    setWorkbenchIntents(names) { window.__intents = names; },
    quote(q) { window.__quotes.push(q); textarea.value = q.source + "\\n" + q.text; },
    shown() {},
    dispose() {},
  };
}
`;

const server = createServer(async (req, res) => {
  const url = new URL(req.url, "http://127.0.0.1");
  if (url.pathname === "/") {
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    res.end(htmlPage);
    return;
  }
  if (url.pathname === "/chat-stub.js") {
    res.writeHead(200, { "Content-Type": "text/javascript; charset=utf-8" });
    res.end(chatStubJs);
    return;
  }
  if (url.pathname.startsWith("/plugins/")) {
    const file = resolve(repo, "." + url.pathname);
    if (!file.startsWith(resolve(repo, "plugins") + "/")) {
      res.writeHead(403).end();
      return;
    }
    const content = readFileSync(file);
    res.writeHead(200, { "Content-Type": file.endsWith(".css") ? "text/css" : "text/javascript" });
    res.end(content);
    return;
  }
  if (url.pathname.startsWith("/api/dashboard/markdown_reader/")) {
    inFlightApi += 1;
    try {
      const delayMs = Math.floor(jitterRng() * 95);
      if (delayMs > 0) await new Promise((r) => setTimeout(r, delayMs));
      const upstream = await fetch(`http://127.0.0.1:${backendInfo.port}${url.pathname}${url.search}`);
      const buf = Buffer.from(await upstream.arrayBuffer());
      const headers = {};
      upstream.headers.forEach((v, k) => { headers[k] = v; });
      res.writeHead(upstream.status, headers);
      res.end(buf);
    } catch (err) {
      if (!res.headersSent) res.writeHead(502, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "io_error", message: String(err) }));
    } finally {
      inFlightApi -= 1;
    }
    return;
  }
  res.writeHead(404).end();
});

await new Promise((r) => server.listen(0, "127.0.0.1", r));
const origin = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH ?? "/usr/bin/chromium",
  headless: true,
  args: ["--no-sandbox"],
});

async function waitQuiet(page) {
  const deadline = Date.now() + 8000;
  while (Date.now() < deadline) {
    if (inFlightApi === 0) {
      await page.waitForTimeout(35);
      if (inFlightApi === 0) return;
    }
    await page.waitForTimeout(25);
  }
}

async function checkInvariants(page, tag) {
  const audit = await page.evaluate(() => {
    const problems = [];
    const vw = window.innerWidth;
    const reader = document.querySelector(".md-reader");
    if (reader && reader.checkVisibility?.()) {
      const rr = reader.getBoundingClientRect();
      if (rr.right > vw + 1.5) problems.push(`reader overflows viewport: ${rr.right} > ${vw}`);
      const crumbs = [...reader.querySelectorAll(".md-crumbs .md-crumb")].filter((c) => c.checkVisibility?.());
      for (let i = 1; i < crumbs.length; i++) {
        const a = crumbs[i - 1].getBoundingClientRect();
        const b = crumbs[i].getBoundingClientRect();
        if (b.left < a.right - 0.5) problems.push(`crumb overlap: ${crumbs[i - 1].textContent} vs ${crumbs[i].textContent}`);
      }
      const nav = reader.querySelector(".md-nav");
      const list = reader.querySelector(".md-list");
      if (nav && nav.checkVisibility?.() && list) {
        const lh = list.getBoundingClientRect().height;
        if (lh < 80) problems.push(`.md-list crushed to ${lh}px`);
      }
      for (const special of reader.querySelectorAll(".md-entry[data-kind='other']")) {
        if (!special.disabled) problems.push(`special file entry not disabled: ${special.textContent}`);
      }
    }
    return {
      problems,
      workers: window.workerStats,
      headerActions: window.__headerActions.map((a) => ({ id: a.id, active: a.active })),
    };
  });
  assert.deepEqual(audit.problems, [], `${tag} invariant violation: ${JSON.stringify(audit)}`);
  assert.ok(audit.workers.live <= 1, `${tag} leaked syntax workers: ${JSON.stringify(audit.workers)}`);
  assert.ok(audit.headerActions.some((a) => a.id === "tool:markdown-reader"), `${tag} missing header file button`);
  assert.ok(audit.headerActions.some((a) => a.id === "workbench"), `${tag} missing header workbench button`);
}

const WIDTHS = [
  { width: 1365, height: 860 },
  { width: 1024, height: 768 },
  { width: 760, height: 900 },
  { width: 390, height: 844 },
  { width: 320, height: 680 },
];

const summary = [];

try {
  for (const seed of [20261112, 42, 998244353]) {
    jitterRng = random(seed ^ 0x5a5a5a5a);
    const rng = random(seed);
    const pick = (arr) => arr[Math.floor(rng() * arr.length)];
    const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 }, hasTouch: seed === 42 });
    const page = await ctx.newPage();
    page.setDefaultTimeout(5000);
    const errors = [];
    page.on("pageerror", (err) => errors.push(err.message));

    await page.goto(origin + "/");
    await page.locator("button[data-header-action='tool:markdown-reader']").waitFor();
    await page.locator("button[data-header-action='tool:markdown-reader']").click();
    await waitQuiet(page);
    await checkInvariants(page, `seed=${seed}:init`);

    // 确定性冒烟：验证并发乱序竞态下旧请求绝不覆盖新打开的文件。
    await page.evaluate((ws) => {
      void window.__openIntent("workspace.file.open.v1", { path: `${ws}/logs/huge.jsonl` });
      void window.__openIntent("workspace.document.open.v1", { path: `${ws}/memory/VEDA.md` });
    }, backendInfo.workspace);
    await waitQuiet(page);
    assert.equal(await page.locator(".md-name").textContent(), "VEDA.md", `seed=${seed}: 最新意图应胜出`);

    const counts = {};
    const ACTIONS = [
      "header_file", "switch_session", "root_chip", "crumb", "entry",
      "submit_path", "pin", "hidden", "filter", "view_mode", "quote", "intent", "resize",
    ];
    const PATH_SAMPLES = [
      backendInfo.workspace,
      `${backendInfo.workspace}/memory/MEMORY.md`,
      `${backendInfo.workspace}/paged`,
      `${backendInfo.workspace}/logs/huge.jsonl`,
      `${backendInfo.workspace}/sessions.db`,
      `${backendInfo.workspace}/assets/avatar.png`,
      `${backendInfo.repo}/src/app.ts`,
      "~/memo.md",
      "/nonexistent/server/path-404",
    ];

    for (let step = 1; step <= 60; step += 1) {
      const action = pick(ACTIONS);
      counts[action] = (counts[action] ?? 0) + 1;
      try {
        const panelOpen = await page.locator(".md-reader").isVisible().catch(() => false);
        if (!panelOpen && action !== "header_file" && action !== "intent" && action !== "resize") {
          await page.locator("button[data-header-action='tool:markdown-reader']").click();
        } else if (action === "header_file") {
          await page.locator("button[data-header-action='tool:markdown-reader']").click();
        } else if (action === "switch_session") {
          const next = pick(["main", "proj", "other", ""]);
          await page.evaluate((s) => window.__setSession(s), next);
        } else if (action === "root_chip") {
          const navBtn = page.locator(".md-bar button", { hasText: "目录" });
          if (await navBtn.getAttribute("aria-pressed") === "false") await navBtn.click();
          const chips = page.locator(".md-roots .md-root-chip");
          const n = await chips.count();
          if (n > 0) await chips.nth(Math.floor(rng() * n)).click({ timeout: 1500 });
        } else if (action === "crumb") {
          const navBtn = page.locator(".md-bar button", { hasText: "目录" });
          if (await navBtn.getAttribute("aria-pressed") === "false") await navBtn.click();
          const crumbs = page.locator(".md-crumbs .md-crumb");
          const n = await crumbs.count();
          if (n > 0) await crumbs.nth(Math.floor(rng() * n)).click({ timeout: 1500 });
        } else if (action === "entry") {
          const navBtn = page.locator(".md-bar button", { hasText: "目录" });
          if (await navBtn.getAttribute("aria-pressed") === "false") await navBtn.click();
          const rows = page.locator(".md-list .md-entry:not([disabled])");
          const n = await rows.count();
          if (n > 0) await rows.nth(Math.floor(rng() * n)).click({ timeout: 1500 });
        } else if (action === "submit_path") {
          const navBtn = page.locator(".md-bar button", { hasText: "目录" });
          if (await navBtn.getAttribute("aria-pressed") === "false") await navBtn.click();
          const pathBtn = page.locator(".md-nav-actions button", { hasText: "路径" });
          if (await pathBtn.getAttribute("aria-pressed") === "false") await pathBtn.click();
          await page.getByRole("textbox", { name: "服务端目录或文件路径" }).fill(pick(PATH_SAMPLES));
          await page.locator(".md-open button[type='submit']").click({ timeout: 1500 });
        } else if (action === "pin") {
          const navBtn = page.locator(".md-bar button", { hasText: "目录" });
          if (await navBtn.getAttribute("aria-pressed") === "false") await navBtn.click();
          const pin = page.locator(".md-nav-actions button", { hasText: "收藏" });
          if (await pin.isVisible()) await pin.click({ timeout: 1500 });
        } else if (action === "hidden") {
          const navBtn = page.locator(".md-bar button", { hasText: "目录" });
          if (await navBtn.getAttribute("aria-pressed") === "false") await navBtn.click();
          await page.locator(".md-filter-row button", { hasText: "隐藏项" }).click({ timeout: 1500 });
        } else if (action === "filter") {
          const navBtn = page.locator(".md-bar button", { hasText: "目录" });
          if (await navBtn.getAttribute("aria-pressed") === "false") await navBtn.click();
          await page.getByRole("searchbox", { name: "筛选当前目录" }).fill(pick(["", "mem", "item-1", "不存在"]));
        } else if (action === "view_mode") {
          const candidates = page.locator(".md-tools button:visible, .md-banner button:visible");
          const n = await candidates.count();
          if (n > 0) await candidates.nth(Math.floor(rng() * n)).click({ timeout: 1500 });
        } else if (action === "quote") {
          const lines = page.locator(".md-code-ln");
          const n = await lines.count();
          if (n > 0) {
            await lines.nth(Math.floor(rng() * Math.min(n, 20))).click({ timeout: 1500 });
            const q = page.locator(".md-tools button", { hasText: "引用" });
            if (await q.isVisible()) await q.click({ timeout: 1500 });
          }
        } else if (action === "intent") {
          const target = pick([
            { name: "workspace.file.open.v1", payload: { path: `${backendInfo.repo}/src/app.ts`, line: 15 } },
            { name: "workspace.document.open.v1", payload: { path: `${backendInfo.workspace}/memory/MEMORY.md` } },
            { name: "workspace.file.open.v1", payload: { path: `${backendInfo.workspace}/watch/health.sh`, line: 2 } },
          ]);
          await page.evaluate(async ({ name, payload }) => {
            if (!window.__session) window.__setSession("main");
            await window.__openIntent(name, payload);
          }, target);
        } else if (action === "resize") {
          await page.setViewportSize(pick(WIDTHS));
        }
      } catch {
        // 随机切换瞬间按钮因面板收起不可点属于正常 monkey 空操作。
      }
      if (step % 5 === 0 || step === 60) {
        await waitQuiet(page);
        assert.deepEqual(errors, [], `seed=${seed} step=${step} pageerror: ${errors.join("; ")}`);
        await checkInvariants(page, `seed=${seed}:step=${step}`);
      }
    }
    await page.screenshot({ path: resolve(evidence, `monkey-seed-${seed}.png`) });
    summary.push({ seed, steps: 60, counts });
    await ctx.close();
  }
  writeFileSync(resolve(evidence, "monkey-results.json"), JSON.stringify(summary, null, 2));
  console.log(JSON.stringify({ status: "ok", summary }, null, 2));
} finally {
  await browser.close();
  server.close();
  backendProc.kill("SIGTERM");
}
