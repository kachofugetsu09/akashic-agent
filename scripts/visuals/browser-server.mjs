// 仅回放 visuals_scenario.py 导出的实际消息、附件和安装资产。
import { existsSync } from "node:fs";
import { createServer } from "node:http";
import { readFile, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve, join, extname } from "node:path";
import { build } from "vite";
import react from "@vitejs/plugin-react";

const repo = process.cwd();
const evidence = resolve(process.argv[2]);
const port = Number(process.argv[3] ?? 4330);
const output = await mkdtemp(join(tmpdir(), "akashic-visuals-browser-"));
await build({ configFile: false, root: repo, plugins: [react()],
  worker: { format: "es" },
  resolve: { alias: { "@": resolve("frontend/chat/src"), mermaid: resolve("frontend/chat/src/mermaid-stub.ts") } },
  build: { outDir: output, emptyOutDir: false, target: "es2022",
    rollupOptions: { input: resolve("scripts/visuals/browser-entry.tsx"), output: { entryFileNames: "app.js", assetFileNames: "[name][extname]" } } } });
const catalog = await readFile(join(evidence, "catalog.json"));
const html = `<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Akashic · 可视化插件验收</title><link rel="stylesheet" href="/browser-entry.css"><style>nav{display:flex;flex-wrap:wrap;gap:8px;padding:12px;border-bottom:1px solid var(--ak-color-border-default)}nav button{min-height:44px;padding:8px;border:1px solid var(--ak-color-border-default)}main{max-width:760px;margin:20px auto;padding:0 8px}</style><div id="root"></div><script type="module" src="/app.js"></script></html>`;
let rejectFrame = false;
const server = createServer(async (req, res) => {
  const url = new URL(req.url, "http://127.0.0.1");
  try {
    if (url.pathname === "/favicon.ico") { res.writeHead(204).end(); return; }
    if (url.pathname === "/") { res.setHeader("content-type", "text/html; charset=utf-8"); res.end(html); return; }
    if (url.pathname === "/visuals/frame.html") {
      if (rejectFrame) { res.writeHead(403).end("forbidden_contract"); return; }
      // 与 plugins/visuals/dashboard.py 相同：静态入口由响应 CSP 强制沙箱，文档经 postMessage 交付。
      res.setHeader("content-type", "text/html; charset=utf-8");
      res.setHeader("content-security-policy", "sandbox allow-scripts; default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; font-src data:; connect-src 'none'; frame-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'self'");
      res.end(await readFile(resolve("plugins/visuals/bootstrap.html"))); return;
    }
    if (url.pathname === "/api/chat/plugin-ui/catalog") { res.setHeader("content-type", "application/json"); res.end(catalog); return; }
    const route = url.pathname === "/api/chat/plugin-ui/asset" ? "module.js"
      : url.pathname.startsWith("/api/chat/artifacts/") ? `${url.pathname.split("/").at(-1)}.html`
      : ["/messages.json", "/showcase-messages.json"].includes(url.pathname) ? url.pathname.slice(1) : null;
    // 预检截图按 .png 导出；同名 HTML 不存在时取它。
    const artifact = route?.endsWith(".html") && url.pathname.startsWith("/api/chat/artifacts/");
    const path = artifact && !existsSync(join(evidence, route)) ? join(evidence, route.replace(/\.html$/, ".png"))
      : route ? join(evidence, route) : join(output, url.pathname);
    if (!path.startsWith(route ? evidence + "/" : output + "/")) { res.writeHead(403).end(); return; }
    res.setHeader("content-type", ({ ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".html": "text/html; charset=utf-8", ".woff2": "font/woff2", ".png": "image/png" })[extname(path)] ?? "application/octet-stream");
    res.end(await readFile(path));
  } catch (error) { console.error(error); res.writeHead(500).end(String(error)); }
});
server.listen(port, "127.0.0.1", () => console.log(`Visuals browser acceptance: http://127.0.0.1:${port}`));
for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, () => server.close(async () => { await rm(output, { recursive: true }); process.exit(); }));

if (process.argv.includes("--check")) {
  const { chromium } = await import("playwright-core");
  const { default: assert } = await import("node:assert/strict");
  let browser;
  try {
    browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH });
    // 1. Real saved tool messages render in the production chat components.
    for (const width of [390, 1024]) {
      const page = await browser.newPage({ viewport: { width, height: 844 } });
      await page.goto(`http://127.0.0.1:${port}/?basic`);
      const frame = page.frameLocator(".message-figure iframe");
      await frame.locator("#slope").waitFor();
      await frame.locator("#slope").press("End");
      assert.equal(await frame.locator("#result").textContent(), "4.0");
      assert.equal(await frame.locator("body").evaluate(() => {
        try { parent.document.body; return false; } catch (error) { return error.name === "SecurityError"; }
      }), true);
      assert.equal(await page.locator(".message-figure .attachment-description").count(), 0);
      await page.waitForFunction(() => {
        const figure = document.querySelector(".message-figure");
        const frame = figure?.querySelector("iframe");
        return frame && frame.offsetHeight > 300 && figure.offsetHeight >= frame.offsetHeight;
      });
      await page.screenshot({ path: join(evidence, `chat-${width}.png`), fullPage: true });
      await page.close();
    }
    // 2. A real HTTP 403 reproduces the old bootstrap failure; source stays downloadable.
    rejectFrame = true;
    const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
    const failed = page.waitForResponse(response => response.url().endsWith("/visuals/frame.html"));
    await page.goto(`http://127.0.0.1:${port}/?basic`);
    assert.equal((await failed).status(), 403);
    await page.getByText("交互内容暂不可用：加载超时", { exact: true }).waitFor({ timeout: 15_000 });
    const figure = page.locator(".message-figure");
    assert.equal(await figure.locator(".attachment-description").count(), 0);
    assert.equal(await figure.locator("iframe").count(), 0);
    assert.ok(!(await figure.textContent()).includes("正文不要复述"));
    const download = figure.locator("a.message-figure-download");
    const response = await page.request.get(new URL(await download.getAttribute("href"), page.url()).href);
    assert.equal(response.status(), 200);
    assert.ok((await response.text()).includes('id="slope"'));
    await page.screenshot({ path: join(evidence, "chat-frame-failure.png"), fullPage: true });
    await page.locator(".inline-reply-process summary.process-trigger").click();
    await page.locator("button.tool-step-summary").click();
    await page.getByText(/正文不要复述/).waitFor();
    await page.close();
    console.log(JSON.stringify({ responsive_chat: true, slider: true, http_403_fallback: true,
      no_internal_guidance: true, source_download: true }));
  } finally {
    await browser?.close();
    await new Promise(resolve => server.close(resolve));
    await rm(output, { recursive: true });
  }
}
