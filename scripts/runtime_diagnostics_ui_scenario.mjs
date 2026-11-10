// 实际 Web Host 不加载诊断前端，运维 HTTP 和浏览器请求约束保持可观察。
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { mkdtemp, writeFile } from "node:fs/promises";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const root = fileURLToPath(new URL("../", import.meta.url));
const output = await mkdtemp(join(tmpdir(), "akashic-diagnostics-ui-"));
const listener = createServer();
listener.listen(0, "127.0.0.1");
await once(listener, "listening");
const port = listener.address().port;
await new Promise(resolve => listener.close(resolve));
const base = `http://127.0.0.1:${port}`;
const service = spawn(process.env.PYTHON ?? join(root, ".venv/bin/python"), [join(root, "scripts/runtime_diagnostics_scenario.py"), "--port", String(port)],
  { stdio: ["ignore", "pipe", "pipe"] });
let log = "", browser;
service.stderr.on("data", chunk => { log = (log + chunk).slice(-200_000); });
service.stdout.on("data", chunk => { log = (log + chunk).slice(-200_000); });
const checks = [];
try {
  const deadline = Date.now() + 30_000;
  while (true) {
    if (service.exitCode !== null) throw new Error(`夹具退出 ${service.exitCode}: ${log}`);
    try { if ((await fetch(base + "/dashboard")).ok) break; }
    catch (error) { if (!(error instanceof TypeError)) throw error; }
    if (Date.now() > deadline) throw new Error(`夹具启动超时: ${log}`);
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  browser = await chromium.launch({ headless: true,
    ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
  const context = await browser.newContext({ viewport: { width: 1280, height: 1000 }, acceptDownloads: true });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  page.setDefaultTimeout(20_000);
  await page.goto(base + "/dashboard");
  await page.locator(".unified-shell").waitFor();
  await page.locator(".conversation-page-chat .akashic-chat").waitFor();
  assert.equal(await page.getByRole("heading", { name: "运行诊断", exact: true }).count(), 0);
  assert.equal(await page.locator(".product-band").count(), 0);
  const catalog = await (await fetch(base + "/api/chat/web-ui/bootstrap")).json();
  assert.ok(catalog.modules.every(module => module.pluginId !== "runtime_diagnostics@lab"));
  checks.push("真实宿主不加载诊断模块，不显示诊断页面或顶部导航");

  const response = await fetch(base + "/api/dashboard/runtime-diagnostics");
  assert.equal(response.status, 200);
  const report = await response.json();
  assert.equal(report.schema, "akashic.diagnostics.v1");
  assert.ok(report.samples.length > 0);
  assert.equal(await page.evaluate(async () =>
    (await fetch("/api/dashboard/runtime-diagnostics")).status), 403);
  checks.push("运维 HTTP 可读取真实采样，浏览器无模块身份的请求仍被拒绝");
  await page.screenshot({ path: join(output, "desktop.png"), fullPage: true });
  await page.setViewportSize({ width: 320, height: 1000 });
  assert.equal(await page.locator(".product-band").count(), 0);
  await page.screenshot({ path: join(output, "mobile.png"), fullPage: true });
  assert.deepEqual(errors, []);
  await writeFile(join(output, "result.json"), JSON.stringify({ checks, errors }, null, 2));
  console.log(JSON.stringify({ output, checks }));
} finally {
  if (browser) await browser.close();
  if (service.exitCode === null) {
    service.kill("SIGINT");
    await once(service, "exit");
  }
  await writeFile(join(output, "service.log"), log);
}
