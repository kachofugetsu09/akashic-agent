// 实际 Web Host → 安装插件的 HTTP 许可 → 实际分配追踪；不替换产品接口。
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
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
  await page.getByRole("heading", { name: "内存变化", exact: true }).waitFor();
  assert.equal(await page.locator(".diagnostics-line").count(), 2);
  await page.getByLabel("显示 cgroup 曲线").check();
  assert.equal(await page.locator(".diagnostics-line").count(), 3);
  await page.getByLabel("显示 cgroup 曲线").uncheck();
  checks.push("实际宿主加载诊断页面，主进程与 cgroup 曲线可独立查看");

  // 限时停止通过真实计时器；分配通过另一个普通插件的实际诊断入口。
  await page.getByLabel("追踪时长").selectOption("10");
  await page.getByRole("button", { name: "开始剖析", exact: true }).click();
  await page.getByRole("status").filter({ hasText: "正在追踪" }).waitFor();
  assert.equal((await fetch(base + "/__scenario/allocate", { method: "POST" })).status, 200);
  await page.getByRole("status").filter({ hasText: "剖析完成" }).waitFor();
  await page.locator(".diagnostics-allocations").getByText(/diagnostics_probe_allocations\.py:2/).first().waitFor();
  assert.equal(await page.getByRole("alert").count(), 0);
  checks.push("浏览器开启真实剖析，自动停止后可查看实际分配文件");

  const downloadReady = page.waitForEvent("download");
  await page.getByRole("button", { name: "导出诊断", exact: true }).click();
  const download = await downloadReady;
  const reportPath = join(output, download.suggestedFilename());
  await download.saveAs(reportPath);
  const report = JSON.parse(await readFile(reportPath, "utf8"));
  assert.equal(report.schema, "akashic.diagnostics.v1");
  assert.equal(report.profile.state, "complete");
  assert.ok(report.profile.traced_bytes > 24 * 1024 * 1024);
  assert.ok(report.events.some(row => row.operation === "scenario.allocate"));
  checks.push("下载 JSON 保留真实样本、操作和剖析结果");

  // 窄屏与暗色使用真实主题 cookie；轮询失败保留已有数据，恢复后继续读取。
  await page.getByLabel("筛选插件").fill("diagnostics_probe");
  await page.getByRole("button", { name: "跟随最新", exact: true }).click();
  await page.getByRole("heading", { name: "运行诊断", exact: true }).scrollIntoViewIfNeeded();
  await page.screenshot({ path: join(output, "desktop.png"), fullPage: true });
  await page.setViewportSize({ width: 320, height: 1000 });
  assert.equal(await page.evaluate(() => {
    const page = document.querySelector(".diagnostics-page");
    return page.scrollWidth > page.clientWidth || document.documentElement.scrollWidth > innerWidth;
  }), false);
  await page.screenshot({ path: join(output, "mobile.png"), fullPage: true });
  await context.addCookies([{ name: "akashic_theme", value: "dark", url: base }]);
  await page.reload();
  await page.getByRole("heading", { name: "内存变化", exact: true }).waitFor();
  assert.equal(await page.locator("html").getAttribute("data-theme"), "dark");
  await page.screenshot({ path: join(output, "dark.png"), fullPage: true });
  await page.route("**/api/dashboard/runtime-diagnostics?after=*", route => route.fulfill({ status: 503, body: "scenario unavailable" }));
  await page.getByRole("alert").filter({ hasText: "503" }).waitFor();
  assert.ok(await page.locator(".diagnostics-line").count());
  await page.unroute("**/api/dashboard/runtime-diagnostics?after=*");
  await page.getByRole("alert").waitFor({ state: "detached" });
  checks.push("320px 无横向溢出、暗色可用、读取失败保留曲线并可恢复");
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
