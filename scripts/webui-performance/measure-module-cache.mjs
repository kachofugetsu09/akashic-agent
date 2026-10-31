/** 在真实隔离 Runtime 上比较提前重开、HTTP 缓存命中与缓存逐出；不发送消息。 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { chromium } from "playwright-core";

const { values } = parseArgs({ options: {
  url: { type: "string" }, browser: { type: "string" }, output: { type: "string" },
  profile: { type: "string", default: "weak" }, cpu: { type: "string", default: "1" },
} });
for (const key of ["url", "browser", "output"]) assert.ok(values[key], `需要 --${key}`);
assert.ok(["lan", "weak"].includes(values.profile));
const cpu = Number(values.cpu);
assert.ok(Number.isFinite(cpu) && cpu >= 1);
const output = resolve(values.output);
mkdirSync(output, { recursive: false });
const network = values.profile === "lan"
  ? { latency: 20, downloadThroughput: 12500000, uploadThroughput: 12500000 }
  : { latency: 800, downloadThroughput: 125000, uploadThroughput: 125000 };
const browser = await chromium.launch({ executablePath: values.browser });
const report = { browser: browser.version(), network, cpu, plannedSamples: 3,
  harnessSha256: createHash("sha256").update(readFileSync(import.meta.filename)).digest("hex"), runs: [] };
const save = () => writeFileSync(resolve(output, "report.json"), JSON.stringify(report, null, 2));
const entry = new URL("/?session=akashic%3Astartup-benchmark", values.url).href;

/** 看到固定尾页、保留输入草稿且完成绘制才算就绪，版本错配与模块失败直接报错。 */
async function ready(page) {
  const tail = page.locator('[data-message-id="startup-099"] .message-row');
  await tail.waitFor({ timeout: 45000 });
  assert.ok((await tail.textContent()).includes("启动基准消息 099"));
  await page.evaluate(() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done))));
  const box = await tail.boundingBox();
  assert.ok(box && box.y >= 0 && box.y + box.height <= 900, "尾页正文未完整进入视口");
  const editor = page.locator('.composer-wrap textarea');
  await editor.fill("缓存基准草稿");
  assert.equal(await editor.inputValue(), "缓存基准草稿");
  await editor.fill("");
  await page.evaluate(() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done))));
  assert.equal(await page.getByText("聊天页面已更新", { exact: false }).count(), 0);
  assert.equal(await page.locator(".web-host-entry-error").count(), 0);
}

try {
  const response = await fetch(new URL("/api/chat/web-ui/bootstrap", values.url));
  assert.ok(response.ok, `目录读取失败: ${response.status}`);
  report.bootstrapSha256 = createHash("sha256").update(Buffer.from(await response.arrayBuffer())).digest("hex");
  report.lockSha256 = createHash("sha256").update(readFileSync(new URL("../../package-lock.json", import.meta.url))).digest("hex");
  // 1. 固定已完成引导的客户端；初访不节流，让重开确实发生在 20 秒预热之前。
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, serviceWorkers: "block" });
  await context.addInitScript(() => localStorage.setItem("akashic.onboarding.seen", "1"));
  const page = await context.newPage();
  const cdp = await context.newCDPSession(page);
  await cdp.send("Network.enable");
  await cdp.send("Emulation.setCPUThrottlingRate", { rate: cpu });
  const initialNavigation = performance.now();
  await page.goto(entry);
  await ready(page);
  for (const mode of ["early-reload", "cached", "evicted"]) {
    // 2. 命中场景先真实下载 URL 模块；逐出场景清 HTTP 缓存，保留本机摘要提示。
    if (mode === "cached") {
      await cdp.send("Network.emulateNetworkConditions", { offline: false, latency: 0, downloadThroughput: -1, uploadThroughput: -1 });
      await page.evaluate(async () => {
        const response = await fetch("/api/chat/web-ui/bootstrap");
        if (!response.ok) throw new Error(`目录读取失败: ${response.status}`);
        const bootstrap = await response.json();
        await Promise.all(bootstrap.modules.map(async module => {
          const response = await fetch(`/api/chat/web-ui/module?plugin_id=${encodeURIComponent(module.pluginId)}&sha256=${module.moduleSha256}`);
          if (!response.ok) throw new Error(`模块预热失败: ${response.status}`);
          await response.arrayBuffer();
        }));
        await document.fonts.ready;
      });
    }
    if (mode === "evicted") await cdp.send("Network.clearBrowserCache");
    await cdp.send("Network.emulateNetworkConditions", { offline: false, ...network });
    const row = { mode, success: false, readyMs: null, errors: [] };
    report.runs.push(row);
    const error = reason => row.errors.push(String(reason));
    const consoleError = message => { if (message.type() === "error" && message.text().includes("[web-host]")) error(message.text()); };
    page.on("pageerror", error);
    page.on("console", consoleError);
    const started = performance.now();
    try {
      if (mode === "early-reload") {
        row.initialPageMs = started - initialNavigation;
        assert.ok(row.initialPageMs < 20000, "初访已超过模块预热等待，不能标为提前重开");
      }
      await page.goto(entry, { waitUntil: "commit", timeout: 45000 });
      await ready(page);
      row.readyMs = performance.now() - started;
      row.resources = await page.evaluate(() => performance.getEntriesByType("resource")
        .filter(entry => entry.name.includes("/web-ui/module"))
        .map(entry => ({ url: entry.name, transfer: entry.transferSize, decoded: entry.decodedBodySize, duration: entry.duration })));
      row.moduleNetworkBytes = row.resources.reduce((sum, resource) => sum + resource.transfer, 0);
      await page.screenshot({ path: resolve(output, `${mode}.png`) });
      row.success = row.errors.length === 0;
    } catch (reason) { row.errors.push(String(reason)); }
    finally {
      page.off("pageerror", error);
      page.off("console", consoleError);
      save();
      console.log(JSON.stringify({ mode, success: row.success, readyMs: row.readyMs, moduleNetworkBytes: row.moduleNetworkBytes, errors: row.errors }));
    }
  }
  await context.close();
} catch (reason) {
  report.setupError = String(reason);
} finally {
  // 3. 初始化和取证失败也留在报告里，不能用缺失的样本当作通过。
  report.complete = report.runs.length === report.plannedSamples && !report.setupError;
  save();
  await browser.close();
}
if (!report.complete || report.runs.some(row => !row.success)) process.exitCode = 1;
