/** 在完整开发页面上重读真实目录，并通过 SIGHUP 验证真实插件换代。 */
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { chromium } from "playwright-core";
import { createServer } from "vite";

const { values } = parseArgs({ options: {
  url: { type: "string" }, browser: { type: "string" }, state: { type: "string" },
  pid: { type: "string" }, output: { type: "string" },
} });
if (Object.values(values).length !== 5) throw new Error("需要 url/browser/state/pid/output");
if (new URL(values.url).hostname !== "127.0.0.1") throw new Error("只对本地隔离 Runtime 验收");
const state = resolve(values.state), output = resolve(values.output), pid = Number(values.pid);
const command = readFileSync(`/proc/${pid}/cmdline`, "utf8").split("\0");
assert.ok(command.includes(state) && command.some(item => item.endsWith("serve-startup-runtime.py")));
mkdirSync(output, { recursive: false });
const asset = resolve(state, "plugins/projects/plugin_ui.js");
const saved = readFileSync(asset);
writeFileSync(resolve(output, "project_ui.before-reload.js"), saved, { flag: "wx" });
process.env.AKASHIC_API_ORIGIN = values.url;
const server = await createServer({ configFile: resolve("frontend/chat/vite.harness.config.ts"),
  server: { host: "127.0.0.1", port: 0, hmr: false }, logLevel: "error" });
await server.listen();
const browser = await chromium.launch({ executablePath: values.browser, headless: true });
let changed = false;
try {
  // 1. 使用开发入口才能调用公开的目录读取函数；应用、插件与所有 API 均为真实实现。
  const page = await browser.newPage({ viewport: { width: 1365, height: 900 } });
  page.setDefaultTimeout(30000);
  let sockets = 0, follows = 0, statuses = 0;
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("websocket", socket => {
    if (new URL(socket.url()).pathname !== "/ws") return;
    sockets += 1;
    socket.on("framereceived", ({ payload }) => {
      const frame = JSON.parse(String(payload));
      if (frame.type === "session.following") follows += 1;
      if (frame.type === "reply.status") statuses += 1;
    });
  });
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/assets/?session=akashic%3Astartup-benchmark`);
  await page.locator('[data-message-id="startup-099"] .message-row').waitFor();
  const refresh = () => page.evaluate(async () => {
    const ui = await import("/assets/src/plugin-ui-runtime.tsx");
    await ui.loadWebPluginCatalog();
    await new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done)));
  });
  await refresh();
  const initial = sockets;
  for (let index = 0; index < 3; index++) await refresh();
  assert.equal(sockets, initial, "相同目录不应重建聊天连接");
  const before = await fetch(new URL("/api/chat/plugin-ui/catalog", values.url)).then(r => r.json());

  assert.ok(before.items.some(item => item.id === "projects"), "需要已激活的项目界面插件");

  // 2. 修改的只是隔离源码副本；正式扫描和 apply 发布全新的目录修订。
  writeFileSync(asset, Buffer.concat([saved, Buffer.from("\n/* connection generation scenario */\n")]));
  changed = true;
  process.kill(pid, "SIGHUP");
  let after;
  const transitionStatuses = [];
  const deadline = Date.now() + 60000;
  while (Date.now() < deadline) {
    const response = await fetch(new URL("/api/chat/plugin-ui/catalog", values.url));
    if (response.ok) {
      after = await response.json();
      if (after.catalog_revision !== before.catalog_revision) break;
    } else {
      // 目录暂停时可以明确不可用；其它 HTTP 错误直接使验收失败。
      transitionStatuses.push(response.status);
      assert.equal(response.status, 503);
    }
    await new Promise(done => setTimeout(done, 100));
  }
  assert.notEqual(after?.catalog_revision, before.catalog_revision);
  assert.ok(after?.catalog_revision);
  const previousFollows = follows, previousStatuses = statuses;
  await refresh();
  while (follows === previousFollows || statuses === previousStatuses) {
    if (Date.now() >= deadline) throw new Error("换代后未恢复消息和回复状态订阅");
    await new Promise(done => setTimeout(done, 25));
  }
  assert.equal(sockets, initial + 1, "真实换代只重建一次");
  await refresh();
  assert.equal(sockets, initial + 1);
  await page.locator('[data-message-id="startup-099"] .message-row').waitFor();
  assert.deepEqual(errors, []);
  const report = { sameCatalogReloads: 3, sameCatalogReconnects: 0, realGenerationReconnects: 1,
    before: before.catalog_revision, after: after.catalog_revision, historyVisible: true, transitionStatuses };
  writeFileSync(resolve(output, "report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
} finally {
  await browser.close();
  await server.close();
  if (changed) {
    writeFileSync(asset, saved);
    process.kill(pid, "SIGHUP");
  }
}
