/** 真实浏览器与 TCP 断线验收；只对一次性基准 Runtime 操作。 */
import assert from "node:assert/strict";
import { once } from "node:events";
import { mkdirSync, writeFileSync } from "node:fs";
import net from "node:net";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { chromium } from "playwright-core";

const { values } = parseArgs({ options: {
  url: { type: "string" }, browser: { type: "string" }, output: { type: "string" },
} });
if (!values.url || !values.browser || !values.output) throw new Error("需要 url/browser/output");
const upstream = new URL(values.url);
if (upstream.hostname !== "127.0.0.1") throw new Error("只对本地隔离 Runtime 验收");
const output = resolve(values.output);
mkdirSync(output, { recursive: false });
const connections = new Set(), chatConnections = new Set();
// 1. 字节原样中继；主动销毁 TCP，而非伪造 WebSocket close 事件。
const proxy = net.createServer(client => {
  const server = net.connect(Number(upstream.port), upstream.hostname);
  connections.add(client);
  client.once("data", data => {
    if (data.toString().startsWith("GET /ws?")) chatConnections.add(client);
  });
  client.on("error", () => server.destroy());
  server.on("error", () => client.destroy());
  client.on("close", () => { connections.delete(client); chatConnections.delete(client); server.destroy(); });
  server.on("close", () => client.destroy());
  client.pipe(server).pipe(client);
});
proxy.listen(0, "127.0.0.1");
await once(proxy, "listening");
const origin = `http://127.0.0.1:${proxy.address().port}`;
const browser = await chromium.launch({ executablePath: values.browser, headless: true });
const context = await browser.newContext({ viewport: { width: 1365, height: 900 } });
await context.addInitScript(() => localStorage.setItem("akashic.onboarding.seen", "1"));
const pages = [], results = {}, errors = [];

async function waitFor(check, description) {
  const deadline = Date.now() + 15000;
  while (!check()) {
    if (Date.now() >= deadline) throw new Error(`等待失败：${description}`);
    await new Promise(resolveWait => setTimeout(resolveWait, 25));
  }
}

async function openPage() {
  /** 每个真实页面保留自己的帧与连接观测，不向应用注入新连接 owner。 */
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  const record = { page, sockets: 0, follows: 0, activity: 0, pins: 0 };
  page.on("pageerror", error => errors.push(error.message));
  page.on("request", request => {
    if (new URL(request.url()).pathname === "/api/chat/navigation/pins") record.pins += 1;
  });
  page.on("websocket", socket => {
    if (new URL(socket.url()).pathname !== "/ws") return;
    record.sockets += 1;
    socket.on("framereceived", ({ payload }) => {
      const frame = JSON.parse(String(payload));
      if (frame.type === "session.following") record.follows += 1;
      if (frame.type === "sessions.activity" && frame.snapshot) record.activity += 1;
      if (frame.type === "error") errors.push(frame.message);
    });
  });
  pages.push(record);
  await page.goto(`${origin}/?session=akashic%3Astartup-benchmark`);
  await page.locator(".conversation-page-chat .akashic-chat").locator('[data-message-id="startup-099"] .message-row').waitFor();
  await waitFor(() => record.follows > 0 && record.activity > 0, "消息与摘要基线");
  for (const frame of page.frames()) await frame.waitForFunction(() => document.fonts.status === "loaded");
  return record;
}

try {
  // 2. 五个页面独立连接；同一页面的界面初始化不会继续重建。
  const first = await openPage();
  for (let index = 1; index < 5; index++) await openPage();
  await waitFor(() => chatConnections.size === 5, "五个页面各一条聊天连接");
  results.startupSockets = pages.map(row => row.sockets);
  assert.ok(results.startupSockets.every(count => count <= 2), JSON.stringify(results.startupSockets));
  const frame = first.page.locator(".conversation-page-chat .akashic-chat");
  const editor = frame.locator('textarea[name="message"]');
  await editor.fill("断线前保留的草稿");

  // 3. 同时切断真实 TCP；当前会话、摘要和草稿必须恢复。
  const before = pages.map(row => ({ sockets: row.sockets, follows: row.follows, activity: row.activity }));
  const started = performance.now();
  for (const client of chatConnections) client.destroy();
  await waitFor(() => pages.every((row, index) => row.follows > before[index].follows
    && row.activity > before[index].activity), "断网后重新跟随");
  results.reconnectMs = performance.now() - started;
  assert.equal(await editor.inputValue(), "断线前保留的草稿");
  assert.ok(pages.every((row, index) => row.sockets === before[index].sockets + 1));
  await waitFor(() => chatConnections.size === 5, "恢复后没有残留连接");

  // 4. 真实目录提交触发通知；长连接必须调用 chatReady 后的新回调。
  const pinReads = first.pins;
  const endpoint = `${origin}/api/chat/sessions/akashic%3Astartup-benchmark/rename`;
  const response = await fetch(endpoint, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "连接生命周期验收" }) });
  assert.equal(response.status, 200);
  await waitFor(() => first.pins > pinReads, "元数据通知刷新置顶列表");
  await frame.getByRole("button", { name: "新会话", exact: true }).click();
  await frame.locator('[data-message-id="startup-099"]').waitFor({ state: "detached" });
  await frame.locator("button.conversation-session").filter({ hasText: "连接生命周期验收" }).click();
  await frame.locator('[data-message-id="startup-099"] .message-row').waitFor();
  await waitFor(() => chatConnections.size === 5, "切回已有会话" );
  results.sessionSwitch = true;
  assert.deepEqual(errors, []);
  await first.page.screenshot({ path: resolve(output, "restored.png") });
  const restored = await fetch(endpoint, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "" }) });
  assert.equal(restored.status, 200);
  for (const { page } of pages) await page.close();
  await waitFor(() => chatConnections.size === 0, "页面关闭释放上游连接");
  results.closedSockets = true;
  writeFileSync(resolve(output, "report.json"), JSON.stringify(results, null, 2));
  console.log(JSON.stringify(results));
} finally {
  await browser.close();
  for (const socket of connections) socket.destroy();
  await new Promise(resolveClose => proxy.close(resolveClose));
}
