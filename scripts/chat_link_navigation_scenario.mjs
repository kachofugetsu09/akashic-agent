/** 用聊天夹具页面（与外壳里同一挂载入口）验证正文链接、流式链接与会话导航；仅访问本地夹具。 */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { chromium } from "playwright-core";
import { startDesktopFixtureServer } from "./webui-performance/desktop-fixture-server.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { values } = parseArgs({ options: {
  browser: { type: "string", default: process.env.CHROME_BIN ?? "/opt/google/chrome/chrome" },
  output: { type: "string", default: "/tmp/akashic-chat-link-navigation" },
} });
const output = resolve(values.output);
await mkdir(output, { recursive: true, mode: 0o700 });
const bundle = resolve(output, "bundle");
const built = spawnSync(process.execPath, [resolve(root, "node_modules/vite/bin/vite.js"),
  "build", "--config", "frontend/chat/vite.harness.config.ts", "--outDir", bundle, "--emptyOutDir"],
{ cwd: root, encoding: "utf8" });
assert.equal(built.status, 0, built.stderr);
const fixture = await startDesktopFixtureServer(bundle, { historyCount: 2 });
const browser = await chromium.launch({ executablePath: values.browser, headless: true });
const checks = [];
const errors = [];
const destination = `${fixture.origin}/blocked-destination`;
const markdown = `**PR 地址：** [查看 PR](${destination})\n\n<${destination}?autolink=1>\n\n[邮件](mailto:reader@example.com) · [页内](#reading)`;
try {
  // 1. 聊天与宿主同一文档，HTTP 历史和 WebSocket 来自一次性夹具；目标网页禁止被嵌入，也不影响新页打开。
  const context = await browser.newContext({ viewport: { width: 1365, height: 914 } });
  const page = await context.newPage();
  page.on("pageerror", (error) => errors.push(error.message));
  await context.route(`${fixture.origin}/blocked-destination*`, (route) => route.fulfill({
    status: 200, contentType: "text/html; charset=utf-8", headers: { "X-Frame-Options": "DENY" },
    body: "<h1>目标网页</h1>",
  }));
  await context.route(`${fixture.origin}/api/chat/sessions/*/messages*`, async (route) => {
    const response = await route.fetch();
    const history = await response.json();
    for (const item of history.items) {
      if (item.author === "assistant") item.body.parts = [{ kind: "text", value: markdown }];
    }
    await route.fulfill({ response, json: history });
  });
  await page.goto(`${fixture.origin}/?session=perf-session&keep=1`);
  const chat = page.locator(".akashic-chat");
  await chat.getByRole("link", { name: "查看 PR", exact: true }).waitFor();

  // 2. 历史里的外部链接在新页打开，opener 为空，宿主页面不离开。
  await chat.getByRole("button", { name: "纯文本性能会话", exact: true }).click();
  await chat.getByRole("button", { name: "性能基线会话", exact: true }).click();
  const link = chat.getByRole("link", { name: "查看 PR", exact: true });
  await link.waitFor();
  await page.screenshot({ path: resolve(output, "before-link.png") });
  const originalPageUrl = page.url();
  const popupPromise = context.waitForEvent("page");
  await link.click();
  const popup = await popupPromise;
  await popup.getByRole("heading", { name: "目标网页" }).waitFor();
  assert.equal(await popup.evaluate(() => window.opener), null);
  assert.equal(popup.url(), destination);
  await popup.close();
  assert.equal(page.url(), originalPageUrl);
  assert.equal(await chat.locator(`a[href="${destination}?autolink=1"]`).getAttribute("target"), "_blank");
  assert.equal(await chat.locator('a[href^="mailto:"]').getAttribute("target"), null);
  assert.equal(await chat.locator('a[href="#reading"]').getAttribute("target"), null);
  checks.push("历史显式链接和自动链接在新页打开；目标 DENY 不影响；opener 为空；宿主页面 URL 不变");

  // 3. 真实 reply.status 经 WebSocket 进入流式渲染器，与历史链接行为一致。
  const streaming = await fetch(`${fixture.origin}/__fixture/stream?count=1&terminal=0&delta=${encodeURIComponent(`[流式网页](${destination})`)}`, { method: "POST" });
  assert.equal(streaming.status, 200);
  const streamLink = chat.locator(".reply-activity a", { hasText: "流式网页" });
  await streamLink.waitFor();
  const streamedPopupPromise = context.waitForEvent("page");
  await streamLink.click();
  const streamedPopup = await streamedPopupPromise;
  await streamedPopup.getByRole("heading", { name: "目标网页" }).waitFor();
  await streamedPopup.close();
  assert.equal(page.url(), originalPageUrl);
  checks.push("流式外部链接继续在新页打开，对话保持可用");

  // 4. 新会话清理 session 路由参数，保留宿主页面的其他参数；刷新后仍是新会话。
  await chat.getByRole("button", { name: "新会话", exact: true }).click();
  const fresh = new URL(page.url());
  assert.equal(fresh.searchParams.has("session"), false);
  assert.equal(fresh.searchParams.get("keep"), "1");
  await page.reload();
  await chat.getByRole("button", { name: "新会话", exact: true }).waitFor();
  await page.screenshot({ path: resolve(output, "new-chat.png") });
  checks.push("新会话清理 session 参数并保留其他页面选项，刷新后保持");
  assert.deepEqual(errors, [], "对话不得产生未处理的页面错误");
  await writeFile(resolve(output, "report.json"), JSON.stringify({ checks, errors }, null, 2));
  console.log(JSON.stringify({ checks, errors, output }, null, 2));
} finally {
  await browser.close();
  await fixture.close();
}
