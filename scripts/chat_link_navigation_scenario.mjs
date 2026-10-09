/** 用真实对话 bundle 验证嵌入页的链接、返回和会话导航；仅访问本地夹具。 */
import assert from "node:assert/strict";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { chromium } from "playwright-core";
import { startDesktopFixtureServer } from "./webui-performance/desktop-fixture-server.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { values } = parseArgs({ options: {
  browser: { type: "string", default: process.env.CHROME_BIN ?? "/opt/google/chrome/chrome" },
  bundle: { type: "string", default: resolve(root, "static/chat") },
  output: { type: "string", default: "/tmp/akashic-chat-link-navigation" },
  baseline: { type: "boolean", default: false },
} });
const output = resolve(values.output);
await mkdir(output, { recursive: true, mode: 0o700 });
const fixture = await startDesktopFixtureServer(resolve(values.bundle), { historyCount: 2 });
const browser = await chromium.launch({ executablePath: values.browser, headless: true });
const checks = [];
const errors = [];
const destination = `${fixture.origin}/blocked-destination`;
const markdown = `**PR 地址：** [查看 PR](${destination})\n\n<${destination}?autolink=1>\n\n[邮件](mailto:reader@example.com) · [页内](#reading)`;
try {
  // 1. 生产对话 bundle 在真实 iframe 内运行，HTTP 历史和 WebSocket 来自一次性夹具。
  const context = await browser.newContext({ viewport: { width: 1365, height: 914 } });
  const page = await context.newPage();
  page.on("pageerror", (error) => errors.push(error.message));
  await context.route(`${fixture.origin}/blocked-destination*`, (route) => route.fulfill({
    status: 200, contentType: "text/html; charset=utf-8", headers: { "X-Frame-Options": "DENY" },
    body: "<h1>目标网页</h1>",
  }));
  await context.route(`${fixture.origin}/chat*`, async (route) => route.fulfill({
    contentType: "text/html", body: await readFile(resolve(values.bundle, "index.html"), "utf8"),
  }));
  await context.route(`${fixture.origin}/link-shell`, (route) => route.fulfill({
    contentType: "text/html", body: `<!doctype html><meta charset="utf-8"><title>嵌入对话夹具</title>
      <style>body{margin:0;background:#f7f5ef}header{padding:16px}iframe{width:100%;height:850px;border:0}</style>
      <header>Akashic · 对话（测试宿主）</header>
      <iframe title="Akashic 对话" src="/chat?embedded=1&session=perf-session&keep=1"></iframe>`,
  }));
  await context.route(`${fixture.origin}/api/chat/sessions/*/messages*`, async (route) => {
    const response = await route.fetch();
    const history = await response.json();
    for (const item of history.items) {
      if (item.author === "assistant") item.body.parts = [{ kind: "text", value: markdown }];
    }
    await route.fulfill({ response, json: history });
  });
  await page.goto(`${fixture.origin}/link-shell`);
  const frame = await page.locator("iframe").elementHandle().then((element) => element.contentFrame());
  assert(frame, "必须加载实际对话 iframe");
  await frame.getByRole("link", { name: "查看 PR", exact: true }).waitFor();
  assert.equal(await frame.locator(".product-band").count(), 0);

  // 2. 切换会话之后返回或刷新，嵌入页仍不拥有产品顶栏。
  await frame.getByRole("button", { name: "纯文本性能会话", exact: true }).click();
  await frame.getByRole("button", { name: "性能基线会话", exact: true }).click();
  await frame.getByRole("link", { name: "查看 PR", exact: true }).waitFor();
  const afterSwitch = new URL(frame.url());
  assert.equal(afterSwitch.searchParams.get("embedded"), values.baseline ? null : "1");
  if (!values.baseline) {
    assert.equal(afterSwitch.searchParams.get("keep"), "1");
    assert.equal(afterSwitch.searchParams.has("session"), false);
  }
  await page.screenshot({ path: resolve(output, "before-link.png") });
  const link = frame.getByRole("link", { name: "查看 PR", exact: true });
  if (values.baseline) {
    const denied = page.waitForEvent("console", { predicate: (message) => /x-frame-options/iu.test(message.text()) });
    await link.click();
    await denied;
    await page.screenshot({ path: resolve(output, "blocked.png") });
    await page.evaluate(() => window.history.back());
    await frame.locator(".product-band").waitFor();
    await page.screenshot({ path: resolve(output, "double-header.png") });
    checks.push("基线重现：会话切换丢失 embedded；正文链接在 iframe 内被 DENY 拦截；浏览器返回显示内层顶栏");
  } else {
    const originalFrameUrl = frame.url();
    const originalPageUrl = page.url();
    const popupPromise = context.waitForEvent("page");
    await link.click();
    const popup = await popupPromise;
    await popup.waitForLoadState("domcontentloaded");
    await popup.screenshot({ path: resolve(output, "popup.png") });
    await popup.getByRole("heading", { name: "目标网页" }).waitFor();
    assert.equal(await popup.evaluate(() => window.opener), null);
    assert.equal(popup.url(), destination);
    await popup.close();
    assert.equal(frame.url(), originalFrameUrl);
    assert.equal(page.url(), originalPageUrl);
    assert.equal(await frame.locator(".product-band").count(), 0);
    assert.equal(await frame.locator(`a[href="${destination}?autolink=1"]`).getAttribute("target"), "_blank");
    assert.equal(await frame.locator('a[href^="mailto:"]').getAttribute("target"), null);
    assert.equal(await frame.locator('a[href="#reading"]').getAttribute("target"), null);
    checks.push("历史显式链接和自动链接在新页打开；目标 DENY 不拦截新页；opener 为空；对话 URL 和顶栏不变");

    // 3. 即使另一路径导航离开 iframe，浏览器返回仍按嵌入模式重载。
    await assert.rejects(frame.goto(destination), /ERR_BLOCKED_BY_RESPONSE/u);
    await page.evaluate(() => window.history.back());
    await frame.getByRole("link", { name: "查看 PR", exact: true }).waitFor();
    assert.equal(await frame.locator(".product-band").count(), 0);
    await frame.goto(frame.url());
    await frame.getByRole("link", { name: "查看 PR", exact: true }).waitFor();
    assert.equal(await frame.locator(".product-band").count(), 0);
    checks.push("会话切换后 iframe 离开、浏览器返回及刷新均保留嵌入模式，不增加顶栏");

    // 4. 真实 reply.status 经 WebSocket 进入流式渲染器，与历史链接行为一致。
    const streaming = await fetch(`${fixture.origin}/__fixture/stream?count=1&terminal=0&delta=${encodeURIComponent(`[流式网页](${destination})`)}`, { method: "POST" });
    assert.equal(streaming.status, 200);
    const streamLink = frame.locator(".reply-activity a", { hasText: "流式网页" });
    await streamLink.waitFor();
    const streamedPopupPromise = context.waitForEvent("page");
    await streamLink.click();
    const streamedPopup = await streamedPopupPromise;
    await streamedPopup.getByRole("heading", { name: "目标网页" }).waitFor();
    await streamedPopup.close();
    assert.equal(await frame.locator(".product-band").count(), 0);
    checks.push("流式外部链接继续在新页打开，对话保持可用");

    await frame.getByRole("button", { name: "新会话", exact: true }).click();
    assert.equal(new URL(frame.url()).searchParams.get("embedded"), "1");
    assert.equal(new URL(frame.url()).searchParams.get("keep"), "1");
    await frame.goto(frame.url());
    await frame.getByRole("button", { name: "新会话", exact: true }).waitFor();
    assert.equal(await frame.locator(".product-band").count(), 0);
    await page.screenshot({ path: resolve(output, "new-chat.png") });
    checks.push("新建会话及其刷新保留嵌入模式和其他页面选项");

    const standalone = await context.newPage();
    await standalone.goto(`${fixture.origin}/chat?session=perf-session&surface=runtime&keep=1`);
    await standalone.getByRole("button", { name: "新会话", exact: true }).click();
    const standaloneUrl = new URL(standalone.url());
    assert.equal(standaloneUrl.searchParams.has("session"), false);
    assert.equal(standaloneUrl.searchParams.has("surface"), false);
    assert.equal(standaloneUrl.searchParams.get("keep"), "1");
    assert.equal(await standalone.locator(".product-band").count(), 1);
    checks.push("独立对话仍显示一个顶栏；新会话清理 session/surface 路由参数");
    assert.deepEqual(errors, [], "对话不得产生未处理的页面错误");
  }
  await writeFile(resolve(output, "report.json"), JSON.stringify({ checks, errors }, null, 2));
  console.log(JSON.stringify({ checks, errors, output }, null, 2));
} finally {
  await browser.close();
  await fixture.close();
}
