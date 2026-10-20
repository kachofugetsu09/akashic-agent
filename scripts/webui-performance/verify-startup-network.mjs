/** 用真实页面验证预取不会提前执行 SDK，以及目录失败后的原页重试。 */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { chromium } from "playwright-core";

const { values } = parseArgs({ options: {
  url: { type: "string" }, browser: { type: "string" }, output: { type: "string" },
} });
if (!values.url || !values.browser || !values.output) throw new Error("需要 --url、--browser 和 --output");
mkdirSync(values.output);
// Check the served build: early module parsing must see the import map first.
const html = await (await fetch(values.url)).text();
const importMap = html.search(/<script\b[^>]*type=["']importmap["']/i);
const moduleLoad = html.search(/<(?:script\b[^>]*type=["']module["']|link\b[^>]*rel=["']modulepreload["'])/i);
assert.ok(importMap >= 0 && moduleLoad > importMap, "构建产物必须先声明 import map，再预加载或执行模块");
const browser = await chromium.launch({ executablePath: values.browser, headless: true });
const results = [];
try {
  for (const name of ["runtime-delay", "bootstrap-retry"]) {
    const context = await browser.newContext({ viewport: { width: 412, height: 915 } });
    await context.addInitScript(() => localStorage.setItem("akashic.onboarding.seen", "1"));
    const page = await context.newPage();
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    let release;
    const gate = new Promise(resolveGate => { release = resolveGate; });
    try {
      // 1. 使用明确的网络 gate，不靠 sleep 猜测运行时是否已经完成。
      if (name === "runtime-delay") {
        await page.route("**/dashboard/assets/runtime-*.js", async route => {
          await gate;
          await route.continue();
        });
        const sdk = page.waitForResponse(response => response.url().includes("/dashboard/assets/sdk/") && response.ok());
        await page.goto(new URL("/?session=akashic%3Astartup-benchmark", values.url).href, { waitUntil: "commit" });
        await sdk;
        assert.deepEqual(errors, [], "SDK 预取不能在共享 runtime 就绪前执行");
        release();
      } else {
        let attempts = 0;
        await page.route("**/api/chat/web-ui/bootstrap", async route => {
          attempts++;
          if (attempts === 1) await route.fulfill({ status: 503, body: "暂时不可用" });
          else await route.continue();
        });
        await page.goto(new URL("/?session=akashic%3Astartup-benchmark", values.url).href, { waitUntil: "commit" });
        await page.getByRole("button", { name: "重新加载", exact: true }).click();
        await page.locator(".conversation-page-chat .akashic-chat").waitFor();
        assert.equal(attempts, 2, "一次原页重试只读取一次新目录");
      }
      // 2. 确认仍只有一个聊天实例，真实历史和输入框可用。
      const chat = page.locator(".conversation-page-chat .akashic-chat");
      await chat.locator('[data-message-id="startup-099"] .message-row').waitFor();
      const editor = chat.locator('textarea[name="message"]');
      await editor.fill("网络恢复草稿");
      assert.equal(await editor.inputValue(), "网络恢复草稿");
      assert.equal(await page.locator(".akashic-chat").count(), 1);
      assert.deepEqual(errors, []);
      await page.screenshot({ path: resolve(values.output, `${name}.png`) });
      results.push({ name, success: true });
    } finally {
      release();
      await context.close();
    }
  }
} finally {
  await browser.close();
  writeFileSync(resolve(values.output, "report.json"), JSON.stringify(results, null, 2));
}
console.log(JSON.stringify(results));
