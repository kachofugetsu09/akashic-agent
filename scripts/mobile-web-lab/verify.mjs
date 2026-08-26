import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

import { chromium } from "playwright-core";

import { startMobileWebLabServer } from "./server.mjs";

const lab = await startMobileWebLabServer({
  root: process.env.AKASHIC_MOBILE_WEB_LAB_ROOT,
});
let browser;
try {
  browser = await chromium.launch({ executablePath: chromiumExecutable(), headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  await context.addInitScript({ path: resolve("node_modules", "axe-core", "axe.min.js") });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(lab.origin, { waitUntil: "networkidle" });
  await page.waitForFunction(() => document.body.dataset.labReady === "true");

  const mobile = page.frameLocator("#lab-frame");
  await mobile.getByText("我们以后改手机聊天界面的样式", { exact: false }).waitFor();
  await page.getByRole("button", { name: "流式生成" }).click();
  await mobile.getByText("浏览器现在运行的就是手机里那一份 React 界面", { exact: false }).waitFor({ timeout: 12_000 });
  await mobile.getByPlaceholder("输入消息").fill("从 Browser Bridge 发送一条消息");
  await mobile.getByRole("button", { name: "发送消息" }).click();
  await page.locator("#lab-activity strong", { hasText: "sendMessage" }).waitFor();
  await mobile.getByText("这条回复由 Browser Bridge 接住发送动作后生成", { exact: false }).waitFor({ timeout: 12_000 });

  await mobile.getByRole("button", { name: "添加附件" }).click();
  await page.getByText("chooseAttachments 需要 Android 原生环境", { exact: true }).waitFor();
  const accessibility = {
    lab: await scanAccessibility(page),
    mobile: await scanAccessibility(page.frames()[1]),
  };
  assert.deepEqual(accessibility, { lab: [], mobile: [] });

  const focusPage = await context.newPage();
  await focusPage.setViewportSize({ width: 320, height: 800 });
  await focusPage.goto(`${lab.origin}?focus=1`, { waitUntil: "networkidle" });
  await focusPage.waitForFunction(() => document.body.dataset.labReady === "true");
  const focusMobileFrame = focusPage.frames()[1];
  await focusMobileFrame.waitForFunction(() => document.querySelectorAll(".mobile-message-anchor").length === 4);
  assert.deepEqual(await inspectNarrowLayout(focusMobileFrame), {
    viewportWidth: 320,
    documentWidth: 320,
    roles: ["user", "assistant", "user", "assistant"],
    decorativeRoleLabels: 0,
  });
  assert.deepEqual(await scanAccessibility(focusPage), []);
  assert.deepEqual(await scanAccessibility(focusMobileFrame), []);
  assert.deepEqual(errors, []);
  console.log("Mobile Web Lab browser verification passed");
} finally {
  await browser?.close();
  await lab.close();
}

async function scanAccessibility(frame) {
  return frame.evaluate(async () => {
    const result = await window.axe.run(document, {
      runOnly: { type: "tag", values: ["wcag2a", "wcag2aa"] },
    });
    return result.violations.map((violation) => ({
      id: violation.id,
      impact: violation.impact,
      nodes: violation.nodes.length,
    }));
  });
}

async function inspectNarrowLayout(frame) {
  return frame.evaluate(() => ({
    viewportWidth: window.innerWidth,
    documentWidth: document.documentElement.scrollWidth,
    roles: [...document.querySelectorAll(".mobile-message-anchor")]
      .map((node) => node.classList.contains("user") ? "user" : "assistant"),
    decorativeRoleLabels: document.querySelectorAll(".mobile-manuscript-kicker").length,
  }));
}

function chromiumExecutable() {
  if (process.env.AKASHIC_PERF_CHROMIUM) return process.env.AKASHIC_PERF_CHROMIUM;
  const candidate = ["/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/google-chrome"].find(existsSync);
  if (!candidate) throw new Error("未找到 Chromium；请设置 AKASHIC_PERF_CHROMIUM");
  return candidate;
}
