#!/usr/bin/env node
/**
 * Akashic / 通用 WebUI 光学对齐检测工具（Optical Alignment Blur-Volume Inspector）
 *
 * 通过“前景提取 + 高斯模糊体积渲染 + 像素级质心计算”，将界面中图标按钮与图文控件
 * 转化为高对比模糊体积图，帮助 Agent 与开发者直观/定量发现光学偏移问题。
 *
 * 用法：
 *   node scripts/check_optical_alignment.mjs --url http://127.0.0.1:44150 [选项]
 *
 * 选项：
 *   --url <地址>            目标页面地址（必填）
 *   --auth <secret文件>     携带 Bearer Token 文件（可选）
 *   --storage <json文件>    Playwright storageState（可选）
 *   --out <输出目录>        结果落盘目录（默认 /tmp/akashic-optical-alignment）
 *   --browser <路径>        Chromium 可执行路径（默认 CHROMIUM_PATH / /usr/bin/chromium）
 *   --akashic-views         遍历 Akashic 典型视图（聊天页、侧栏、设置弹框、模型页、窄屏）
 *   --dark                  同时测试暗色主题
 *   --threshold <px>        光学偏移告警阈值（默认 1.5 CSS px）
 *   --strict                存在超过阈值项时以非零退出码退出
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { chmodSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { extname, resolve } from "node:path";
import { parseArgs } from "node:util";
import { chromium } from "playwright-core";
import {
  analyzeAndRenderInCanvasPage,
  analyzeNestedInCanvasPage,
  collectNestedLayoutInFrame,
  collectOpticalTargetsInFrame,
  disableInkIsolationInFrame,
  enableInkIsolationInFrame,
} from "./optical-alignment/engine.mjs";

const defaultBrowser = () => {
  if (process.env.CHROMIUM_PATH && existsSync(process.env.CHROMIUM_PATH)) return process.env.CHROMIUM_PATH;
  if (process.env.CHROME_BIN && existsSync(process.env.CHROME_BIN)) return process.env.CHROME_BIN;
  const fallback = `${process.env.HOME || ""}/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome`;
  if (existsSync(fallback)) return fallback;
  return "/usr/bin/chromium";
};

const { values } = parseArgs({
  options: {
    url: { type: "string" },
    auth: { type: "string" },
    storage: { type: "string" },
    candidate: { type: "string" },
    out: { type: "string", default: "/tmp/akashic-optical-alignment" },
    browser: { type: "string", default: defaultBrowser() },
    "akashic-views": { type: "boolean", default: false },
    dark: { type: "boolean", default: false },
    threshold: { type: "string", default: "1.5" },
    strict: { type: "boolean", default: false },
    dpr: { type: "string", default: "2" },
  },
});

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".json": "application/json; charset=utf-8",
  ".woff2": "font/woff2",
  ".woff": "font/woff",
  ".ttf": "font/ttf",
  ".png": "image/png",
};

async function attachCandidateRoutes(context, candidateRoot) {
  const root = resolve(candidateRoot);
  const { readdirSync } = await import("node:fs");
  const dashFiles = existsSync(resolve(root, "static/dashboard")) ? readdirSync(resolve(root, "static/dashboard")) : [];
  const dashIndexCss = dashFiles.find((f) => /^index-.*\.css$/.test(f));

  await context.route("**/*", async (route) => {
    const req = route.request();
    const url = new URL(req.url());
    if (req.method() !== "GET" && req.method() !== "HEAD") return route.continue();
    if (url.pathname.startsWith("/assets/")) {
      const file = url.pathname.slice("/assets/".length);
      const p = resolve(root, "static/chat", file);
      if (!file.includes("..") && existsSync(p)) {
        return route.fulfill({ body: readFileSync(p), contentType: MIME[extname(file)] || "application/octet-stream" });
      }
    }
    if (url.pathname.startsWith("/dashboard/assets/") && url.pathname.endsWith(".css")) {
      const file = url.pathname.slice("/dashboard/assets/".length);
      const mapped = /^index-.*\.css$/.test(file) && dashIndexCss ? dashIndexCss : file;
      const p = resolve(root, "static/dashboard", mapped);
      if (!mapped.includes("..") && existsSync(p)) {
        return route.fulfill({ body: readFileSync(p), contentType: MIME[".css"] });
      }
    }
    if (url.pathname === "/api/chat/web-ui/bootstrap") {
      const response = await route.fetch();
      const data = await response.json().catch(() => null);
      if (data && Array.isArray(data.modules)) {
        for (const m of data.modules) {
          const id = m.pluginId.split("@")[0].replaceAll("-", "_");
          const jsPath = resolve(root, "plugins", id, "web_module.js");
          const cssPath = resolve(root, "plugins", id, "web_module.css");
          if (existsSync(jsPath)) {
            m.module = readFileSync(jsPath, "utf8");
            m.moduleBytes = Buffer.byteLength(m.module);
            m.moduleSha256 = createHash("sha256").update(m.module).digest("hex");
          }
          if (existsSync(cssPath)) {
            m.stylesheet = readFileSync(cssPath, "utf8");
            m.stylesheetBytes = Buffer.byteLength(m.stylesheet);
            m.stylesheetSha256 = createHash("sha256").update(m.stylesheet).digest("hex");
          }
        }
        return route.fulfill({ response, json: data });
      }
    }
    return route.continue();
  });
}

assert(values.url, "请提供 --url");
const outDir = resolve(values.out);
mkdirSync(outDir, { recursive: true, mode: 0o700 });
chmodSync(outDir, 0o700);
const threshold = Number(values.threshold);
const dpr = Number(values.dpr);

export async function captureOpticalSnapshot(page, canvasPage, { label, outDir, dpr = 2 }) {
  const viewport = page.viewportSize();
  const targets = [];

  // 1. 收集主文档和所有可见同源 iframe 里的目标控件
  for (let fi = 0; fi < page.frames().length; fi++) {
    const frame = page.frames()[fi];
    let ox = 0;
    let oy = 0;
    if (frame !== page.mainFrame()) {
      const el = await frame.frameElement().catch(() => null);
      const box = await el?.boundingBox().catch(() => null);
      if (!el || !box || box.width < 10 || box.height < 10) continue;
      const vis = await el.isVisible().catch(() => false);
      if (!vis) continue;
      ox = box.x;
      oy = box.y;
    }
    const items = await frame.evaluate(collectOpticalTargetsInFrame, `f${fi}`).catch(() => []);
    for (const item of items) {
      const x = item.rect.x + ox;
      const y = item.rect.y + oy;
      if (x + item.rect.w <= 0 || y + item.rect.h <= 0 || x >= viewport.width || y >= viewport.height) continue;
      targets.push({
        ...item,
        rect: { ...item.rect, x, y },
        iconRect: { ...item.iconRect, x: item.iconRect.x + ox, y: item.iconRect.y + oy },
      });
    }
  }

  // 1.5 采集 L2/L3/L4 多层嵌套容器几何
  let nestedLayout = null;
  for (const frame of page.frames()) {
    const fbox = frame === page.mainFrame()
      ? { x: 0, y: 0 }
      : await (await frame.frameElement().catch(() => null))?.boundingBox().catch(() => null);
    if (!fbox) continue;
    const sub = await frame.evaluate(collectNestedLayoutInFrame).catch(() => null);
    if (!sub) continue;
    const shift = (r) => (r ? { ...r, x: +(r.x + fbox.x).toFixed(2), y: +(r.y + fbox.y).toFixed(2), r: +(r.r + fbox.x).toFixed(2), b: +(r.b + fbox.y).toFixed(2), cx: +(r.cx + fbox.x).toFixed(2), cy: +(r.cy + fbox.y).toFixed(2) } : null);
    if (!nestedLayout || sub.hasChatMain) {
      const shifted = { ...sub };
      for (const k of ["chatMain", "sidebar", "heading", "toolsBar", "firstMsgP", "userBubble", "scrollReturn", "composerWrap", "composer", "textarea", "modelPicker", "composerTool", "actionBtn", "composerReply", "statsLine", "codeHdr", "codeTitle", "codeCopy"]) {
        shifted[k] = shift(sub[k]);
      }
      if (!nestedLayout) {
        nestedLayout = shifted;
      } else {
        for (const k of Object.keys(shifted)) {
          if (shifted[k] != null && shifted[k] !== false) nestedLayout[k] = shifted[k];
        }
      }
    } else if (!nestedLayout.settingsDialog && sub.settingsDialog) {
      nestedLayout.settingsDialog = shift(sub.settingsDialog);
      nestedLayout.settingsNavH2 = shift(sub.settingsNavH2);
      nestedLayout.settingsHeadingH3 = shift(sub.settingsHeadingH3);
      nestedLayout.settingsClose = shift(sub.settingsClose);
    }
  }

  // 2. 原图截屏
  const origBuf = await page.screenshot({ animations: "disabled" });

  // 3. 开启前缀通道分离模式并截墨迹图
  let inkBuf;
  try {
    for (const frame of page.frames()) {
      await frame.evaluate(enableInkIsolationInFrame).catch(() => {});
    }
    await page.waitForTimeout(60);
    inkBuf = await page.screenshot({ animations: "disabled" });
  } finally {
    for (const frame of page.frames()) {
      await frame.evaluate(disableInkIsolationInFrame).catch(() => {});
    }
  }

  // 4. 在专用 Canvas Page 中计算光学质心与生成拼版/透视图
  const origB64 = origBuf.toString("base64");
  const inkB64 = inkBuf.toString("base64");
  const analyzed = await analyzeAndRenderInCanvasPage(canvasPage, {
    origPngBase64: origB64,
    inkPngBase64: inkB64,
    dpr,
    viewport,
    targets,
    label,
  });

  const safeLabel = label.replace(/[^a-zA-Z0-9_-]/g, "_");
  const overlayFile = `overlay-${safeLabel}.png`;
  const sheetFile = `sheet-${safeLabel}.png`;
  writeFileSync(resolve(outDir, overlayFile), Buffer.from(analyzed.overlayPngBase64, "base64"));
  writeFileSync(resolve(outDir, sheetFile), Buffer.from(analyzed.sheetPngBase64, "base64"));

  let nestedFile = null;
  let composerFile = null;
  let nestedChecks = [];
  if (nestedLayout) {
    const nested = await analyzeNestedInCanvasPage(canvasPage, {
      origPngBase64: origB64,
      inkPngBase64: inkB64,
      dpr,
      viewport,
      layout: nestedLayout,
      label,
    });
    nestedChecks = nested.checks;
    nestedFile = `nested-${safeLabel}.png`;
    writeFileSync(resolve(outDir, nestedFile), Buffer.from(nested.nestedPngBase64, "base64"));
    if (nested.composerPngBase64) {
      composerFile = `composer-${safeLabel}.png`;
      writeFileSync(resolve(outDir, composerFile), Buffer.from(nested.composerPngBase64, "base64"));
    }
  }

  return {
    label,
    viewport,
    count: analyzed.count,
    overlayFile,
    sheetFile,
    nestedFile,
    composerFile,
    nestedChecks,
    results: analyzed.results,
  };
}

async function dismissOnboarding(page) {
  const ob = page.locator("dialog.onboarding-surface[open]");
  if (await ob.count()) {
    await ob.getByRole("button", { name: /^(稍后再说|我已经配好了，直接聊天)$/ }).first().click().catch(() => {});
    await page.waitForTimeout(400);
  }
}

async function main() {
  const headers = values.auth ? { authorization: `Bearer ${readFileSync(values.auth, "utf8").trim()}` } : {};
  const browser = await chromium.launch({ executablePath: values.browser, headless: true });
  const canvasCtx = await browser.newContext({ viewport: { width: 1280, height: 800 }, deviceScaleFactor: 1 });
  const canvasPage = await canvasCtx.newPage();

  const snapshots = [];

  try {
    const scenarios = values["akashic-views"]
      ? [
          { name: "desktop-light", viewport: { width: 1440, height: 900 }, colorScheme: "light", mobile: false },
          ...(values.dark ? [{ name: "desktop-dark", viewport: { width: 1440, height: 900 }, colorScheme: "dark", mobile: false }] : []),
          { name: "mobile-light", viewport: { width: 390, height: 844 }, colorScheme: "light", mobile: true },
        ]
      : [{ name: "page", viewport: { width: 1440, height: 900 }, colorScheme: "light", mobile: false }];

    for (const sc of scenarios) {
      const context = await browser.newContext({
        viewport: sc.viewport,
        deviceScaleFactor: dpr,
        hasTouch: sc.mobile,
        isMobile: sc.mobile,
        colorScheme: sc.colorScheme,
        extraHTTPHeaders: headers,
        ...(values.storage ? { storageState: values.storage } : {}),
      });
      if (values.candidate) await attachCandidateRoutes(context, values.candidate);
      const page = await context.newPage();
      await page.goto(values.url, { waitUntil: "domcontentloaded" });
      await page.waitForTimeout(2000);
      await dismissOnboarding(page);

      if (sc.colorScheme === "dark") {
        // 如果 Akashic 支持主题切换，确保进入暗色主题
        await page.evaluate(() => {
          document.documentElement.dataset.theme = "dark";
        }).catch(() => {});
        await page.waitForTimeout(300);
      }

      // 主视图
      snapshots.push(await captureOpticalSnapshot(page, canvasPage, { label: `${sc.name}-main`, outDir, dpr }));

      if (values["akashic-views"]) {
        // 若有历史会话，点进第一条历史会话并展开过程区，让消息气泡、代码复制、工具过程、胶囊选择器都参与检测
        const firstSession = page.locator(".conversation-session-row button, .navigation-session__open, [data-session-id]").first();
        if (await firstSession.isVisible().catch(() => false)) {
          await firstSession.click().catch(() => {});
          await page.waitForTimeout(800);
        }
        const turnSummary = page.locator(".turn-process-summary[aria-expanded='false']").first();
        if (await turnSummary.isVisible().catch(() => false)) {
          await turnSummary.click().catch(() => {});
          await page.waitForTimeout(400);
        }
        // 向上滚一点让回到底部按钮 .desktop-scroll-return 出现
        await page.evaluate(() => {
          const sc = [...document.querySelectorAll("*")].find(
            (e) => e.scrollHeight > e.clientHeight + 40 && getComputedStyle(e).overflowY !== "visible" && e.clientWidth > 300,
          );
          if (sc) sc.scrollTop = Math.max(0, sc.scrollTop - 500);
        });
        await page.waitForTimeout(450);
        // 悬停非最后一条助手回复（避免与底栏输入卡重叠）与代码块
        const assistantMsgs = page.locator(".web-message-anchor.timeline-output, .web-message-anchor.assistant");
        const msgIdx = (await assistantMsgs.count()) >= 2 ? 0 : 0;
        if (await assistantMsgs.nth(msgIdx).isVisible().catch(() => false)) {
          await assistantMsgs.nth(msgIdx).hover().catch(() => {});
          await page.waitForTimeout(250);
        }
        const codeBlock = page.locator(".static-code-block, .reading-code-block").first();
        if (await codeBlock.isVisible().catch(() => false)) {
          await codeBlock.hover().catch(() => {});
          await page.waitForTimeout(200);
        }
        snapshots.push(await captureOpticalSnapshot(page, canvasPage, { label: `${sc.name}-conversation`, outDir, dpr }));

        // 如果输入区有文字（显示发送按钮而不是 disabled 空状态），再测一次有文字状态下的发送按钮/折叠
        const textarea = page.locator("textarea[aria-label='消息']").first();
        if (await textarea.isVisible().catch(() => false)) {
          await textarea.fill("光学对齐测试");
          await page.waitForTimeout(200);
          snapshots.push(await captureOpticalSnapshot(page, canvasPage, { label: `${sc.name}-composer-filled`, outDir, dpr }));
          await textarea.fill("");
        }

        // 桌面端打开工作台侧栏
        if (!sc.mobile) {
          const wbBtn = page.locator("button[data-header-action='workbench'], button[aria-label*='工作台']").first();
          if (await wbBtn.isVisible().catch(() => false)) {
            await wbBtn.click().catch(() => {});
            await page.waitForTimeout(500);
            snapshots.push(await captureOpticalSnapshot(page, canvasPage, { label: `${sc.name}-workbench`, outDir, dpr }));
          }
        }

        // 设置弹框（功能设置 + 模型设置 + 插件详情）
        await page.evaluate(() => { window.location.hash = "plugins"; });
        await page.waitForTimeout(600);
        const settingsOpen = await page.locator("dialog.shell-settings-dialog[open]").isVisible().catch(() => false);
        if (settingsOpen) {
          snapshots.push(await captureOpticalSnapshot(page, canvasPage, { label: `${sc.name}-settings-plugins`, outDir, dpr }));
          const pluginRow = page.locator("dialog.shell-settings-dialog[open] .shell-inventory-row, dialog.shell-settings-dialog[open] .shell-plugin-card").first();
          if (await pluginRow.isVisible().catch(() => false)) {
            await pluginRow.click().catch(() => {});
            await page.waitForTimeout(400);
            snapshots.push(await captureOpticalSnapshot(page, canvasPage, { label: `${sc.name}-settings-plugin-detail`, outDir, dpr }));
          }
          await page.evaluate(() => { window.location.hash = "models"; });
          await page.waitForTimeout(600);
          snapshots.push(await captureOpticalSnapshot(page, canvasPage, { label: `${sc.name}-settings-models`, outDir, dpr }));
          await page.keyboard.press("Escape");
          await page.waitForTimeout(300);
        }

        // 窄屏侧栏抽屉打开状态
        if (sc.mobile) {
          const navBtn = page.getByRole("button", { name: /打开侧边栏|导航|菜单/ }).first();
          if (await navBtn.isVisible().catch(() => false)) {
            await navBtn.click();
            await page.waitForTimeout(450);
            snapshots.push(await captureOpticalSnapshot(page, canvasPage, { label: `${sc.name}-drawer`, outDir, dpr }));
          }
        }
      }

      await context.close();
    }
  } finally {
    await browser.close();
  }

  const allItems = snapshots.flatMap((s) => s.results.map((r) => ({ view: s.label, ...r })));
  const findings = allItems
    .filter((item) => item.severityScore >= threshold)
    .sort((a, b) => b.severityScore - a.severityScore);
  const nestedChecks = snapshots.flatMap((s) => s.nestedChecks || []);
  const nestedFindings = nestedChecks.filter((c) => !c.passed);

  const report = {
    url: values.url,
    thresholdPx: threshold,
    totalInspected: allItems.length,
    findingCount: findings.length,
    nestedCheckCount: nestedChecks.length,
    nestedFindingCount: nestedFindings.length,
    snapshots: snapshots.map(({ label, viewport, count, overlayFile, sheetFile, nestedFile, composerFile }) => ({
      label, viewport, count, overlayFile, sheetFile, nestedFile, composerFile,
    })),
    nestedFindings,
    nestedChecks,
    findings,
    allItems,
  };

  writeFileSync(resolve(outDir, "report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({
    outDir,
    totalInspected: report.totalInspected,
    findingCount: report.findingCount,
    nestedCheckCount: report.nestedCheckCount,
    nestedFindingCount: report.nestedFindingCount,
    nestedFindings,
    topFindings: findings.slice(0, 15).map((f) => ({
      view: f.view,
      category: f.category,
      label: f.label,
      selectorHint: f.selectorHint,
      blurDx: f.blurDx,
      blurDy: f.blurDy,
      iconTextDeltaY: f.iconTextDeltaY,
      inkMargins: f.inkMargins,
      severityScore: f.severityScore,
    })),
  }, null, 2));

  if (values.strict && (findings.length > 0 || nestedFindings.length > 0)) {
    process.exitCode = 1;
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
