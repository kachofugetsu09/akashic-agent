/** 聊天阅读排版验收：验证 390px/320px 窄屏与 1280px 宽屏共用字阶、字体、间距，且无横向溢出。 */
import { createServer } from "node:http";
import { readFileSync, existsSync, writeFileSync, mkdirSync, mkdtempSync } from "node:fs";
import { resolve, extname, dirname } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import assert from "node:assert/strict";
import { WebSocketServer } from "ws";
import { chromium } from "playwright-core";
import { desktopModels } from "../../../scripts/webui-performance/fixtures.mjs";

const repo = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
const root = process.env.EVIDENCE_DIR ?? mkdtempSync(resolve(tmpdir(), "akashic-mobile-reading-"));
mkdirSync(root, { recursive: true });

const build = spawnSync(
  process.execPath,
  [resolve(repo, "node_modules/vite/bin/vite.js"), "build", "--config", "frontend/chat/vite.harness.config.ts", "--outDir", root + "/dist"],
  { cwd: repo, encoding: "utf8" },
);
writeFileSync(root + "/build.log", (build.stdout ?? "") + (build.stderr ?? ""));
assert.equal(build.status, 0, "前端构建失败，见 " + root + "/build.log");

const SAMPLE_MD = `网下，万一 \`adbd\` 端口被扫到会有风险。所以 Android 的系统设置直接监听了 \`NetworkCapabilities.TRANSPORT_WIFI\` ——**只要当前没连 Wi-Fi，系统就会自动把「无线调试」开关强制关掉并置灰。**

**2. 但最关键的真相是：只需要在拉起 \`HelperMain\` 的那一秒用一下 ADB，之后开流量也一直能用！**

你看 \`HelperMain\` 启动那条命令里的两个词：\`nohup setsid app_process ...\`。

- 当你在家里 Wi-Fi 下配对并启动过一次（或者用电脑插 USB 跑过一次 \`adb shell\`）之后，这个运行在 \`UID 2000\` 的 \`HelperMain\` **常驻进程已经脱离了 \`adbd\` 的进程组，归系统的 \`init\` 进程管了！**
- 它和 \`akashic-android-shell\` 之间走的是手机内存里的 **Binder IPC**，而 \`akashic-android-shell\` 和你家里的 \`hua-home\` 服务器之间走的是普通的 HTTPS/WebSocket。
- **也就是说：只要 \`HelperMain\` 在手机开机/连 Wi-Fi 时拉起过一次，就算你出门切到 5G 流量，系统把“无线调试”自动关掉了，\`HelperMain\` (UID 2000) 依然活在手机后台！** 你在外面用 5G 流量跟我聊天，照样能用后台独立虚拟屏幕，直到你重启手机为止！

**3. 手机随便在外面没 Wi-Fi 时怎么开无线调试？**

如果你人在外面只有 5G 流量，又刚好重启了手机想开无线调试，还有两个小技巧：`;

const sessionKey = "akashic:mobile-reading-verify";
const items = [
  {
    id: "msg-user-1",
    seq: 0,
    session_id: sessionKey,
    timestamp: "2026-11-04T12:44:00.000Z",
    author: "user",
    actor: "human",
    source: "akashic",
    attachments: [],
    metadata: {},
    body: { kind: "input", parts: [{ kind: "text", value: "阅读当前项目的 suba" }] },
  },
  {
    id: "msg-assistant-1",
    seq: 1,
    session_id: sessionKey,
    timestamp: "2026-11-04T12:45:00.000Z",
    author: "assistant",
    actor: "model",
    source: "akashic",
    attachments: [],
    metadata: {},
    body: { kind: "output", finish: "complete", parts: [{ kind: "reasoning", display: "thinking", text: "先核对 **字体** 与间距。\n\n宽窄屏共用一套排版。" }, { kind: "text", value: SAMPLE_MD }] },
  },
];

const json = (res, data, status = 200) => {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(data));
};

const server = createServer((req, res) => {
  const path = new URL(req.url, "http://localhost").pathname;
  if (path === "/api/shell/state") return json(res, { status: "ready", configured: true, chatReady: true });
  if (path === "/api/chat/models") return json(res, desktopModels(1));
  if (path === "/api/chat/sessions") {
    return json(res, {
      items: [{ key: sessionKey, first_message_content: "阅读当前项目的 suba", updated_at: items[1].timestamp, message_count: 2 }],
      next_cursor: null,
    });
  }
  if (path.startsWith("/api/chat/sessions/")) {
    return json(res, { version: 2, items, through_seq: 1, has_more: false, before_seq: null });
  }
  if (path === "/api/chat/plugin-ui/catalog") return json(res, { catalog_revision: "0".repeat(64), items: [] });
  if (path === "/api/runtime/host-bridge") return json(res, { state: "healthy", failures: 0 });
  const file = resolve(root + "/dist", path === "/" || path === "/chat" ? "index.html" : path.replace(/^\/assets\//, "").replace(/^\//, ""));
  if (!file.startsWith(root + "/dist/") || !existsSync(file)) return res.writeHead(404).end();
  const mime = { ".js": "text/javascript", ".css": "text/css", ".html": "text/html", ".woff2": "font/woff2", ".svg": "image/svg+xml" }[extname(file)];
  res.writeHead(200, { "content-type": mime ?? "application/octet-stream" });
  res.end(readFileSync(file));
});

const ws = new WebSocketServer({ noServer: true });
server.on("upgrade", (req, socket, head) => {
  if (new URL(req.url, "http://localhost").pathname === "/ws") {
    ws.handleUpgrade(req, socket, head, (client) => {
      client.send(JSON.stringify({ type: "reply.status", version: 2, session_id: sessionKey, snapshot_id: "snap-1", available: true, items: [] }));
    });
  } else socket.destroy();
});

await new Promise((r) => server.listen(0, "127.0.0.1", r));
const origin = "http://127.0.0.1:" + server.address().port;
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH ?? "/usr/bin/chromium",
  headless: true,
  args: ["--no-sandbox"],
});

const results = {};

/** 读取真实消息组件的排版，比较视口间可观察的差异。 */
function readingMetrics() {
  const topic = document.querySelector(".agent-content .reading-topic");
  const content = topic.closest(".agent-content");
  const topicStrong = document.querySelector(".agent-content .reading-topic > strong");
  const code = document.querySelector(".agent-content code");
  const ul = document.querySelector(".agent-content .markdown-reading ul");
  const items = [...document.querySelectorAll(".agent-content .markdown-reading ul > li")];
  const bodyStrong = document.querySelector(".agent-content ul > li strong");

  const csContent = getComputedStyle(content);
  const csTopic = getComputedStyle(topic);
  const csTopicStrong = getComputedStyle(topicStrong);
  const csCode = getComputedStyle(code);
  const csUl = getComputedStyle(ul);
  const csSecondLi = getComputedStyle(items[1]);
  const csBodyStrong = getComputedStyle(bodyStrong);

  const bodyFontSize = parseFloat(csContent.fontSize);
  const bodyLineHeight = parseFloat(csContent.lineHeight);
  const lineLeadingGap = bodyLineHeight - bodyFontSize;

  return {
    fontFamily: csContent.fontFamily,
    bodyFontSize,
    bodyLineHeight,
    paragraphGap: getComputedStyle(content.querySelector("p")).marginBlockEnd,
    lineLeadingGap,
    topicFontSize: parseFloat(csTopic.fontSize),
    topicLineHeight: parseFloat(csTopic.lineHeight),
    topicMarginTop: parseFloat(csTopic.marginBlockStart),
    topicMarginBottom: parseFloat(csTopic.marginBlockEnd),
    topicWeight: Number(csTopicStrong.fontWeight),
    bodyStrongWeight: Number(csBodyStrong.fontWeight),
    codeFontSize: parseFloat(csCode.fontSize),
    codePaddingInline: parseFloat(csCode.paddingInlineStart),
    ulPaddingInlineStart: parseFloat(csUl.paddingInlineStart),
    listItemGap: parseFloat(csSecondLi.marginBlockStart),
  };
}

try {
  // 1. 390px 手机视口：验证 13 : 15 : 17px 三阶尺、亲密性间距（列表项间距 > 行内空隙）与界面无衬线字体
  const mobileCtx = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2 });
  const mobilePage = await mobileCtx.newPage();
  const errors = [];
  mobilePage.on("pageerror", (e) => errors.push(e.message));
  await mobilePage.goto(origin + "/?session=" + encodeURIComponent(sessionKey));
  await mobilePage.locator(".agent-content .markdown-reading").waitFor();
  await mobilePage.evaluate(() => document.fonts.ready);

  const mobileMetrics = await mobilePage.evaluate(readingMetrics);

  results.mobile390 = mobileMetrics;
  assert.equal(errors.length, 0, "页面报错：" + JSON.stringify(errors));
  assert.ok(await mobilePage.evaluate(() => document.documentElement.scrollWidth <= innerWidth), "390px 发生横向溢出");
  assert.ok(!mobileMetrics.fontFamily.includes("Source Serif 4"), "聊天正文应使用界面无衬线字体，实际：" + mobileMetrics.fontFamily);
  assert.equal(mobileMetrics.bodyFontSize, 15, "窄屏正文字号应为 15px (0.9375rem)，实际：" + mobileMetrics.bodyFontSize);
  assert.equal(mobileMetrics.bodyLineHeight, 24, "窄屏正文行高应为 24px (1.6)，实际：" + mobileMetrics.bodyLineHeight);
  assert.ok(Math.abs(mobileMetrics.topicFontSize - 17) < 0.2, "小标题字号应约为 17px (1.1333em)，实际：" + mobileMetrics.topicFontSize);
  assert.ok(Math.abs(mobileMetrics.codeFontSize - 12.75) < 0.25, "行内代码字号应约为 12.75px (0.85em)，实际：" + mobileMetrics.codeFontSize);
  assert.ok(
    mobileMetrics.codeFontSize < mobileMetrics.bodyFontSize && mobileMetrics.bodyFontSize < mobileMetrics.topicFontSize,
    "未形成 code < body < topic 三阶字号梯度：" + JSON.stringify(mobileMetrics),
  );
  assert.ok(
    mobileMetrics.topicMarginTop > mobileMetrics.topicMarginBottom * 2,
    "小标题未满足上疏下亲的亲密性原则：" + JSON.stringify(mobileMetrics),
  );
  assert.ok(
    mobileMetrics.listItemGap > mobileMetrics.lineLeadingGap,
    "列表项间距应大于行内空隙（亲密性原则），实际：" + JSON.stringify(mobileMetrics),
  );
  assert.ok(
    mobileMetrics.topicWeight > mobileMetrics.bodyStrongWeight,
    "标题字重应高于正文加粗字重，避免强调通胀：" + JSON.stringify(mobileMetrics),
  );
  assert.ok(mobileMetrics.ulPaddingInlineStart <= 18, "窄屏列表缩进应 <= 18px 以释放行宽，实际：" + mobileMetrics.ulPaddingInlineStart);

  await mobilePage.screenshot({ path: root + "/mobile-390-settled.png", animations: "disabled" });
  await mobilePage.locator(".thinking-row .process-trigger").click();
  const thinkingFont = await mobilePage.locator(".process-text").evaluate((node) => getComputedStyle(node).fontFamily);
  assert.equal(thinkingFont, mobileMetrics.fontFamily, "展开的思考应沿用聊天字体");
  results.thinkingFont = thinkingFont;
  await mobileCtx.close();

  // 2. 320px 极窄视口：确认长标识符换行不产生横向滚动条
  const narrowCtx = await browser.newContext({ viewport: { width: 320, height: 720 } });
  const narrowPage = await narrowCtx.newPage();
  await narrowPage.goto(origin + "/?session=" + encodeURIComponent(sessionKey));
  await narrowPage.locator(".agent-content .markdown-reading").waitFor();
  const narrowOverflow = await narrowPage.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    innerWidth: window.innerWidth,
  }));
  results.narrow320 = narrowOverflow;
  assert.ok(narrowOverflow.scrollWidth <= narrowOverflow.innerWidth, "320px 发生横向溢出：" + JSON.stringify(narrowOverflow));
  await narrowPage.screenshot({ path: root + "/mobile-320-settled.png", animations: "disabled" });
  await narrowCtx.close();

  // 3. 1280px 宽屏：同一内容的排版与手机完全一致，只有列宽和控件布局响应视口。
  const desktopCtx = await browser.newContext({ viewport: { width: 1280, height: 840 } });
  const desktopPage = await desktopCtx.newPage();
  await desktopPage.goto(origin + "/?session=" + encodeURIComponent(sessionKey));
  await desktopPage.locator(".agent-content .markdown-reading").waitFor();
  await desktopPage.evaluate(() => document.fonts.ready);
  const desktopMetrics = await desktopPage.evaluate(readingMetrics);
  results.desktop1280 = desktopMetrics;
  assert.deepEqual(desktopMetrics, mobileMetrics, "宽窄屏的字体、字阶或间距不同");
  assert.ok(await desktopPage.evaluate(() => document.documentElement.scrollWidth <= innerWidth), "1280px 发生横向溢出");
  await desktopPage.screenshot({ path: root + "/desktop-1280-settled.png", animations: "disabled" });
  await desktopPage.locator(".thinking-row .process-trigger").click();
  assert.equal(await desktopPage.locator(".process-text").evaluate((node) => getComputedStyle(node).fontFamily), thinkingFont, "宽窄屏思考字体不同");
  await desktopCtx.close();

  writeFileSync(root + "/results.json", JSON.stringify(results, null, 2));
  console.log("验收通过，证据目录:", root);
} finally {
  await browser.close();
  server.close();
}
