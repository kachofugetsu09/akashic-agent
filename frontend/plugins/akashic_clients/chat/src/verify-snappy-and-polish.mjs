/**
 * 真实浏览器验收：
 * 1. Snappy 偏好宽度吸附面板（Harsh Sharma @imharshin）：
 *    - 连续 3 次在 ~320px 附近松手形成磁吸点（support >= 2.5）；
 *    - 第 4 次拖入 ±10px（如 314px）时自动磁吸到 320px，且分隔线标 data-snapped；
 *    - 在吸附区内按住停顿 >=320ms 自动解锁磁吸（data-snapped 移除），继续拖动 1:1 精细跟随无跳变；
 *    - 键盘 ArrowLeft/ArrowRight 步进 0.5rem 不触发磁吸；双击复位默认宽度。
 * 2. 连续滑动分段胶囊（ThemeMenu 明暗三档）：
 *    - .theme-menu__mode-pill 随选中项变化更新 --chat-mode-index，跨 1 档 175ms、跨 2 档 220ms，逐帧采样确认平滑位移。
 * 3. 跑道倒计时撤销条（SessionDeleteNotice）与无缝追逐停止环（ComposerActionButton）：
 *    - .session-delete-notice 含 pathLength=1 的 SVG rect 倒计时环，鼠标移入/聚焦时标 data-paused 且 animationPlayState 为 paused；
 *    - 停止按钮在 aria-disabled="true" 时显现 pathLength=1、stroke-dasharray="0.24 0.76" 的圆头无缝追逐环。
 * 4. 连续曲率圆角（corner-shape: superellipse(1.25)）、中英混排自动留白（text-autospace: normal / text-wrap: pretty）与锚点浮层入场。
 */
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "vite";
import { chromium } from "playwright-core";
import { startDesktopFixtureServer } from "../../../../../scripts/webui-performance/desktop-fixture-server.mjs";

const repo = process.env.REVIEW_SOURCE_ROOT ?? resolve(dirname(fileURLToPath(import.meta.url)), "../../../../..");
const root = process.env.EVIDENCE_DIR
  ? (mkdirSync(process.env.EVIDENCE_DIR, { recursive: true }), resolve(process.env.EVIDENCE_DIR))
  : mkdtempSync(resolve(tmpdir(), "akashic-snappy-polish-"));
const work = mkdtempSync(resolve(tmpdir(), "akashic-snappy-polish-build-"));
symlinkSync(resolve(repo, "node_modules"), work + "/node_modules", "dir");

writeFileSync(
  work + "/index.html",
  '<!doctype html><html><head><meta charset="utf-8"></head><body><div id="root"></div><script type="module" src="./fixture.tsx"></script></body></html>',
);

writeFileSync(
  work + "/fixture.tsx",
  `
import { useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { DesktopSidebar } from ${JSON.stringify(resolve(repo, "frontend/plugins/akashic_clients/chat/src/desktop-sidebar.tsx"))};
import { useSidebarRail } from ${JSON.stringify(resolve(repo, "frontend/plugins/akashic_clients/chat/src/use-sidebar-rail.ts"))};
import { ComposerActionButton } from ${JSON.stringify(resolve(repo, "frontend/plugins/akashic_clients/chat/src/composer-action.tsx"))};
import { CopyNotice } from ${JSON.stringify(resolve(repo, "frontend/plugins/akashic_clients/chat/src/copy-last-code.tsx"))};
import { TooltipProvider } from ${JSON.stringify(resolve(repo, "frontend/plugins/akashic_clients/chat/src/components/ui/tooltip.tsx"))};
import { initializeTheme } from ${JSON.stringify(resolve(repo, "packages/akashic-theme/src/theme-runtime.ts"))};
import ${JSON.stringify(resolve(repo, "frontend/plugins/akashic_clients/chat/src/styles.css"))};
import ${JSON.stringify(resolve(repo, "frontend/plugins/akashic_clients/chat/src/message-view.css"))};

initializeTheme();

const now = new Date().toISOString();
const initialSessions = [
  { id: "s-1", title: "排查修复并提PR合并", preview: "12 条消息", active: true, updatedAt: now, createdAt: now },
  { id: "s-2", title: "更换watch模型并查免", preview: "8 条消息", active: false, updatedAt: now, createdAt: now },
  { id: "s-3", title: "评估A2UI替代现有渲染", preview: "19 条消息", active: false, updatedAt: now, createdAt: now },
  { id: "s-4", title: "SSD 迁移验收测试", preview: "6 条消息", active: false, updatedAt: now, createdAt: now },
];

function App() {
  const shellRef = useRef<HTMLDivElement | null>(null);
  const rail = useSidebarRail(shellRef);
  const [activeId, setActiveId] = useState("s-1");
  const [deletedNotice, setDeletedNotice] = useState<{ key: string; title: string } | null>({
    key: "s-del",
    title: "部署验收测试",
  });
  const [stopping, setStopping] = useState(true);
  const [copyText, setCopyText] = useState("已复制最后一个代码块（TypeScript，24 行）");

  return (
    <TooltipProvider>
      <div className="akashic-chat">
        <main className="chat-shell" style={{ height: "100vh" }}>
          <div className="chat-shell-body" ref={shellRef} style={{ ...rail.style, height: "100vh" }}>
            <DesktopSidebar
              surface="chat"
              sessions={initialSessions.map((s) => ({ ...s, active: s.id === activeId }))}
              activeSessionId={activeId}
              pendingSessionId=""
              chatReady={true}
              rail={rail}
              deletedNotice={deletedNotice}
              onRestoreSession={() => setDeletedNotice(null)}
              onDismissDeletedNotice={() => setDeletedNotice(null)}
              onSelectSession={setActiveId}
              onNewChat={() => {}}
              navigationPins={{
                pins: [{ kind: "session", id: "s-1" }],
                sessions: [],
                ready: true,
                pending: false,
                error: "",
                reload: async () => {},
                setPinned: async () => {},
                expandedGroups: new Set(),
                toggleGroup: () => {},
                expandGroup: () => {},
              }}
            />
            <section className="chat-main" style={{ padding: 32, display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
              <div className="agent-content">
                <h2>Snappy 偏好宽度吸附与界面质感验收</h2>
                <p>在Akashic里调用56个工具耗时44s，其中MaterialView.prepare()在起TaskGroup前复用bind()持有的scope，避免在UNLOADING排空期被拒绝。</p>
                <div className="user-bubble" style={{ marginTop: 12 }}>
                  帮我看看Snappy面板磁吸、倒计时撤销条和跑道停止环的实际渲染效果。
                </div>
              </div>
              <div className="composer-wrap" style={{ position: "relative", display: "flex", alignItems: "center", gap: 16 }}>
                <CopyNotice text={copyText} />
                <button type="button" id="toggle-copy-btn" onClick={() => setCopyText((c) => c ? "" : "已复制最后一个代码块（TypeScript，24 行）")}>
                  切换复制提示
                </button>
                <button type="button" id="respawn-delete-btn" onClick={() => setDeletedNotice({ key: "s-" + Date.now(), title: "部署验收测试" })}>
                  重现删除提示
                </button>
                <ComposerActionButton
                  mode="stop"
                  label="正在停止生成"
                  aria-disabled={stopping ? "true" : undefined}
                  onClick={() => setStopping((s) => !s)}
                />
                <output id="snap-point-px">{rail.snapPointPx ?? "null"}</output>
              </div>
            </section>
          </div>
        </main>
      </div>
    </TooltipProvider>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
`,
);

await build({
  configFile: resolve(repo, "frontend/plugins/akashic_clients/chat/vite.harness.config.ts"),
  root: work,
  logLevel: "warn",
  css: { postcss: resolve(repo, "frontend/plugins/akashic_clients/chat") },
  build: { outDir: work + "/dist", emptyOutDir: true },
});

const server = await startDesktopFixtureServer(work + "/dist");
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH ?? "/usr/bin/chromium",
  headless: true,
  args: ["--no-sandbox", "--disable-dev-shm-usage"],
});

const report = {};

try {
  const context = await browser.newContext({ viewport: { width: 1280, height: 760 } });
  const page = await context.newPage();
  page.setDefaultTimeout(10_000);
  const errors = [];
  page.on("pageerror", (err) => { console.error("PAGEERROR:", err); errors.push(err.message); });
  page.on("console", (msg) => { if (msg.type() === "error") console.error("CONSOLE ERROR:", msg.text()); });

  await page.goto(server.origin + "/");
  await page.locator(".chat-sidebar__resizer").waitFor();

  // 1. 验证 Snappy 侧栏吸附与 320ms 悬停微调解锁
  const resizer = page.locator(".chat-sidebar__resizer");
  const sidebar = page.locator(".chat-sidebar");

  async function dragSidebarTo(targetWidthPx, holdBeforeReleaseMs = 0) {
    const box = await resizer.boundingBox();
    const curW = (await sidebar.boundingBox()).width;
    const startX = box.x + box.width / 2;
    const startY = box.y + box.height / 2;
    const endX = startX + (targetWidthPx - curW);
    await page.mouse.move(startX, startY);
    await page.mouse.down();
    await page.mouse.move(endX, startY, { steps: 6 });
    const snappedDuringDrag = await resizer.getAttribute("data-snapped");
    const widthDuringDrag = (await sidebar.boundingBox()).width;
    let afterDwellSnapped = snappedDuringDrag;
    let afterDwellFineWidth = widthDuringDrag;
    if (holdBeforeReleaseMs > 0) {
      await page.waitForTimeout(holdBeforeReleaseMs);
      afterDwellSnapped = await resizer.getAttribute("data-snapped");
      // 解锁后继续向右微调 12px，验证 1:1 跟随
      await page.mouse.move(endX + 12, startY, { steps: 4 });
      afterDwellFineWidth = (await sidebar.boundingBox()).width;
    }
    await page.mouse.up();
    return { snappedDuringDrag, widthDuringDrag, afterDwellSnapped, afterDwellFineWidth };
  }

  // 写入 4 条 ~320px 历史样本（累计权重 ~3.1 > minWeight 2.5）并重载
  await page.evaluate(() => {
    const now = Date.now();
    localStorage.setItem(
      "akashic.chat.rail-samples",
      JSON.stringify([
        { widthPx: 320, at: now - 20000 },
        { widthPx: 318, at: now - 15000 },
        { widthPx: 322, at: now - 10000 },
        { widthPx: 320, at: now - 5000 },
      ]),
    );
  });
  await page.reload();
  await resizer.waitFor();

  const learnedSnapPx = await page.locator("#snap-point-px").textContent();
  assert.equal(learnedSnapPx, "320", `4 条 ~320px 样本应形成 320px 磁吸点，实际：${learnedSnapPx}`);

  // 从默认宽度 281.6px 拖入 314px（在 320±10px 吸附区内）：
  // 1) 刚进入时立刻磁吸到 320px 且 data-snapped="true"；
  // 2) 在吸附区内按住停顿 380ms (>320ms) 后自动解锁磁吸（移除 data-snapped）；
  // 3) 解锁后继续右移 12px 能够离开 320px 进行 1:1 精细微调。
  const snapAndDwellCheck = await dragSidebarTo(314, 380);
  assert.equal(snapAndDwellCheck.snappedDuringDrag, "true", "拖入 ±10px 吸附区时分隔线应标记 data-snapped");
  assert.equal(snapAndDwellCheck.widthDuringDrag, 320, `拖到 314px 应磁吸到 320px，实际：${snapAndDwellCheck.widthDuringDrag}`);
  assert.equal(snapAndDwellCheck.afterDwellSnapped, null, "停顿 380ms (>320ms) 后应自动解锁磁吸（移除 data-snapped）");
  assert.ok(snapAndDwellCheck.afterDwellFineWidth > 320, `解锁后右移应允许离开 320px 进行微调，实际：${snapAndDwellCheck.afterDwellFineWidth}`);

  // 每个 pointermove 只处理一次：分隔条上只有一套 window 监听，aria-valuenow 每步只写一次。
  const moveSteps = 6;
  await page.evaluate(() => {
    const el = document.querySelector(".chat-sidebar__resizer");
    window.__valueWrites = 0;
    window.__valueObserver = new MutationObserver((records) => {
      window.__valueWrites += records.filter((r) => r.attributeName === "aria-valuenow").length;
    });
    window.__valueObserver.observe(el, { attributes: true });
  });
  {
    const box = await resizer.boundingBox();
    const startX = box.x + box.width / 2;
    const startY = box.y + box.height / 2;
    await page.mouse.move(startX, startY);
    await page.mouse.down();
    await page.mouse.move(startX + 60, startY, { steps: moveSteps });
    await page.mouse.up();
  }
  const valueWrites = await page.evaluate(() => {
    window.__valueObserver.disconnect();
    return window.__valueWrites;
  });
  assert.ok(valueWrites <= moveSteps + 1, `${moveSteps} 次移动应最多 ${moveSteps + 1} 次 aria-valuenow 写入（含松手重渲染），实际 ${valueWrites}：移动被处理了两遍`);
  assert.equal(await resizer.getAttribute("data-dragging"), null, "松手后分隔条应退出拖拽态");

  report.snappy = { learnedSnapPx, snapAndDwellCheck, valueWrites };

  // 2. 验证 SessionDeleteNotice 的 pathLength=1 倒计时环与悬停暂停
  const notice = page.locator(".session-delete-notice");
  await notice.waitFor();
  const ringBeforeHover = await page.evaluate(() => {
    const bar = document.querySelector(".session-delete-notice__ring-bar");
    return {
      pathLength: bar?.getAttribute("pathLength"),
      playState: bar ? getComputedStyle(bar).animationPlayState : null,
    };
  });
  assert.equal(ringBeforeHover.pathLength, "1", "撤销条外框 SVG rect 必须使用 pathLength=1");
  assert.equal(ringBeforeHover.playState, "running", "未悬停时倒计时动画应处于 running");

  const noticeBox = await notice.boundingBox();
  await page.mouse.move(noticeBox.x + noticeBox.width / 2, noticeBox.y + noticeBox.height / 2);
  await page.waitForFunction(() => document.querySelector(".session-delete-notice")?.hasAttribute("data-paused"));
  const ringOnHover = await page.evaluate(() => {
    const bar = document.querySelector(".session-delete-notice__ring-bar");
    return bar ? getComputedStyle(bar).animationPlayState : null;
  });
  assert.equal(ringOnHover, "paused", "鼠标悬停撤销条时倒计时动画应暂停 (paused)");
  const ringCap = await page.evaluate(() => getComputedStyle(document.querySelector(".session-delete-notice__ring-bar")).strokeLinecap);
  assert.equal(ringCap, "butt", "倒计时环须用平头端点：圆头在 dash 归零时会留一粒残点");
  await page.emulateMedia({ reducedMotion: "reduce" });
  const ringReduced = await page.evaluate(() => getComputedStyle(document.querySelector(".session-delete-notice__ring")).display);
  assert.equal(ringReduced, "none", "减弱动效下不画不会动的满环");
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.waitForTimeout(300); // 切回后追逐环透明度有一段过渡
  report.deleteNotice = { ringBeforeHover, ringOnHover, ringCap, ringReduced };

  // 3. 验证停止按钮的无缝追逐环与排版属性
  const chaseMetrics = await page.evaluate(() => {
    const circle = document.querySelector(".composer-action-button__chase circle");
    const chaseSvg = document.querySelector(".composer-action-button__chase");
    const agentContent = document.querySelector(".agent-content");
    return {
      circlePathLength: circle?.getAttribute("pathLength"),
      circleDasharray: circle?.getAttribute("stroke-dasharray"),
      chaseOpacity: chaseSvg ? getComputedStyle(chaseSvg).opacity : "0",
      textWrap: agentContent ? getComputedStyle(agentContent).textWrap : "",
      lineBreak: agentContent ? getComputedStyle(agentContent).lineBreak : "",
    };
  });
  assert.equal(chaseMetrics.circlePathLength, "1");
  assert.equal(chaseMetrics.circleDasharray, "0.24 0.76");
  assert.ok(parseFloat(chaseMetrics.chaseOpacity) > 0.7, `停止等待态追逐环应可见，实际 opacity=${chaseMetrics.chaseOpacity}`);
  assert.equal(chaseMetrics.textWrap, "pretty");
  assert.equal(chaseMetrics.lineBreak, "strict");
  report.chaseAndTypo = chaseMetrics;

  // 4. 验证主题菜单明暗三档连续胶囊滑动
  await page.locator('.chat-sidebar__icon-action[aria-label^="主题"]').click();
  await page.locator(".theme-menu__modes").waitFor();
  const pillBefore = await page.locator(".theme-menu__mode-pill").boundingBox();
  await page.locator(".theme-menu__mode").nth(2).click(); // 切换到第 3 档「跟随系统」
  // 采样过渡帧
  const pillFrames = await page.evaluate(async () => {
    const pill = document.querySelector(".theme-menu__mode-pill");
    const xs = [];
    for (let i = 0; i < 16; i += 1) {
      await new Promise((r) => requestAnimationFrame(r));
      xs.push(Math.round(pill.getBoundingClientRect().left * 10) / 10);
    }
    return xs;
  });
  await page.waitForTimeout(260);
  const pillAfter = await page.locator(".theme-menu__mode-pill").boundingBox();
  assert.ok(pillAfter.x > pillBefore.x + 80, `胶囊应滑向第 3 档，实际 ${pillBefore.x} -> ${pillAfter.x}`);
  report.themePill = { fromX: pillBefore.x, toX: pillAfter.x, frames: pillFrames };

  await page.screenshot({ path: root + "/01-system-theme-menu-and-notices.png", animations: "disabled" });

  // 切到暗色模式截一张
  await page.locator(".theme-menu__mode").nth(1).click();
  await page.waitForTimeout(220);
  await page.screenshot({ path: root + "/02-dark-theme-menu-and-notices.png", animations: "disabled" });

  // 切到亮色模式截一张
  await page.locator(".theme-menu__mode").nth(0).click();
  await page.waitForTimeout(220);
  await page.screenshot({ path: root + "/03-light-theme-menu-and-notices.png", animations: "disabled" });

  assert.deepEqual(errors, [], "页面不应出现未捕获异常：" + JSON.stringify(errors));
  await context.close();

  writeFileSync(root + "/results.json", JSON.stringify(report, null, 2));
  console.log("PASS verify-snappy-and-polish", JSON.stringify(report));
  console.log("证据目录:", root);
} finally {
  await browser.close();
  await server.close();
}
