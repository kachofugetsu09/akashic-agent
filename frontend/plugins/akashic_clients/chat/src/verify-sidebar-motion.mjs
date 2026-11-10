/**
 * 侧边栏减法重构与不对称折叠动效的真实浏览器验收：
 * 1. 字阶与对齐：12px 分区头 < 13.5px 项目名(550) / 顶层会话(450) > 13px 项目子会话，且 .chat-shell 全局 --chat-type-control 保持 0.875rem；
 *    品牌标识、搜索图标、分区头、项目文件夹图标、顶层会话标题、空搜索提示共享同一条 20px 左起笔线（浮点误差 <= 0.5px），
 *    项目内子会话带 1px 树形竖线且标题与父项目名严格左对齐（46px）。
 * 2. 双路径项目新建与悬停几何稳定性：移除展开首行常驻的「+ 新对话」占位行；桌面端悬停项目行尾 120ms 原地淡出折叠箭头（0px 位移、标题 0px 缩窄）、
 *    浮现互不重叠的「+」与「⋯」，未展开项目也可一步新建对话；空态 NewChatScope 具备清晰胶囊表面与 1px 细描边。
 * 3. 不对称过渡动效与搜索瞬时性：用户点击展开 200ms (0fr→1fr)、收起 120ms (1fr→0fr，首帧即标 inert + aria-hidden 并移出 roving tabindex，结束后卸载 DOM)；
 *    搜索驱动的分组开合带 is-instant 瞬时完成、清空搜索首帧即卸载；紧凑抽屉（≤820px / 触屏）保持 12px/16px 变量（22px 轴线）与 44px 触控热区。
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
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
  : mkdtempSync(resolve(tmpdir(), "akashic-sidebar-motion-"));
const work = mkdtempSync(resolve(tmpdir(), "akashic-sidebar-motion-build-"));
symlinkSync(resolve(repo, "node_modules"), work + "/node_modules", "dir");

writeFileSync(
  work + "/index.html",
  '<!doctype html><html><head><meta charset="utf-8"></head><body><div id="root"></div><script type="module" src="./fixture.tsx"></script></body></html>',
);

writeFileSync(
  work + "/fixture.tsx",
  `
import { useCallback, useState } from "react";
import { createRoot } from "react-dom/client";
import { DesktopSidebar } from ${JSON.stringify(resolve(repo, "frontend/plugins/akashic_clients/chat/src/desktop-sidebar.tsx"))};
import { NewChatScope } from ${JSON.stringify(resolve(repo, "frontend/plugins/akashic_clients/chat/src/new-chat-scope.tsx"))};
import { TooltipProvider } from ${JSON.stringify(resolve(repo, "frontend/plugins/akashic_clients/chat/src/components/ui/tooltip.tsx"))};
import { initializeTheme, setTheme } from ${JSON.stringify(resolve(repo, "packages/akashic-theme/src/theme-runtime.ts"))};
import ${JSON.stringify(resolve(repo, "frontend/plugins/akashic_clients/chat/src/styles.css"))};

initializeTheme();
(window as any).__setTheme = (id: string) => setTheme(id);

const now = new Date().toISOString();
const initialSessions = [
  { id: "pin-1", title: "字节转正答辩材料梳理", preview: "", active: false, updatedAt: now, createdAt: now },
  { id: "p1-s1", title: "raft-source 记忆系统调研", preview: "", active: true, scope: { project: "no-recall" }, scoped: true, updatedAt: now, createdAt: now },
  { id: "p1-s2", title: "手机端侧虚拟屏调试", preview: "", active: false, scope: { project: "no-recall" }, scoped: true, updatedAt: now, createdAt: now },
  { id: "p2-s1", title: "插件边界与契约校验", preview: "", active: false, scope: { project: "arch" }, scoped: true, updatedAt: now, createdAt: now },
  { id: "rec-1", title: "西历2236 原文逐句精读", preview: "", active: false, updatedAt: now, createdAt: now },
  { id: "rec-2", title: "通勤纸面天气卡片调色", preview: "", active: false, status: "unread" as const, updatedAt: now, createdAt: now },
];

const projectGroups = [
  { id: "no-recall", label: "akashic-no-recall", detail: "记忆系统与端侧实验" },
  { id: "arch", label: "akashic-architecture", detail: "核心架构讨论" },
];

function App() {
  const compact = new URLSearchParams(location.search).get("compact") === "1";
  const [activeId, setActiveId] = useState("p1-s1");
  const [activeGroup, setActiveGroup] = useState<{ dimension: string; id: string; label: string } | null>({
    dimension: "project",
    id: "no-recall",
    label: "akashic-no-recall",
  });
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set(["project:no-recall"]));
  const [lastStartedGroup, setLastStartedGroup] = useState("");

  const toggleGroup = useCallback((key: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  const expandGroup = useCallback((key: string) => {
    setExpanded((prev) => (prev.has(key) ? prev : new Set(prev).add(key)));
  }, []);

  const groupsState = {
    sections: [
      {
        source: {
          key: "projects:r1",
          pluginId: "projects",
          pluginRevision: "r1",
          provider: {
            dimension: "project",
            label: "项目",
            create: { title: "创建项目", submitLabel: "创建", render: () => {} },
            load: async () => projectGroups,
          },
          context: () => ({ query: async () => projectGroups }),
        },
        groups: projectGroups,
      },
    ],
    revision: 0,
    ready: true,
    active: activeGroup,
    reload: () => {},
    startChat: (ref: { dimension: string; id: string }) => {
      const found = projectGroups.find((g) => g.id === ref.id);
      setActiveId("");
      setActiveGroup({ dimension: ref.dimension, id: ref.id, label: found?.label ?? ref.id });
      setLastStartedGroup(ref.id);
    },
  };

  const sessions = initialSessions.map((s) => ({ ...s, active: s.id === activeId }));

  const sidebar = (
    <DesktopSidebar
      surface="chat"
      sessions={sessions}
      activeSessionId={activeId}
      pendingSessionId=""
      chatReady={true}
      themeLabel="浅色"
      onSelectSession={setActiveId}
      onNewChat={() => {
        setActiveId("");
        setActiveGroup(null);
        setLastStartedGroup("ungrouped");
      }}
      onCycleTheme={() => {}}
      navigationPins={{
        pins: [{ kind: "session", id: "pin-1" }],
        sessions: [],
        ready: true,
        pending: false,
        error: "",
        reload: async () => {},
        setPinned: async () => {},
        expandedGroups: expanded,
        toggleGroup,
        expandGroup,
      }}
      groups={groupsState as any}
    />
  );

  if (compact) {
    return (
      <TooltipProvider>
        <div className="akashic-chat" data-chat-width="820" data-chat-touch="">
          <main className="chat-shell" style={{ height: "100vh" }}>
            <div role="dialog" className="compact-navigation-dialog" data-state="open" style={{ position: "relative" }}>
              {sidebar}
            </div>
          </main>
        </div>
      </TooltipProvider>
    );
  }

  return (
    <TooltipProvider>
      <div className="akashic-chat">
        <main className="chat-shell" style={{ height: "100vh" }}>
          <div className="chat-shell-body" style={{ height: "100vh" }}>
            {sidebar}
            <section className="chat-main is-empty" style={{ position: "relative", padding: 32, display: "grid", placeItems: "center" }}>
              <div className="composer-wrap home" style={{ position: "relative", inset: "auto", width: 420 }}>
                <NewChatScope
                  groups={groupsState as any}
                  onStart={(ref) => groupsState.startChat(ref)}
                  onUngrouped={() => {
                    setActiveId("");
                    setActiveGroup(null);
                    setLastStartedGroup("ungrouped");
                  }}
                />
                <div className="composer is-expanded" style={{ minHeight: 96, padding: 12 }}>
                  <div style={{ color: "var(--chat-faint)", fontSize: 14 }}>说说想做什么…</div>
                </div>
                <output id="started-group">{lastStartedGroup}</output>
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

const middles = (heights) => {
  const from = heights[0];
  const to = heights.at(-1);
  const low = Math.min(from, to);
  const high = Math.max(from, to);
  return heights.filter((h) => h > low + 0.5 && h < high - 0.5).length;
};

async function sampleGroupToggle(page, groupKey, frames = 22) {
  return page.evaluate(
    async ({ groupKey, frames }) => {
      const group = document.querySelector(`[data-group-key="${CSS.escape(groupKey)}"]`);
      const button = group.querySelector(".navigation-group__open");
      button.click();
      const heights = [];
      let duration = "";
      let firstFrameState = null;
      for (let i = 0; i < frames; i += 1) {
        await new Promise((r) => requestAnimationFrame(r));
        const disclosure = group.querySelector(".navigation-group__disclosure");
        if (i === 0) {
          firstFrameState = {
            mounted: Boolean(disclosure),
            ariaHidden: disclosure?.getAttribute("aria-hidden") ?? null,
            inert: disclosure?.hasAttribute("inert") ?? null,
            tabStopsInside: disclosure ? disclosure.querySelectorAll('[tabindex="0"]').length : 0,
          };
        }
        if (disclosure && !duration) duration = getComputedStyle(disclosure).transitionDuration;
        heights.push(disclosure ? Math.round(disclosure.getBoundingClientRect().height * 100) / 100 : 0);
      }
      return {
        heights,
        duration,
        firstFrameState,
        final: heights.at(-1),
        mountedAtEnd: Boolean(group.querySelector(".navigation-group__disclosure")),
      };
    },
    { groupKey, frames },
  );
}

const report = {};

try {
  // ===== 场景 1：宽屏桌面侧边栏 =====
  const context = await browser.newContext({ viewport: { width: 1080, height: 680 } });
  const page = await context.newPage();
  page.setDefaultTimeout(10_000);
  const errors = [];
  page.on("pageerror", (err) => errors.push(err.message));

  await page.goto(server.origin + "/");
  await page.locator(".navigation-group__sessions .conversation-session").first().waitFor();

  // 1. 验证常驻 "+ 新对话" 占位子行已彻底移除
  assert.equal(await page.locator(".group-session--new").count(), 0, "展开项目内不应再有常驻 .group-session--new 占位行");

  // 2. 验证左起笔线严格对齐（8 个点浮点误差 <= 0.5px）、字阶与全局 token 隔离
  const geometry = await page.evaluate(() => {
    const xOf = (sel) => document.querySelector(sel).getBoundingClientRect().left;
    const fontOf = (sel) => {
      const s = getComputedStyle(document.querySelector(sel));
      return { size: parseFloat(s.fontSize), weight: Number(s.fontWeight) };
    };
    const treeStyle = getComputedStyle(document.querySelector(".navigation-group__sessions"));
    const scopeStyle = getComputedStyle(document.querySelector(".new-chat-scope"));
    const shellControlToken = getComputedStyle(document.querySelector(".chat-shell")).getPropertyValue("--chat-type-control").trim();
    const sidebarControlToken = getComputedStyle(document.querySelector(".chat-sidebar")).getPropertyValue("--chat-type-control").trim();
    return {
      leftAxis: {
        brandMark: xOf(".chat-sidebar__mark"),
        searchIcon: xOf(".chat-sidebar__search svg"),
        pinnedHeader: xOf(".pinned-navigation .navigation-section__header span"),
        pinnedTitle: xOf(".pinned-navigation .conversation-session__title strong"),
        projectHeader: xOf('.navigation-section[aria-label="项目"] .navigation-section__header span'),
        projectFolder: xOf('[data-group-key="project:no-recall"] .folder-mark'),
        recentHeader: xOf(".conversation-navigation__heading--section span"),
        recentTitle: xOf(".conversation-session-list .conversation-session__title strong"),
      },
      nestedAxis: {
        projectName: xOf('[data-group-key="project:no-recall"] .navigation-group__name'),
        childSessionTitle: xOf('[data-group-key="project:no-recall"] .navigation-group__sessions .conversation-session__title strong'),
      },
      fonts: {
        sectionHeader: fontOf(".navigation-section__header"),
        projectRow: fontOf(".navigation-group__open"),
        projectName: fontOf(".navigation-group__name"),
        topSession: fontOf(".conversation-session-list .conversation-session__title strong"),
        childSessionActive: fontOf('[data-group-key="project:no-recall"] .conversation-session.active .conversation-session__title strong'),
        childSessionIdle: fontOf('[data-group-key="project:no-recall"] .conversation-session:not(.active) .conversation-session__title strong'),
      },
      tokens: { shellControlToken, sidebarControlToken },
      treeBorderLeft: treeStyle.borderInlineStartWidth,
      scopeBorder: scopeStyle.borderTopWidth,
    };
  });
  report.geometry = geometry;

  for (const [key, val] of Object.entries(geometry.leftAxis)) {
    assert.ok(Math.abs(val - 20) <= 0.5, `左主轴线 ${key} 应为 20px (±0.5px)，实际：${val}`);
  }
  assert.ok(
    Math.abs(geometry.nestedAxis.projectName - 46) <= 0.5 &&
      Math.abs(geometry.nestedAxis.childSessionTitle - 46) <= 0.5,
    "项目名与项目下子会话标题必须严格对齐在 46px (±0.5px)，实际：" + JSON.stringify(geometry.nestedAxis),
  );
  assert.equal(geometry.treeBorderLeft, "1px", "展开子树左侧应有 1px 树形引导线");
  assert.equal(geometry.scopeBorder, "1px", "NewChatScope 应有 1px 显式胶囊描边");
  assert.equal(parseFloat(geometry.tokens.shellControlToken), 0.875, ".chat-shell 全局 --chat-type-control 应保持 0.875rem");
  assert.equal(parseFloat(geometry.tokens.sidebarControlToken), 0.84375, ".chat-sidebar 局部 --chat-type-control 应为 0.84375rem");
  assert.deepEqual(geometry.fonts.sectionHeader, { size: 12, weight: 500 });
  assert.deepEqual(geometry.fonts.projectName, { size: 13.5, weight: 550 });
  assert.deepEqual(geometry.fonts.topSession, { size: 13.5, weight: 450 });
  assert.deepEqual(geometry.fonts.childSessionActive, { size: 13, weight: 550 });
  assert.deepEqual(geometry.fonts.childSessionIdle, { size: 13, weight: 450 });

  // 截图 1：浅色默认态
  await page.screenshot({ path: root + "/01-light-default.png", animations: "disabled" });

  // 3. 开启 CDP screencast 录制完整动效过程
  const framesDir = root + "/gif-frames";
  mkdirSync(framesDir, { recursive: true });
  const frameTimestamps = [];
  const cdp = await context.newCDPSession(page);
  cdp.on("Page.screencastFrame", async ({ data, sessionId, metadata }) => {
    const idx = frameTimestamps.length;
    frameTimestamps.push(metadata.timestamp);
    writeFileSync(`${framesDir}/${String(idx).padStart(5, "0")}.jpg`, Buffer.from(data, "base64"));
    await cdp.send("Page.screencastFrameAck", { sessionId }).catch(() => {});
  });
  await cdp.send("Page.startScreencast", {
    format: "jpeg",
    quality: 85,
    maxWidth: 760,
    maxHeight: 520,
    everyNthFrame: 1,
  });

  // 4. 验证路径 A 与悬停几何稳定性（箭头零位移、标题零收缩、+ 与 ⋯ 零重叠）
  const archRow = page.locator('[data-group-key="project:arch"] .navigation-group__row');
  const readArchHoverMetrics = () =>
    page.evaluate(() => {
      const row = document.querySelector('[data-group-key="project:arch"]');
      const nameRect = row.querySelector(".navigation-group__name").getBoundingClientRect();
      const chevEl = row.querySelector(".navigation-group__chevron");
      const plusEl = row.querySelector(".navigation-group__new-chat");
      const menuEl = row.querySelector(".navigation-menu-trigger");
      const chevRect = chevEl.getBoundingClientRect();
      const plusRect = plusEl.getBoundingClientRect();
      const menuRect = menuEl.getBoundingClientRect();
      return {
        nameWidth: Math.round(nameRect.width * 100) / 100,
        chevLeft: Math.round(chevRect.left * 100) / 100,
        plusRight: Math.round(plusRect.right * 100) / 100,
        menuLeft: Math.round(menuRect.left * 100) / 100,
        plusOpacity: getComputedStyle(plusEl).opacity,
        chevOpacity: getComputedStyle(chevEl).opacity,
      };
    });

  const beforeHover = await readArchHoverMetrics();
  assert.equal(beforeHover.plusOpacity, "0", "未悬停时项目行 + 按钮应隐藏");
  assert.equal(beforeHover.chevOpacity, "1", "未悬停时折叠箭头应可见");

  const archBox = await archRow.boundingBox();
  await page.mouse.move(archBox.x + archBox.width / 2, archBox.y + archBox.height / 2, { steps: 6 });
  await page.waitForTimeout(180);

  const afterHover = await readArchHoverMetrics();
  report.hoverTransition = { beforeHover, afterHover };
  assert.equal(afterHover.plusOpacity, "1", "悬停项目行后 + 按钮应浮现");
  assert.equal(afterHover.chevOpacity, "0", "悬停项目行后折叠箭头应原地淡出");
  assert.equal(afterHover.nameWidth, beforeHover.nameWidth, "悬停项目行时项目名宽度不应被压缩");
  assert.equal(afterHover.chevLeft, beforeHover.chevLeft, "悬停项目行时折叠箭头不应发生水平跳动");
  assert.ok(
    afterHover.plusRight <= afterHover.menuLeft,
    `+ 按钮右缘 (${afterHover.plusRight}) 不得与 ⋯ 菜单左缘 (${afterHover.menuLeft}) 重叠`,
  );
  await page.screenshot({ path: root + "/02-light-hover-project.png", animations: "disabled" });

  // 点击折叠项目行上的 "+" 按钮，无需展开直接在该项目新建对话
  await page.locator('[data-group-key="project:arch"] .navigation-group__new-chat').click();
  assert.equal(await page.locator("#started-group").textContent(), "arch", "点击项目行 + 应直接在该项目新建对话");
  // 将 arch 收回折叠态，以便后续专门验证「未展开项目在搜索命中与清空时的瞬时挂载/卸载」
  await page.locator('[data-group-key="project:arch"] .navigation-group__open').click();
  await page.waitForFunction(
    () => document.querySelector('[data-group-key="project:arch"] .navigation-group__disclosure') === null,
  );

  // 5. 验证收起动效（120ms snap, 1fr → 0fr）、首帧 inert/aria-hidden/roving 排除与过渡后 DOM 卸载
  await page.mouse.move(10, 10);
  await page.waitForTimeout(150);

  // 先将焦点放在即将被收起的子会话上，确认收起首帧 roving tabindex 立即移出
  await page.locator('[data-group-key="project:no-recall"] .conversation-session').first().focus();
  const collapse = await sampleGroupToggle(page, "project:no-recall", 18);
  report.collapse = collapse;
  assert.equal(collapse.duration, "0.12s", "收起过渡时长应为 120ms (0.12s)，实际：" + collapse.duration);
  assert.deepEqual(
    collapse.firstFrameState,
    { mounted: true, ariaHidden: "true", inert: true, tabStopsInside: 0 },
    "收起第 1 帧必须挂载且标 aria-hidden=true、inert 并清零内部 tabindex=0，实际：" + JSON.stringify(collapse.firstFrameState),
  );
  assert.ok(middles(collapse.heights) >= 1, "收起过程应有中间过渡帧，实际：" + collapse.heights.join(","));
  await page.waitForFunction(
    () => document.querySelector('[data-group-key="project:no-recall"] .navigation-group__disclosure') === null,
  );
  assert.equal(
    await page.locator('[data-group-key="project:no-recall"] .navigation-group__disclosure').count(),
    0,
    "收起过渡结束后 .navigation-group__disclosure 应从 DOM 卸载",
  );

  // 6. 验证展开动效（200ms expand, 0fr → 1fr）逐帧采样
  const expand = await sampleGroupToggle(page, "project:no-recall", 22);
  report.expand = expand;
  assert.equal(expand.duration, "0.2s", "展开过渡时长应为 200ms (0.2s)，实际：" + expand.duration);
  assert.ok(middles(expand.heights) >= 2, "展开过程应有多帧平滑高度过渡，实际：" + expand.heights.join(","));
  await page.waitForTimeout(240);

  await cdp.send("Page.stopScreencast").catch(() => {});

  // 7. 验证搜索驱动的开合走 is-instant（0s 过渡、清空搜索立即卸载，无 140ms 残留，且无匹配提示左对齐 20px）
  const searchInput = page.getByRole("textbox", { name: "搜索会话" });
  await searchInput.fill("插件边界");
  const searchOpenState = await page.evaluate(() => {
    const disc = document.querySelector('[data-group-key="project:arch"] .navigation-group__disclosure');
    return {
      mounted: Boolean(disc),
      isInstant: disc?.classList.contains("is-instant") ?? false,
      duration: disc ? getComputedStyle(disc).transitionDuration : null,
    };
  });
  report.searchOpenState = searchOpenState;
  assert.deepEqual(
    searchOpenState,
    { mounted: true, isInstant: true, duration: "0s" },
    "搜索展开命中分组应带 is-instant 且过渡时长为 0s",
  );

  await searchInput.fill("__no_match_at_all__");
  const emptySearchLeft = await page.evaluate(
    () => document.querySelector(".navigation-search-empty").getBoundingClientRect().left,
  );
  assert.ok(Math.abs(emptySearchLeft - 20) <= 0.5, `空搜索提示左缘应为 20px (±0.5px)，实际：${emptySearchLeft}`);

  await searchInput.press("Escape");
  const afterSearchClear = await page.evaluate(() => ({
    archCount: document.querySelectorAll('[data-group-key="project:arch"] .navigation-group__disclosure').length,
    noRecallInstant: document
      .querySelector('[data-group-key="project:no-recall"] .navigation-group__disclosure')
      ?.classList.contains("is-instant") ?? false,
  }));
  assert.deepEqual(
    afterSearchClear,
    { archCount: 0, noRecallInstant: true },
    "清空搜索后：折叠项目应立即卸载，原已展开项目重新挂载应带 is-instant 不重播动画",
  );

  // 8. 验证路径 B：空态 NewChatScope 点击打开下拉并切换归属
  await page.locator(".new-chat-scope").click();
  await page.getByRole("menuitem", { name: /不归属项目/ }).waitFor();
  await page.screenshot({ path: root + "/03-light-scope-menu.png", animations: "disabled" });
  await page.getByRole("menuitem", { name: /不归属项目/ }).click();
  assert.equal(await page.locator("#started-group").textContent(), "ungrouped");

  // 9. 深色模式截图验证
  await page.evaluate(() => (window).__setTheme("dark"));
  await page.waitForTimeout(100);
  await page.screenshot({ path: root + "/04-dark-default.png", animations: "disabled" });

  assert.deepEqual(errors, [], "桌面页面不应有运行时错误");
  await context.close();

  // ===== 场景 2：紧凑抽屉与触屏断点（≤820px, data-chat-touch）=====
  const compactCtx = await browser.newContext({ viewport: { width: 780, height: 840 } });
  const compactPage = await compactCtx.newPage();
  const compactErrors = [];
  compactPage.on("pageerror", (err) => compactErrors.push(err.message));
  await compactPage.goto(server.origin + "/?compact=1");
  await compactPage.locator(".navigation-group__open").first().waitFor();

  const compactMetrics = await compactPage.evaluate(() => {
    const sidebar = document.querySelector(".compact-navigation-dialog .chat-sidebar");
    const sidebarLeft = sidebar.getBoundingClientRect().left;
    const cs = getComputedStyle(sidebar);
    const headerX = document.querySelector(".pinned-navigation .navigation-section__header span").getBoundingClientRect().left - sidebarLeft;
    const row = document.querySelector('[data-group-key="project:arch"]');
    const chev = row.querySelector(".navigation-group__chevron");
    const plus = row.querySelector(".navigation-group__new-chat");
    const chevRect = chev.getBoundingClientRect();
    const plusRect = plus.getBoundingClientRect();
    return {
      railInline: cs.getPropertyValue("--chat-rail-inline").trim(),
      groupGap: cs.getPropertyValue("--chat-group-gap").trim(),
      headerX,
      chevOpacity: getComputedStyle(chev).opacity,
      plusOpacity: getComputedStyle(plus).opacity,
      plusWidth: plusRect.width,
      plusHeight: plusRect.height,
      chevRight: Math.round(chevRect.right * 100) / 100,
      plusLeft: Math.round(plusRect.left * 100) / 100,
    };
  });
  report.compactMetrics = compactMetrics;
  assert.equal(compactMetrics.railInline, "12px", "紧凑抽屉内 .chat-sidebar 的 --chat-rail-inline 应为 12px");
  assert.equal(compactMetrics.groupGap, "16px", "紧凑抽屉内 .chat-sidebar 的 --chat-group-gap 应为 16px");
  assert.ok(Math.abs(compactMetrics.headerX - 22) <= 0.5, `紧凑抽屉左轴线应为 22px (±0.5px)，实际：${compactMetrics.headerX}`);
  assert.equal(compactMetrics.chevOpacity, "1", "触屏下折叠箭头应常驻可见");
  assert.equal(compactMetrics.plusOpacity, "1", "触屏下项目行 + 按钮应常驻可见");
  assert.equal(compactMetrics.plusWidth, 44, "触屏下项目行 + 按钮热区宽应为 44px");
  assert.equal(compactMetrics.plusHeight, 44, "触屏下项目行 + 按钮热区高应为 44px");
  assert.ok(compactMetrics.chevRight <= compactMetrics.plusLeft, "触屏下折叠箭头与 + 按钮不得重叠");
  await compactPage.screenshot({ path: root + "/05-compact-touch-drawer.png", animations: "disabled" });
  assert.deepEqual(compactErrors, [], "紧凑抽屉页面不应有运行时错误");
  await compactCtx.close();

  // 合成 GIF（若安装了 ffmpeg 且帧数 >= 2）
  if (frameTimestamps.length >= 2) {
    const lines = [];
    for (let i = 0; i < frameTimestamps.length; i += 1) {
      const f = `${framesDir}/${String(i).padStart(5, "0")}.jpg`;
      const dur = i + 1 < frameTimestamps.length
        ? Math.max(0.04, Math.min(0.6, frameTimestamps[i + 1] - frameTimestamps[i]))
        : 0.4;
      lines.push(`file '${f}'`, `duration ${dur.toFixed(4)}`);
    }
    lines.push(`file '${framesDir}/${String(frameTimestamps.length - 1).padStart(5, "0")}.jpg'`);
    const concatPath = `${framesDir}/concat.txt`;
    writeFileSync(concatPath, lines.join("\n") + "\n");
    try {
      execFileSync("ffmpeg", [
        "-y", "-f", "concat", "-safe", "0", "-i", concatPath,
        "-vf", "fps=15,scale=680:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];[s1][p]paletteuse",
        root + "/sidebar-motion.gif",
      ], { stdio: "ignore" });
      report.gif = root + "/sidebar-motion.gif";
    } catch {
      // CI 环境无 ffmpeg 时跳过 GIF 合成
    }
  }

  writeFileSync(root + "/results.json", JSON.stringify(report, null, 2));
  console.log("PASS verify-sidebar-motion", JSON.stringify({
    leftAxis: report.geometry.leftAxis,
    nestedAxis: report.geometry.nestedAxis,
    hoverTransition: report.hoverTransition,
    collapseFirstFrame: report.collapse.firstFrameState,
    searchOpenState: report.searchOpenState,
    compactMetrics: report.compactMetrics,
  }));
} finally {
  await browser.close();
  await server.close();
}
console.log("证据目录: " + root);
