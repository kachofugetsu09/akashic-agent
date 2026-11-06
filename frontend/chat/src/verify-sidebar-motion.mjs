/**
 * 侧边栏减法重构与不对称折叠动效的真实浏览器验收：
 * 1. 字阶与对齐：12px 分区头 < 13.5px 项目名(550) / 会话标题(450) > 13px 项目子会话；
 *    品牌、搜索、分区头、项目文件夹图标、顶层会话标题共享同一条 20px 左起笔线，
 *    项目内子会话带 1px 树形竖线且标题与父项目名严格左对齐。
 * 2. 双路径项目新建交互：移除展开首行常驻的「+ 新对话」占位行；桌面端悬停项目行尾 120ms 同格浮现「+」与「⋯」（折叠箭头让位），
 *    未展开项目也可一步新建对话；空态 NewChatScope 具备清晰胶囊表面与 1px 细描边。
 * 3. 不对称过渡动效：展开 200ms (--ak-sys-duration-expand) 0fr→1fr + 内层 4px 下落淡入，
 *    收起 120ms (--ak-sys-duration-snap) 1fr→0fr 后卸载；逐帧采样验证中间帧高度。
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, writeFileSync, symlinkSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "vite";
import { chromium } from "playwright-core";
import { startDesktopFixtureServer } from "../../../scripts/webui-performance/desktop-fixture-server.mjs";

const repo = process.env.REVIEW_SOURCE_ROOT ?? resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
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
import { useState } from "react";
import { createRoot } from "react-dom/client";
import { DesktopSidebar } from ${JSON.stringify(resolve(repo, "frontend/chat/src/desktop-sidebar.tsx"))};
import { NewChatScope } from ${JSON.stringify(resolve(repo, "frontend/chat/src/new-chat-scope.tsx"))};
import { TooltipProvider } from ${JSON.stringify(resolve(repo, "frontend/chat/src/components/ui/tooltip.tsx"))};
import { initializeTheme, setTheme } from ${JSON.stringify(resolve(repo, "frontend/theme/src/theme-runtime.ts"))};
import ${JSON.stringify(resolve(repo, "frontend/chat/src/styles.css"))};

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
  const [activeId, setActiveId] = useState("p1-s1");
  const [activeGroup, setActiveGroup] = useState<{ dimension: string; id: string; label: string } | null>({
    dimension: "project",
    id: "no-recall",
    label: "akashic-no-recall",
  });
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set(["project:no-recall"]));
  const [lastStartedGroup, setLastStartedGroup] = useState("");

  const toggleGroup = (key: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };
  const expandGroup = (key: string) => {
    setExpanded((prev) => (prev.has(key) ? prev : new Set(prev).add(key)));
  };

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

  return (
    <TooltipProvider>
      <div className="akashic-chat">
        <main className="chat-shell" style={{ height: "100vh" }}>
          <div className="chat-shell-body" style={{ height: "100vh" }}>
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
  configFile: resolve(repo, "frontend/chat/vite.harness.config.ts"),
  root: work,
  logLevel: "warn",
  css: { postcss: resolve(repo, "frontend/chat") },
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
      for (let i = 0; i < frames; i += 1) {
        await new Promise((r) => requestAnimationFrame(r));
        const disclosure = group.querySelector(".navigation-group__disclosure");
        if (disclosure && !duration) duration = getComputedStyle(disclosure).transitionDuration;
        heights.push(disclosure ? Math.round(disclosure.getBoundingClientRect().height * 100) / 100 : 0);
      }
      return { heights, duration, final: heights.at(-1) };
    },
    { groupKey, frames },
  );
}

const report = {};

try {
  const context = await browser.newContext({ viewport: { width: 1080, height: 680 } });
  const page = await context.newPage();
  page.setDefaultTimeout(10_000);
  const errors = [];
  page.on("pageerror", (err) => errors.push(err.message));

  await page.goto(server.origin + "/");
  await page.locator(".navigation-group__sessions .conversation-session").first().waitFor();

  // 1. 验证常驻 "+ 新对话" 占位子行已彻底移除
  const newChatChildRows = await page.locator(".group-session--new").count();
  assert.equal(newChatChildRows, 0, "展开项目内不应再有常驻 .group-session--new 占位行");

  // 2. 验证左起笔线严格对齐（20px 主轴线 & 46px 子会话轴线）与字阶关系
  const geometry = await page.evaluate(() => {
    const xOf = (sel) => Math.round(document.querySelector(sel).getBoundingClientRect().left);
    const fontOf = (sel) => {
      const s = getComputedStyle(document.querySelector(sel));
      return { size: parseFloat(s.fontSize), weight: Number(s.fontWeight) };
    };
    const treeStyle = getComputedStyle(document.querySelector(".navigation-group__sessions"));
    const scopeStyle = getComputedStyle(document.querySelector(".new-chat-scope"));
    return {
      leftAxis: {
        pinnedHeader: xOf(".pinned-navigation .navigation-section__header span"),
        pinnedTitle: xOf('.pinned-navigation .conversation-session__title strong'),
        projectHeader: xOf('.navigation-section[aria-label="项目"] .navigation-section__header span'),
        projectFolder: xOf('[data-group-key="project:no-recall"] .folder-mark'),
        recentHeader: xOf(".conversation-navigation__heading--section span"),
        recentTitle: xOf('.conversation-session-list .conversation-session__title strong'),
      },
      nestedAxis: {
        projectName: xOf('[data-group-key="project:no-recall"] .navigation-group__name'),
        childSessionTitle: xOf('[data-group-key="project:no-recall"] .navigation-group__sessions .conversation-session__title strong'),
      },
      fonts: {
        sectionHeader: fontOf(".navigation-section__header"),
        projectRow: fontOf(".navigation-group__open"),
        projectName: fontOf(".navigation-group__name"),
        topSession: fontOf('.conversation-session-list .conversation-session__title strong'),
        childSession: fontOf('.navigation-group__sessions .conversation-session__title strong'),
      },
      treeBorderLeft: treeStyle.borderInlineStartWidth,
      scopeBorder: scopeStyle.borderTopWidth,
    };
  });
  report.geometry = geometry;

  const axisValues = Object.values(geometry.leftAxis);
  for (const val of axisValues) {
    assert.ok(Math.abs(val - axisValues[0]) <= 1, "左主轴线必须严格对齐 (20px)，实际：" + JSON.stringify(geometry.leftAxis));
  }
  assert.ok(
    Math.abs(geometry.nestedAxis.projectName - geometry.nestedAxis.childSessionTitle) <= 1,
    "项目下子会话标题必须与父项目名左对齐，实际：" + JSON.stringify(geometry.nestedAxis),
  );
  assert.equal(geometry.treeBorderLeft, "1px", "展开子树左侧应有 1px 树形引导线");
  assert.equal(geometry.scopeBorder, "1px", "NewChatScope 应有 1px 显式胶囊描边");
  assert.ok(
    geometry.fonts.sectionHeader.size < geometry.fonts.projectRow.size &&
      geometry.fonts.childSession.size < geometry.fonts.projectRow.size,
    "字阶应满足 分区头(12px) < 项目行(13.5px) > 子会话(13px)，实际：" + JSON.stringify(geometry.fonts),
  );

  // 截图 1：浅色默认态
  await page.screenshot({ path: root + "/01-light-default.png", animations: "disabled" });

  // 3. 开启 CDP screencast 录制完整动效过程（悬停项目行浮现 + → 收起 120ms → 展开 200ms）
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

  // 4. 验证路径 A：悬停折叠态项目行，箭头让位、行尾 "+" 浮现，点击直接在该项目新建会话
  const archRow = page.locator('[data-group-key="project:arch"] .navigation-group__row');
  const beforeHover = await page.evaluate(() => {
    const row = document.querySelector('[data-group-key="project:arch"]');
    return {
      plusOpacity: getComputedStyle(row.querySelector(".navigation-group__new-chat")).opacity,
      chevOpacity: getComputedStyle(row.querySelector(".navigation-group__chevron")).opacity,
    };
  });
  assert.equal(beforeHover.plusOpacity, "0", "未悬停时项目行 + 按钮应隐藏");
  assert.equal(beforeHover.chevOpacity, "1", "未悬停时折叠箭头应可见");

  const archBox = await archRow.boundingBox();
  await page.mouse.move(archBox.x + archBox.width / 2, archBox.y + archBox.height / 2, { steps: 6 });
  await page.waitForTimeout(180);

  const afterHover = await page.evaluate(() => {
    const row = document.querySelector('[data-group-key="project:arch"]');
    return {
      plusOpacity: getComputedStyle(row.querySelector(".navigation-group__new-chat")).opacity,
      chevOpacity: getComputedStyle(row.querySelector(".navigation-group__chevron")).opacity,
    };
  });
  report.hoverTransition = { beforeHover, afterHover };
  assert.equal(afterHover.plusOpacity, "1", "悬停项目行后 + 按钮应浮现");
  assert.equal(afterHover.chevOpacity, "0", "悬停项目行后折叠箭头应让位隐藏");
  await page.screenshot({ path: root + "/02-light-hover-project.png", animations: "disabled" });

  // 点击折叠项目行上的 "+" 按钮，无需展开直接在该项目新建对话
  await page.locator('[data-group-key="project:arch"] .navigation-group__new-chat').click();
  assert.equal(await page.locator("#started-group").textContent(), "arch", "点击项目行 + 应直接在该项目新建对话");

  // 5. 验证收起动效（120ms snap, 1fr → 0fr）与展开动效（200ms expand, 0fr → 1fr）逐帧采样
  await page.mouse.move(10, 10);
  await page.waitForTimeout(150);

  const collapse = await sampleGroupToggle(page, "project:no-recall", 18);
  report.collapse = collapse;
  assert.equal(collapse.duration, "0.12s", "收起过渡时长应为 120ms (0.12s)，实际：" + collapse.duration);
  assert.ok(middles(collapse.heights) >= 1, "收起过程应有中间过渡帧，实际：" + collapse.heights.join(","));
  await page.waitForTimeout(220);

  const expand = await sampleGroupToggle(page, "project:no-recall", 22);
  report.expand = expand;
  assert.equal(expand.duration, "0.2s", "展开过渡时长应为 200ms (0.2s)，实际：" + expand.duration);
  assert.ok(middles(expand.heights) >= 2, "展开过程应有多帧平滑高度过渡，实际：" + expand.heights.join(","));
  await page.waitForTimeout(250);

  await cdp.send("Page.stopScreencast").catch(() => {});

  // 6. 验证路径 B：空态 NewChatScope 点击打开下拉并切换归属
  await page.locator(".new-chat-scope").click();
  await page.getByRole("menuitem", { name: /不归属项目/ }).waitFor();
  await page.screenshot({ path: root + "/03-light-scope-menu.png", animations: "disabled" });
  await page.getByRole("menuitem", { name: /不归属项目/ }).click();
  assert.equal(await page.locator("#started-group").textContent(), "ungrouped");

  // 7. 深色模式截图验证
  await page.evaluate(() => (window).__setTheme("dark"));
  await page.waitForTimeout(100);
  await page.screenshot({ path: root + "/04-dark-default.png", animations: "disabled" });

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

  assert.deepEqual(errors, [], "页面不应有运行时错误");
  writeFileSync(root + "/results.json", JSON.stringify(report, null, 2));
  console.log("PASS verify-sidebar-motion", JSON.stringify({
    leftAxis: report.geometry.leftAxis,
    nestedAxis: report.geometry.nestedAxis,
    collapseFrames: report.collapse.heights,
    expandFrames: report.expand.heights,
  }));
} finally {
  await browser.close();
  await server.close();
}
console.log("证据目录: " + root);
