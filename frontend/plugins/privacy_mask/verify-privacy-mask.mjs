/**
 * 截屏隐私打码插件（privacy_mask@release）真实浏览器端到端验收：
 * 1. 正交挂载与一键开关：同时验证 shell.rail-actions.v1 底栏按钮、header.end 顶栏标记与 Alt+M 全局快捷键；
 * 2. 维度一（导航与会话标题骨架条）：侧栏会话标题、项目分组名、项目子会话、顶栏标题与项目副标题化为 48%/74%/56%/62% 错落圆角骨架条，行几何尺寸零偏移，title/data-tip 悬停提示同步剥离并可无损恢复；
 * 3. 维度二（全部内容模糊打码）：用户消息气泡内容、助手回复正文、代码块、工具执行步骤与结果、输入框草稿及工作目录路径整体施加 filter: blur(6px) 高斯模糊（保留外层气泡与卡片轮廓），支持与「导航与会话标题」正交独立开关；
 * 4. 生命周期隔离：插件卸载（release）后彻底清除全部 data-privacy-* 属性与全局单例，零残留。
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, writeFileSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "vite";
import { chromium } from "playwright-core";
import { startDesktopFixtureServer } from "../../../scripts/webui-performance/desktop-fixture-server.mjs";

const repo = process.env.REVIEW_SOURCE_ROOT ?? resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
const root = process.env.EVIDENCE_DIR
  ? (mkdirSync(process.env.EVIDENCE_DIR, { recursive: true }), resolve(process.env.EVIDENCE_DIR))
  : mkdtempSync(resolve(tmpdir(), "akashic-privacy-mask-"));
const work = mkdtempSync(resolve(tmpdir(), "akashic-privacy-mask-build-"));
symlinkSync(resolve(repo, "node_modules"), work + "/node_modules", "dir");

writeFileSync(
  work + "/index.html",
  '<!doctype html><html><head><meta charset="utf-8"></head><body><div id="root"></div><script type="module" src="./fixture.tsx"></script></body></html>',
);

writeFileSync(
  work + "/fixture.tsx",
  `
import { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Search, MessageSquarePlus, Terminal, Folder } from "lucide-react";
import { DesktopSidebar } from ${JSON.stringify(resolve(repo, "frontend/chat/src/desktop-sidebar.tsx"))};
import { exposeChatRuntime } from ${JSON.stringify(resolve(repo, "frontend/chat/src/chat-ui-runtime.tsx"))};
import { connectHost } from ${JSON.stringify(resolve(repo, "frontend/chat/src/chat-host.ts"))};
import { setShellRailActions } from ${JSON.stringify(resolve(repo, "frontend/chat/src/shell-rail-actions.ts"))};
import { TooltipProvider } from ${JSON.stringify(resolve(repo, "frontend/chat/src/components/ui/tooltip.tsx"))};
import { initializeTheme, setTheme } from ${JSON.stringify(resolve(repo, "frontend/theme/src/theme-runtime.ts"))};
import ${JSON.stringify(resolve(repo, "frontend/chat/src/styles.css"))};
import ${JSON.stringify(resolve(repo, "frontend/chat/src/header-meta.css"))};
import { activate as activatePrivacyWeb } from ${JSON.stringify(resolve(repo, "frontend/plugins/privacy_mask/src/index.tsx"))};
import privacySlotDefinition from ${JSON.stringify(resolve(repo, "frontend/plugins/privacy_mask/src/plugin-ui.tsx"))};

exposeChatRuntime();
initializeTheme();
(window as any).__setTheme = (id: string) => setTheme(id);

const now = new Date().toISOString();
const initialSessions = [
  { id: "pin-1", title: "字节转正答辩材料梳理", preview: "", active: false, updatedAt: now, createdAt: now },
  { id: "p1-s1", title: "开发机 Alma 与 Magpie 截屏打码调研", preview: "", active: true, scope: { project: "no-recall" }, scoped: true, updatedAt: now, createdAt: now },
  { id: "p1-s2", title: "手机端侧无线调试虚拟屏", preview: "", active: false, scope: { project: "no-recall" }, scoped: true, updatedAt: now, createdAt: now },
  { id: "p1-s3", title: "零样本 NLI 邮件分类器压测", preview: "", active: false, scope: { project: "no-recall" }, scoped: true, updatedAt: now, createdAt: now },
  { id: "p2-s1", title: "插件边界 R12 静态门禁演进", preview: "", active: false, scope: { project: "arch" }, scoped: true, updatedAt: now, createdAt: now },
  { id: "rec-1", title: "西历2236 原文逐句精读与双关对照", preview: "", active: false, updatedAt: now, createdAt: now },
  { id: "rec-2", title: "家庭服务器软银东京节点出口排障", preview: "", active: false, status: "unread" as const, updatedAt: now, createdAt: now },
];

const projectGroups = [
  { id: "no-recall", label: "akashic-no-recall", detail: "记忆系统与端侧实验" },
  { id: "arch", label: "akashic-architecture", detail: "核心架构讨论" },
];

function App() {
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set(["project:no-recall"]));
  const headerSlotRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const railEntries: any[] = [];
    const disposeWeb = activatePrivacyWeb({
      ui: {
        inject(_mount: string, fn: any) {
          return fn({
            register(entry: any) {
              railEntries.push(entry);
              setShellRailActions(railEntries.map(({ id, order, label, iconSvg }) => ({ id, order, label, iconSvg })));
              return () => {};
            },
          });
        },
      },
      http: {} as any,
      data: {} as any,
    });
    const disconnectHost = connectHost({
      activateRailAction: (id: string) => railEntries.find((e) => e.id === id)?.onActivate(),
      reportSession: () => {},
      reportOverlay: () => {},
      requestIntent: async () => "unhandled",
    });
    let unmountSlot: (() => void) | void;
    if (headerSlotRef.current && privacySlotDefinition.slots["header.end"]) {
      unmountSlot = privacySlotDefinition.slots["header.end"].mount(headerSlotRef.current, {
        slot: "header.end",
        sessionId: "p1-s1",
        onInvalidate: () => () => {},
        reportError: () => {},
        onShown: () => () => {},
        capabilities: {},
        subscribe: () => () => {},
        query: async () => ({}),
        workbench: { accepts: () => false, open: async () => "unhandled" },
      });
    }
    (window as any).__disposePrivacyPlugin = () => {
      if (typeof unmountSlot === "function") unmountSlot();
      disposeWeb();
      disconnectHost();
    };
    return () => {
      if (typeof unmountSlot === "function") unmountSlot();
      disposeWeb();
      disconnectHost();
    };
  }, []);

  const toggleGroup = useCallback((key: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
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
    revision: 1,
    active: { dimension: "project", id: "no-recall", label: "akashic-no-recall" },
    reload: () => {},
    startChat: () => {},
  };

  const pinsState = {
    pins: [{ kind: "session" as const, id: "pin-1" }],
    sessions: [],
    ready: true,
    pending: false,
    error: "",
    reload: async () => {},
    setPinned: async () => {},
    expandedGroups: expanded,
    toggleGroup,
    expandGroup: () => {},
  };

  return (
    <TooltipProvider delayDuration={0}>
      <div className="akashic-chat" data-chat-width="1280">
        <main className="chat-shell" style={{ height: "100vh" }}>
          <div className="chat-shell-body" style={{ "--chat-rail-width": "15.5rem", height: "100%" } as any}>
            <DesktopSidebar
              surface="chat"
              sessions={initialSessions}
              activeSessionId="p1-s1"
              pendingSessionId=""
              chatReady={true}
              groups={groupsState as any}
              navigationPins={pinsState as any}
              onSelectSession={() => {}}
              onNewChat={() => {}}
              onDeleteSession={async () => {}}
              onRenameSession={async () => {}}
              deletedNotice={null}
            />
            <section className="chat-main" style={{ display: "flex", flexDirection: "column", flex: 1, minWidth: 0 }}>
              <header className="conversation-heading">
                <div className="conversation-heading__stack">
                  <h1 className="conversation-heading__title">
                    <span className="conversation-heading__title-text" title="开发机 Alma 与 Magpie 截屏打码调研">
                      开发机 Alma 与 Magpie 截屏打码调研
                    </span>
                  </h1>
                  <div className="header-meta" aria-label="会话信息">
                    <div className="header-meta__group">
                      <button type="button" className="header-meta__project">akashic-no-recall</button>
                      <span className="header-meta__path">
                        <button type="button" title="/home/huashen/coding/akashic-agent">
                          <span className="header-meta__icon"><Folder size={13} /></span>
                          <span>akashic-agent</span>
                        </button>
                      </span>
                    </div>
                  </div>
                </div>
                <div className="plugin-ui-slot" data-slot="header.end">
                  <div className="plugin-ui-host" data-plugin="privacy_mask@release" ref={headerSlotRef} />
                </div>
                <button type="button" className="conversation-heading__find" aria-label="在对话中查找">
                  <Search size={18} strokeWidth={1.75} />
                </button>
                <button type="button" className="conversation-heading__new" aria-label="新会话">
                  <MessageSquarePlus size={18} strokeWidth={1.75} />
                </button>
              </header>

              <div className="conversation-content" style={{ flex: 1, overflow: "auto", padding: "24px 36px", display: "flex", flexDirection: "column", gap: "18px" }}>
                <article className="user-message user-bubble" id="msg-user-1" style={{ alignSelf: "flex-end", maxWidth: "42rem", padding: "12px 16px", borderRadius: "12px", background: "var(--chat-chip)" }}>
                  <div className="chat-markdown" id="user-content-inner">
                    <p>
                      我的开发机上有 Alma 的源码，在 /mnt/data/source-code 文件夹下，帮我看一下它那个一键模糊内容来做截屏的设计，我们是插件化的系统需要保证高度正交化。
                    </p>
                  </div>
                </article>

                <article className="message-bubble agent-content" id="msg-assistant-1" style={{ maxWidth: "46rem", display: "flex", flexDirection: "column", gap: "10px" }}>
                  <div className="tool-step" id="tool-step-1" style={{ display: "inline-flex", alignItems: "center", gap: "8px", padding: "6px 10px", borderRadius: "8px", background: "var(--ak-paper-sheet)", color: "var(--ak-ink-secondary)", fontSize: "13px", width: "fit-content" }}>
                    <Terminal size={14} />
                    <span>ssh arch-dev 检查 /mnt/data/source-code/alma-0.4.467 与 /mnt/data/source-code/magpie</span>
                  </div>
                  <div className="chat-markdown" id="assistant-content-inner" style={{ display: "flex", flexDirection: "column", gap: "10px", lineHeight: 1.65 }}>
                    <p>
                      我已经把正交化的截屏隐私打码插件做好了：提供「导航与会话标题」和「全部内容模糊」两个正交开关，一键把侧栏列表化为错落圆角骨架条，并把对话全部内容整体高斯模糊。
                    </p>
                    <pre id="pre-code-block" style={{ padding: "12px 14px", borderRadius: "8px", background: "var(--ak-paper-sheet)", border: "1px solid var(--ak-rule-subtle)", fontFamily: "var(--ak-type-technical)", fontSize: "12.5px", overflowX: "auto" }}>
                      <code>{"// 双层正交属性驱动，零 DOM 扫描与零跨插件耦合\\nhtml[data-privacy-titles=\\"true\\"] { /* 侧栏与标题骨架条 */ }\\nhtml[data-privacy-content=\\"true\\"] { /* 对话全部内容模糊 */ }"}</code>
                    </pre>
                  </div>
                </article>
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

writeFileSync(
  work + "/chat-ui-shim.ts",
  `
import { exposeChatRuntime } from ${JSON.stringify(resolve(repo, "frontend/chat/src/chat-ui-runtime.tsx"))};
exposeChatRuntime();
const C = (window as any).__akashicRuntime.ChatUi;
export const { render, Notice, HeaderMarker, HeaderMarkerActions, MarqueeText, ChoiceMenu, copyText, ToolText, ToolData, formatDuration } = C;
`,
);

await build({
  configFile: resolve(repo, "frontend/chat/vite.harness.config.ts"),
  root: work,
  logLevel: "warn",
  css: { postcss: resolve(repo, "frontend/chat") },
  resolve: {
    alias: {
      "@": resolve(repo, "frontend/chat/src"),
      "@akashic/chat-ui-v1": work + "/chat-ui-shim.ts",
    },
  },
  build: { outDir: work + "/dist", emptyOutDir: true },
});

const server = await startDesktopFixtureServer(work + "/dist");
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH ?? "/usr/bin/chromium",
  headless: true,
});

try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 }, deviceScaleFactor: 2 });
  const errors = [];
  page.on("pageerror", (err) => errors.push(err.message));

  await page.goto(server.origin, { waitUntil: "networkidle" });
  await page.waitForSelector(".privacy-mask-marker .header-marker__trigger");
  await page.waitForTimeout(150);

  // 截图 1：开启打码前的默认态
  await page.screenshot({ path: resolve(root, "01-unmasked.png") });

  const beforeBaseline = await page.evaluate(() => {
    const firstSession = document.querySelector(".conversation-session").getBoundingClientRect();
    const firstGroup = document.querySelector(".navigation-group__open").getBoundingClientRect();
    return {
      sessionRect: { width: firstSession.width, height: firstSession.height },
      groupRect: { width: firstGroup.width, height: firstGroup.height },
    };
  });

  // 1. 点击侧栏底栏「截屏打码」按钮（shell.rail-actions.v1）一键开启全量打码（标题骨架条 + 全部内容模糊）
  const railBtn = page.locator(".chat-sidebar__footer .chat-sidebar__action", { hasText: "截屏打码" });
  assert.equal(await railBtn.count(), 1, "侧栏底栏应挂载 shell.rail-actions.v1 的截屏打码按钮");
  await railBtn.click();
  await page.waitForTimeout(200);

  const afterEnable = await page.evaluate(() => {
    const html = document.documentElement;
    const titleEl = document.querySelector(".conversation-session__title");
    const titleStyle = getComputedStyle(titleEl);
    const titleAfter = getComputedStyle(titleEl, "::after");
    const firstSession = document.querySelector(".conversation-session").getBoundingClientRect();
    const firstGroup = document.querySelector(".navigation-group__open").getBoundingClientRect();

    const userContentFilter = getComputedStyle(document.getElementById("user-content-inner")).filter;
    const assistantContentFilter = getComputedStyle(document.getElementById("assistant-content-inner")).filter;
    const toolStepFilter = getComputedStyle(document.getElementById("tool-step-1")).filter;
    const pathMetaFilter = getComputedStyle(document.querySelector(".header-meta__path > button")).filter;
    const headingTitle = document.querySelector(".conversation-heading__title-text");

    return {
      maskAttr: html.getAttribute("data-privacy-mask"),
      titlesAttr: html.getAttribute("data-privacy-titles"),
      contentAttr: html.getAttribute("data-privacy-content"),
      titleColor: titleStyle.color,
      afterWidth: parseFloat(titleAfter.width),
      sessionRect: { width: firstSession.width, height: firstSession.height },
      groupRect: { width: firstGroup.width, height: firstGroup.height },
      userContentFilter,
      assistantContentFilter,
      toolStepFilter,
      pathMetaFilter,
      headingHasTitleAttr: headingTitle.hasAttribute("title"),
      headingSavedRawTitle: headingTitle.getAttribute("data-privacy-raw-title"),
    };
  });

  assert.equal(afterEnable.maskAttr, "true");
  assert.equal(afterEnable.titlesAttr, "true");
  assert.equal(afterEnable.contentAttr, "true");
  assert.equal(afterEnable.titleColor, "rgba(0, 0, 0, 0)", "侧栏标题文字应透明并由 ::after 骨架条替代");
  assert(afterEnable.afterWidth > 30, `侧栏标题骨架条宽度应大于 30px，实测 ${afterEnable.afterWidth}`);
  assert.deepEqual(afterEnable.sessionRect, beforeBaseline.sessionRect, "开启打码前后侧栏会话行尺寸必须零变化");
  assert.deepEqual(afterEnable.groupRect, beforeBaseline.groupRect, "开启打码前后项目行尺寸必须零变化");
  assert.match(afterEnable.userContentFilter, /blur\(6px\)/, "用户气泡内部内容应整体高斯模糊");
  assert.match(afterEnable.assistantContentFilter, /blur\(6px\)/, "助手回复与代码块应整体高斯模糊");
  assert.match(afterEnable.toolStepFilter, /blur\(6px\)/, "工具执行步骤应整体高斯模糊");
  assert.match(afterEnable.pathMetaFilter, /blur\(6px\)/, "顶栏目录路径应高斯模糊");
  assert.equal(afterEnable.headingHasTitleAttr, false, "打码期间应剥离会泄漏标题的原生 title 提示");
  assert.equal(afterEnable.headingSavedRawTitle, "开发机 Alma 与 Magpie 截屏打码调研");

  // 2. 点开顶栏「已打码」标记，验证只有两个正交开关（「导航与会话标题」「全部内容模糊」）并截图
  const headerTrigger = page.locator(".privacy-mask-marker .header-marker__trigger");
  await headerTrigger.click();
  await page.waitForSelector(".privacy-mask-panel");
  await page.waitForTimeout(140);

  const switches = await page.locator(".privacy-mask-scope .privacy-mask-scope__label").allInnerTexts();
  assert.deepEqual(switches, ["导航与会话标题", "全部内容模糊"], "面板应严格只包含「导航与会话标题」与「全部内容模糊」两个正交开关");
  await page.screenshot({ path: resolve(root, "02-full-masked-popover.png") });

  // 3. 测试正交独立开关：关闭「全部内容模糊」、仅保留「导航与会话标题」（晒对话内容但隐藏侧栏项目与历史会话）
  const contentSwitch = page.locator(".privacy-mask-scope", { hasText: "全部内容模糊" });
  await contentSwitch.click();
  await page.waitForTimeout(200);

  const titlesOnlyState = await page.evaluate(() => ({
    titlesAttr: document.documentElement.getAttribute("data-privacy-titles"),
    contentAttr: document.documentElement.getAttribute("data-privacy-content"),
    userContentFilter: getComputedStyle(document.getElementById("user-content-inner")).filter,
    titleColor: getComputedStyle(document.querySelector(".conversation-session__title")).color,
  }));
  assert.equal(titlesOnlyState.titlesAttr, "true");
  assert.equal(titlesOnlyState.contentAttr, null);
  assert.equal(titlesOnlyState.userContentFilter, "none", "关闭全部内容模糊后，对话正文应恢复清晰");
  assert.equal(titlesOnlyState.titleColor, "rgba(0, 0, 0, 0)", "侧栏标题应继续保持骨架条打码");
  await page.screenshot({ path: resolve(root, "03-titles-only-mode.png") });

  // 重新开启「全部内容模糊」并切换深色模式截图
  await contentSwitch.click();
  await page.waitForTimeout(200);
  await page.evaluate(() => window.__setTheme("dark"));
  await page.waitForTimeout(150);
  await page.screenshot({ path: resolve(root, "04-dark-full-masked.png") });
  await page.evaluate(() => window.__setTheme("light"));
  await page.waitForTimeout(120);

  // 4. 验证 Alt+M 快捷键一键恢复正常显示
  await page.keyboard.press("Escape");
  await page.waitForTimeout(60);
  await page.keyboard.press("Alt+KeyM");
  await page.waitForTimeout(200);

  const afterDisable = await page.evaluate(() => {
    const html = document.documentElement;
    const headingTitle = document.querySelector(".conversation-heading__title-text");
    const titleEl = document.querySelector(".conversation-session__title");
    return {
      maskAttr: html.getAttribute("data-privacy-mask"),
      restoredTitleAttr: headingTitle.getAttribute("title"),
      titleColor: getComputedStyle(titleEl).color,
      userContentFilter: getComputedStyle(document.getElementById("user-content-inner")).filter,
    };
  });
  assert.equal(afterDisable.maskAttr, null);
  assert.equal(afterDisable.restoredTitleAttr, "开发机 Alma 与 Magpie 截屏打码调研");
  assert.notEqual(afterDisable.titleColor, "rgba(0, 0, 0, 0)");
  assert.equal(afterDisable.userContentFilter, "none");

  // 5. 录制完整交互动效 GIF（默认态 → 一键开启全量打码 → 展开面板切换仅侧栏打码 → 再开启全量打码 → Alt+M 恢复）
  const framesDir = resolve(root, "gif-frames");
  mkdirSync(framesDir, { recursive: true });
  let frameIndex = 0;
  const snapFrames = async (count, intervalMs) => {
    for (let i = 0; i < count; i++) {
      const file = resolve(framesDir, `frame-${String(frameIndex++).padStart(4, "0")}.png`);
      await page.screenshot({ path: file });
      if (intervalMs > 0) await page.waitForTimeout(intervalMs);
    }
  };

  await snapFrames(5, 80);
  await railBtn.click();
  await snapFrames(5, 70);
  await headerTrigger.click();
  await snapFrames(6, 80);
  await contentSwitch.click();
  await snapFrames(6, 80);
  await contentSwitch.click();
  await snapFrames(5, 70);
  await page.keyboard.press("Alt+KeyM");
  await snapFrames(5, 80);

  const gifPath = resolve(root, "privacy-mask-demo.gif");
  execFileSync("ffmpeg", [
    "-y",
    "-framerate",
    "10",
    "-i",
    resolve(framesDir, "frame-%04d.png"),
    "-vf",
    "scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
    gifPath,
  ]);

  // 6. 验证插件卸载（release）后的完全正交零残留
  await page.keyboard.press("Alt+KeyM");
  await page.waitForTimeout(100);
  const afterDispose = await page.evaluate(() => {
    window.__disposePrivacyPlugin();
    const html = document.documentElement;
    return {
      maskAttr: html.getAttribute("data-privacy-mask"),
      titlesAttr: html.getAttribute("data-privacy-titles"),
      contentAttr: html.getAttribute("data-privacy-content"),
      globalRef: typeof window.__akashicPrivacyMask,
    };
  });
  assert.equal(afterDispose.maskAttr, null);
  assert.equal(afterDispose.titlesAttr, null);
  assert.equal(afterDispose.contentAttr, null);
  assert.equal(afterDispose.globalRef, "undefined", "插件卸载后必须彻底释放全局单例与全部 DOM 属性");
  assert.deepEqual(errors, [], `页面运行期不得出现未捕获异常: ${errors.join("; ")}`);

  const summary = {
    ok: true,
    evidenceDir: root,
    switches,
    geometryPreserved: true,
    gifPath,
  };
  writeFileSync(resolve(root, "summary.json"), JSON.stringify(summary, null, 2));
  console.log(JSON.stringify(summary, null, 2));
} finally {
  await browser.close();
  await server.close();
}
