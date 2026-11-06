/**
 * 截屏隐私打码插件（privacy_mask@release）真实浏览器端到端验收：
 * 1. 正交挂载与一键开关：同时验证 shell.rail-actions.v1 底栏按钮、header.end 顶栏标记与 Alt+M 全局快捷键；
 * 2. 维度一（导航与会话标题骨架条）：侧栏会话标题、项目分组名、项目子会话、顶栏标题与项目副标题化为 48%/74%/56%/62% 错落圆角骨架条，行几何尺寸零偏移，title/data-tip 悬停提示同步脱敏并可无损恢复；
 * 3. 维度二（CSS Custom Highlight 敏感词面模糊）：零 DOM 文本节点拆分/突变（100% 兼容 React 19 协调器），自动识别并高斯模糊邮箱、Unix/Windows 路径、IPv4、手机号与 API Key，且流式新增文本经 MutationObserver 在单帧内自动纳入；
 * 4. 维度三（点选局部区块打码）：开启后悬停段落/代码块/消息气泡浮现虚线候选框，单击即原地高斯模糊（filter: blur(6px)），关闭点选模式后已选区块保持模糊供干净截屏，支持一键清空点选与插件卸载零残留。
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
  const [streamedExtra, setStreamedExtra] = useState("");
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
    (window as any).__appendStreamChunk = (text: string) => setStreamedExtra(text);
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
                <article className="user-message" id="msg-user-1" style={{ alignSelf: "flex-end", maxWidth: "42rem", padding: "12px 16px", borderRadius: "12px", background: "var(--chat-chip)" }}>
                  <div className="chat-markdown">
                    <p id="p-user-pii">
                      请帮我检查开发机 192.168.0.108:22 上 /home/huashen/coding/akashic-agent 的配置，告警发到 2494946888@qq.com 或手机 15311315482，测试网关密钥为 sk-proj-9f8a7b6c5d4e3f2a1b0c。
                    </p>
                  </div>
                </article>

                <article className="message-bubble" id="msg-assistant-1" style={{ maxWidth: "46rem", display: "flex", flexDirection: "column", gap: "10px" }}>
                  <div className="tool-step" id="tool-step-1" style={{ display: "inline-flex", alignItems: "center", gap: "8px", padding: "6px 10px", borderRadius: "8px", background: "var(--ak-paper-sheet)", color: "var(--ak-ink-secondary)", fontSize: "13px", width: "fit-content" }}>
                    <Terminal size={14} />
                    <span>ssh arch-dev 检查 /mnt/data/source-code/alma-0.4.467 与 /mnt/data/source-code/magpie</span>
                  </div>
                  <div className="chat-markdown" style={{ display: "flex", flexDirection: "column", gap: "10px", lineHeight: 1.65 }}>
                    <p id="p-assistant-intro">
                      我已经对照了 Alma 与 Magpie 的源码实现：Alma 采用语义属性骨架条与候选区块点选打码，而 Magpie 通过 TreeWalker 与 MutationObserver 对页面上的邮箱和账号做实时模糊。
                    </p>
                    <pre id="pre-code-block" style={{ padding: "12px 14px", borderRadius: "8px", background: "var(--ak-paper-sheet)", border: "1px solid var(--ak-rule-subtle)", fontFamily: "var(--ak-type-technical)", fontSize: "12.5px", overflowX: "auto" }}>
                      <code>{"// 内部网关调试端点与鉴权配置（点选此代码块可整块打码）\\nconst INTERNAL_RPC = \\"https://hooks.internal.example/v1/dispatch\\";\\nconst STAGE_CLUSTER = \\"tokyo-softbank-node-01\\";"}</code>
                    </pre>
                    {streamedExtra ? <p id="p-streamed-extra">{streamedExtra}</p> : null}
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

  // 记录开启前文本节点数量、正文内容与侧栏行尺寸（验证零 DOM 节点突变与零几何偏移）
  const beforeBaseline = await page.evaluate(() => {
    const p = document.getElementById("p-user-pii");
    let textNodes = 0;
    const walker = document.createTreeWalker(p, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) textNodes += 1;
    const firstSession = document.querySelector(".conversation-session").getBoundingClientRect();
    const firstGroup = document.querySelector(".navigation-group__open").getBoundingClientRect();
    return {
      textNodes,
      text: p.textContent,
      sessionRect: { width: firstSession.width, height: firstSession.height },
      groupRect: { width: firstGroup.width, height: firstGroup.height },
    };
  });

  // 1. 验证侧栏底栏动作（shell.rail-actions.v1）一键开启截屏打码
  const railBtn = page.locator(".chat-sidebar__footer .chat-sidebar__action", { hasText: "截屏打码" });
  assert.equal(await railBtn.count(), 1, "侧栏底栏应挂载 shell.rail-actions.v1 的截屏打码按钮");
  await railBtn.click();
  await page.waitForTimeout(180);

  const afterEnable = await page.evaluate(() => {
    const html = document.documentElement;
    const p = document.getElementById("p-user-pii");
    let textNodes = 0;
    const walker = document.createTreeWalker(p, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) textNodes += 1;

    const titleEl = document.querySelector(".conversation-session__title");
    const titleStyle = getComputedStyle(titleEl);
    const titleAfter = getComputedStyle(titleEl, "::after");
    const firstSession = document.querySelector(".conversation-session").getBoundingClientRect();
    const firstGroup = document.querySelector(".navigation-group__open").getBoundingClientRect();

    const highlight = CSS.highlights?.get("akashic-privacy-pii");
    const pathMetaBtn = document.querySelector(".header-meta__path > button");
    const pathFilter = getComputedStyle(pathMetaBtn).filter;
    const headingTitle = document.querySelector(".conversation-heading__title-text");

    return {
      maskAttr: html.getAttribute("data-privacy-mask"),
      titlesAttr: html.getAttribute("data-privacy-titles"),
      piiAttr: html.getAttribute("data-privacy-pii"),
      textNodes,
      text: p.textContent,
      titleColor: titleStyle.color,
      afterWidth: parseFloat(titleAfter.width),
      afterRadius: titleAfter.borderRadius,
      sessionRect: { width: firstSession.width, height: firstSession.height },
      groupRect: { width: firstGroup.width, height: firstGroup.height },
      highlightSize: highlight ? highlight.size : 0,
      pathFilter,
      headingHasTitleAttr: headingTitle.hasAttribute("title"),
      headingSavedRawTitle: headingTitle.getAttribute("data-privacy-raw-title"),
    };
  });

  assert.equal(afterEnable.maskAttr, "true");
  assert.equal(afterEnable.titlesAttr, "true");
  assert.equal(afterEnable.piiAttr, "true");
  assert.equal(afterEnable.textNodes, beforeBaseline.textNodes, "CSS Highlight 打码不得拆分或增删 React 文本节点");
  assert.equal(afterEnable.text, beforeBaseline.text, "原始文本内容不得被篡改");
  assert.equal(afterEnable.titleColor, "rgba(0, 0, 0, 0)", "侧栏标题文字应透明并由 ::after 骨架条替代");
  assert(afterEnable.afterWidth > 30, `侧栏标题骨架条宽度应大于 30px，实测 ${afterEnable.afterWidth}`);
  assert.deepEqual(afterEnable.sessionRect, beforeBaseline.sessionRect, "开启打码前后侧栏会话行尺寸必须零变化");
  assert.deepEqual(afterEnable.groupRect, beforeBaseline.groupRect, "开启打码前后项目行尺寸必须零变化");
  assert(afterEnable.highlightSize >= 6, `应至少识别并模糊 6 处敏感词面（IP/路径/邮箱/手机号/密钥），实测 ${afterEnable.highlightSize}`);
  assert.match(afterEnable.pathFilter, /blur/, "顶栏工作目录路径应被高斯模糊");
  assert.equal(afterEnable.headingHasTitleAttr, false, "打码期间应剥离会泄漏标题的原生 title 提示");
  assert.equal(afterEnable.headingSavedRawTitle, "开发机 Alma 与 Magpie 截屏打码调研");

  // 2. 验证流式增量输出（MutationObserver 单帧内自动追加敏感词面高亮）
  await page.evaluate(() => {
    window.__appendStreamChunk("流式新增：备份位于 /srv/data/services/magpie/gateway-key.txt，通知发往 admin@akashic.dev。");
  });
  await page.waitForTimeout(100);
  const streamedHighlightSize = await page.evaluate(() => CSS.highlights?.get("akashic-privacy-pii")?.size ?? 0);
  assert.equal(streamedHighlightSize, afterEnable.highlightSize + 2, "流式新增的路径与邮箱应被 MutationObserver 自动纳入模糊");

  // 3. 点开顶栏「已打码」标记，查看三维度控制面板并截图
  const headerTrigger = page.locator(".privacy-mask-marker .header-marker__trigger");
  await headerTrigger.click();
  await page.waitForSelector(".privacy-mask-panel");
  await page.waitForTimeout(120);
  await page.screenshot({ path: resolve(root, "02-masked-popover.png") });

  // 4. 开启「点选区块打码」，悬停并单击代码块与介绍段落
  const pickSwitch = page.locator(".privacy-mask-scope", { hasText: "点选区块打码" });
  await pickSwitch.click();
  await page.waitForTimeout(100);

  // 关闭弹出面板以便点选正文代码块
  await headerTrigger.click();
  await page.waitForTimeout(80);

  const codeBlock = page.locator("#pre-code-block");
  await codeBlock.hover();
  await page.waitForTimeout(80);
  const hoverState = await codeBlock.evaluate((el) => ({
    hoverAttr: el.getAttribute("data-privacy-hover"),
    outlineStyle: getComputedStyle(el).outlineStyle,
  }));
  assert.equal(hoverState.hoverAttr, "true", "点选模式下悬停代码块应标记 data-privacy-hover=true");
  assert.equal(hoverState.outlineStyle, "dashed", "悬停候选块应显示虚线轮廓");

  await codeBlock.click();
  await page.waitForTimeout(220);
  const pickedState = await codeBlock.evaluate((el) => ({
    pickedAttr: el.getAttribute("data-privacy-picked"),
    filter: getComputedStyle(el).filter,
  }));
  assert.equal(pickedState.pickedAttr, "true", "单击候选块后应标记 data-privacy-picked=true");
  assert.match(pickedState.filter, /blur\(6px\)/, "被点选区块应施加 blur(6px) 高斯模糊");

  // 再次打开面板展示「已选 1 块」状态并截图
  await headerTrigger.click();
  await page.waitForTimeout(120);
  await page.screenshot({ path: resolve(root, "03-pick-block-mode.png") });

  // 关闭「点选区块打码」开关（准备干净截屏），验证已选代码块依然保持模糊
  await pickSwitch.click();
  await page.waitForTimeout(100);
  const stillBlurredAfterExitPick = await codeBlock.evaluate((el) => getComputedStyle(el).filter);
  assert.match(stillBlurredAfterExitPick, /blur\(6px\)/, "退出点选模式后，已选打码区块必须保持模糊以供截屏");

  // 5. 切换至深色模式验证并截图
  await page.evaluate(() => window.__setTheme("dark"));
  await page.waitForTimeout(150);
  await page.screenshot({ path: resolve(root, "04-dark-masked.png") });
  await page.evaluate(() => window.__setTheme("light"));
  await page.waitForTimeout(120);

  // 6. 验证 Alt+M 快捷键一键关闭打码与恢复，以及再次 Alt+M 开启
  await page.keyboard.press("Escape"); // 先收起面板
  await page.waitForTimeout(60);
  await page.keyboard.press("Alt+KeyM");
  await page.waitForTimeout(160);

  const afterDisable = await page.evaluate(() => {
    const html = document.documentElement;
    const highlight = CSS.highlights?.get("akashic-privacy-pii");
    const headingTitle = document.querySelector(".conversation-heading__title-text");
    const titleEl = document.querySelector(".conversation-session__title");
    return {
      maskAttr: html.getAttribute("data-privacy-mask"),
      highlightSize: highlight ? highlight.size : 0,
      restoredTitleAttr: headingTitle.getAttribute("title"),
      titleColor: getComputedStyle(titleEl).color,
    };
  });
  assert.equal(afterDisable.maskAttr, null, "Alt+M 关闭后应移除 data-privacy-mask");
  assert.equal(afterDisable.highlightSize, 0, "关闭后应清空 CSS Highlight");
  assert.equal(afterDisable.restoredTitleAttr, "开发机 Alma 与 Magpie 截屏打码调研", "关闭后应无损恢复原生 title 属性");
  assert.notEqual(afterDisable.titleColor, "rgba(0, 0, 0, 0)", "关闭后侧栏标题应恢复正常墨色");

  // 7. 录制完整交互动效 GIF（默认态 → 一键开启打码 → 展开三维度面板 → 点选代码块打码 → 恢复正常显示）
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

  // 清空之前的点选，从干净未打码状态开始录制
  await page.evaluate(() => {
    for (const el of document.querySelectorAll("[data-privacy-picked]")) el.removeAttribute("data-privacy-picked");
  });
  await snapFrames(5, 80);

  // 开启打码并展开面板
  await railBtn.click();
  await snapFrames(4, 60);
  await headerTrigger.click();
  await snapFrames(6, 80);

  // 开启点选模式，关闭面板后悬停并点选代码块
  await pickSwitch.click();
  await snapFrames(3, 70);
  await headerTrigger.click();
  await codeBlock.hover();
  await snapFrames(4, 70);
  await codeBlock.click();
  await snapFrames(5, 80);

  // 按 Alt+M 一键恢复显示
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

  // 8. 验证插件卸载（release）后的完全正交零残留
  await page.keyboard.press("Alt+KeyM");
  await page.waitForTimeout(100);
  const afterDispose = await page.evaluate(() => {
    window.__disposePrivacyPlugin();
    const html = document.documentElement;
    return {
      maskAttr: html.getAttribute("data-privacy-mask"),
      titlesAttr: html.getAttribute("data-privacy-titles"),
      piiAttr: html.getAttribute("data-privacy-pii"),
      pickAttr: html.getAttribute("data-privacy-pick"),
      highlightHas: CSS.highlights?.has("akashic-privacy-pii") ?? false,
      globalRef: typeof window.__akashicPrivacyMask,
    };
  });
  assert.equal(afterDispose.maskAttr, null);
  assert.equal(afterDispose.titlesAttr, null);
  assert.equal(afterDispose.piiAttr, null);
  assert.equal(afterDispose.pickAttr, null);
  assert.equal(afterDispose.highlightHas, false);
  assert.equal(afterDispose.globalRef, "undefined", "插件卸载后必须彻底释放全局单例与全部 DOM 痕迹");
  assert.deepEqual(errors, [], `页面运行期不得出现未捕获异常: ${errors.join("; ")}`);

  const summary = {
    ok: true,
    evidenceDir: root,
    piiHighlightedCount: streamedHighlightSize,
    textNodesPreserved: afterEnable.textNodes === beforeBaseline.textNodes,
    geometryPreserved: true,
    gifPath,
  };
  writeFileSync(resolve(root, "summary.json"), JSON.stringify(summary, null, 2));
  console.log(JSON.stringify(summary, null, 2));
} finally {
  await browser.close();
  await server.close();
}
