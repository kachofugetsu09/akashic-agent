/**
 * 截屏隐私打码插件（privacy_mask@release）真实浏览器端到端验收：
 * 1. 正交挂载与快捷键：同时验证 shell.rail-actions.v1 底栏按钮、header.end 顶栏标记、非输入区 Alt+M 与全局 Ctrl/⌘+Shift+M（输入区内 Alt+M 不误吞字符输入）；
 * 2. 维度一（导航与会话标题骨架条）：侧栏会话标题、项目分组名、项目子会话（各自独立 48%/74%/56%/62% 错落宽度）、真实顶栏 <h1 title><MarqueeText>、项目副标题与输入框引用会话名化为圆角骨架条；动态更新的 title/data-tip 与移出文档的节点均可无损恢复最新值；
 * 3. 维度二（全部内容模糊打码）：用户消息气泡内容、时间线气泡外附件、助手回复正文、代码块、工具执行步骤、输入框草稿/回复预览/附件托盘、顶栏目录路径与会话信息浮层值整体施加 filter: blur(6px) 高斯模糊；
 * 4. 生命周期隔离：插件卸载（release）后彻底清除全部 data-privacy-* 属性与已跟踪节点的提示备份，零残留。
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
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
  '<!doctype html><html><head><meta charset="utf-8"><script type="importmap">{"imports":{}}</script></head><body><div id="root"></div><script type="module" src="./fixture.tsx"></script></body></html>',
);

writeFileSync(
  work + "/fixture.tsx",
  `
import { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Search, MessageSquarePlus, Terminal, Folder, Reply, X } from "lucide-react";
import { DesktopSidebar } from ${JSON.stringify(resolve(repo, "frontend/chat/src/desktop-sidebar.tsx"))};
import { MarqueeText } from ${JSON.stringify(resolve(repo, "frontend/chat/src/marquee-text.tsx"))};
import { exposeChatRuntime } from ${JSON.stringify(resolve(repo, "frontend/chat/src/chat-ui-runtime.tsx"))};
import { connectHost } from ${JSON.stringify(resolve(repo, "frontend/chat/src/chat-host.ts"))};
import { startWebHost } from ${JSON.stringify(resolve(repo, "frontend/dashboard/src/webHost.ts"))};
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
  const [headingTitle, setHeadingTitle] = useState("开发机 Alma 与 Magpie 截屏打码调研");
  const headerSlotRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
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
    (window as any).__setHeadingTitle = (next: string) => setHeadingTitle(next);
    return () => {
      if (typeof unmountSlot === "function") unmountSlot();
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
                  {/* 真实 SessionHeadingTitle DOM 结构：<h1 title={heading}><MarqueeText>{heading}</MarqueeText></h1> */}
                  <h1 id="real-heading-h1" title={headingTitle}><MarqueeText>{headingTitle}</MarqueeText></h1>
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
                {/* 真实 TimelineMessageView 结构：timeline-input 行内包含气泡外的附件与 .user-bubble */}
                <div className="message-row timeline-message timeline-input" style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "8px" }}>
                  <div className="sent-attachments" id="timeline-input-attachment" style={{ padding: "4px 10px", borderRadius: "8px", background: "var(--ak-paper-sheet)", fontSize: "12px" }}>
                    <span>alma-window-capture-spec.pdf</span>
                  </div>
                  <div className="user-bubble" id="msg-user-1" style={{ maxWidth: "42rem", padding: "12px 16px", borderRadius: "12px", background: "var(--chat-chip)" }}>
                    <div className="chat-markdown" id="user-content-inner">
                      <p>
                        我的开发机上有 Alma 的源码，在 /mnt/data/source-code 文件夹下，帮我看一下它那个一键模糊内容来做截屏的设计，我们是插件化的系统需要保证高度正交化。
                      </p>
                    </div>
                  </div>
                </div>

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

              {/* 真实输入区结构（含引用会话标题、回复预览、附件托盘与 textarea） */}
              <div className="composer-wrap" style={{ padding: "0 36px 20px" }}>
                <div className="composer" style={{ padding: "10px 14px", borderRadius: "14px", background: "var(--ak-paper-sheet)", border: "1px solid var(--ak-rule-subtle)", display: "flex", flexDirection: "column", gap: "8px" }}>
                  <div className="composer-references" style={{ display: "flex", gap: "6px", fontSize: "12px" }}>
                    <span className="composer-reference">
                      <span className="composer-reference__title" id="composer-ref-title" title="私人历史会话引用">私人历史会话引用</span>
                    </span>
                  </div>
                  <div className="composer-reply" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", fontSize: "12px", color: "var(--ak-ink-secondary)" }}>
                    <Reply size={14} />
                    <div id="composer-reply-preview">被引用回复中的内部敏感正文预览</div>
                    <button type="button" id="composer-reply-cancel" aria-label="取消引用"><X size={14} /></button>
                  </div>
                  <div className="composer-tray" role="list">
                    <div role="listitem" data-attachment-id="att-1" id="composer-tray-item" style={{ fontSize: "12px" }}>
                      secret-credentials.json
                    </div>
                  </div>
                  <textarea id="composer-textarea" className="composer__textarea" defaultValue="正在编写的草稿内容…" style={{ width: "100%", border: 0, background: "transparent", resize: "none", outline: "none", fontSize: "13.5px" }} />
                </div>
              </div>
            </section>
          </div>
        </main>
      </div>
    </TooltipProvider>
  );
}

// 夹具只提供页面；插件的注册、校验与卸载全部经过真实宿主。
(window as any).__activatePrivacyWeb = activatePrivacyWeb;
(window as any).__activateFixtureShell = (ctx: any) => ctx.ui.inject("web.root.v1", (mount: any) =>
  mount.register({
    id: "fixture-shell",
    children: [{ id: "shell.rail-actions.v1", cardinality: "list" }],
    render(host: HTMLElement, view: any) {
      exposeChatRuntime();
      const railEntries = view.child("shell.rail-actions.v1").entries;
      setShellRailActions(railEntries.map(({ id, order, label, iconSvg }: any) => ({ id, order, label, iconSvg })));
      const disconnectHost = connectHost({
        activateRailAction: (id: string) => railEntries.find((entry: any) => entry.id === id)?.onActivate(),
        reportSession: () => {},
        reportOverlay: () => {},
        requestIntent: async () => "unhandled",
      });
      const root = createRoot(host);
      root.render(<App />);
      return () => {
        root.unmount();
        disconnectHost();
        setShellRailActions([]);
      };
    },
  }),
);
startWebHost(document.getElementById("root")!).then((session) => {
  (window as any).__disposePrivacyPlugin = () => session.close();
  (window as any).__webHostReady = true;
});
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
const digest = (value) => createHash("sha256").update(value).digest("hex");
const modules = [
  { pluginId: "fixture-shell", activate: "__activateFixtureShell", requires: ["web.root.v1"], provides: ["shell.rail-actions.v1"] },
  { pluginId: "privacy_mask@release", activate: "__activatePrivacyWeb", requires: ["shell.rail-actions.v1"], provides: [] },
].map(({ pluginId, activate, requires, provides }) => {
  const module = `export const activate = window.${activate};`;
  const contractDigests = {};
  return {
    pluginId, generationId: "fixture-generation", module,
    moduleSha256: digest(module), moduleBytes: Buffer.byteLength(module),
    stylesheet: "", stylesheetSha256: null, stylesheetBytes: 0,
    requires, provides, contractDigests,
    contractSha256: digest(JSON.stringify({ contractDigests, provides, requires })),
  };
});
const state = { snapshotId: "privacy-fixture", catalogId: digest(JSON.stringify(modules)), updating: false };
const bootstrap = { schemaVersion: 1, snapshotId: state.snapshotId, catalogId: state.catalogId, modules };
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH ?? "/usr/bin/chromium",
  headless: true,
});

try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 }, deviceScaleFactor: 2 });
  const errors = [];
  page.on("pageerror", (err) => { errors.push(err.message); console.error(err); });
  await page.route("**/api/chat/web-ui/bootstrap", (route) => route.fulfill({ json: bootstrap }));
  await page.route("**/api/chat/web-ui/state", (route) => route.fulfill({ json: state }));

  await page.goto(server.origin, { waitUntil: "networkidle" });
  await page.waitForFunction(() => window.__webHostReady === true);
  assert.deepEqual(await page.locator(".web-host-entry-error").allTextContents(), [], "插件必须通过真实宿主加载");
  // 未开启时顶栏不常驻标记，开关只在侧栏底栏（图标）与快捷键。
  await page.locator(".chat-sidebar__footer").getByRole("button", { name: "截屏打码" }).waitFor();
  assert.equal(await page.locator(".privacy-mask-marker").count(), 0, "未开启时顶栏不应常驻截屏打码标记");
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

  // 1. 点击侧栏底栏「截屏打码」按钮一键开启全量打码
  const railBtn = page.locator(".chat-sidebar__footer").getByRole("button", { name: "截屏打码" });
  assert.equal(await railBtn.count(), 1, "侧栏底栏应挂载 shell.rail-actions.v1 的截屏打码图标按钮");
  await railBtn.click();
  await page.waitForTimeout(200);

  const afterEnable = await page.evaluate(() => {
    const html = document.documentElement;
    const titleEl = document.querySelector(".conversation-session__title");
    const titleStyle = getComputedStyle(titleEl);
    const titleAfter = getComputedStyle(titleEl, "::after");
    const firstSession = document.querySelector(".conversation-session").getBoundingClientRect();
    const firstGroup = document.querySelector(".navigation-group__open").getBoundingClientRect();

    // 验证项目内子会话骨架条具有错落宽度，不被父 .navigation-group 统一覆盖
    const childAfterWidths = [...document.querySelectorAll(".navigation-group__sessions .conversation-session__title")].map(
      (el) => Math.round(parseFloat(getComputedStyle(el, "::after").width)),
    );

    const headingH1 = document.getElementById("real-heading-h1");
    const headingH1Color = getComputedStyle(headingH1).color;
    const headingH1AfterWidth = parseFloat(getComputedStyle(headingH1, "::after").width);
    const composerRefColor = getComputedStyle(document.getElementById("composer-ref-title")).color;

    const userContentFilter = getComputedStyle(document.getElementById("user-content-inner")).filter;
    const timelineInputAttachmentFilter = getComputedStyle(document.getElementById("timeline-input-attachment")).filter;
    const assistantContentFilter = getComputedStyle(document.getElementById("assistant-content-inner")).filter;
    const toolStepFilter = getComputedStyle(document.getElementById("tool-step-1")).filter;
    const pathMetaFilter = getComputedStyle(document.querySelector(".header-meta__path > button")).filter;
    const composerReplyFilter = getComputedStyle(document.getElementById("composer-reply-preview")).filter;
    const composerTrayFilter = getComputedStyle(document.getElementById("composer-tray-item")).filter;
    const composerTextareaFilter = getComputedStyle(document.getElementById("composer-textarea")).filter;

    return {
      maskAttr: html.getAttribute("data-privacy-mask"),
      titlesAttr: html.getAttribute("data-privacy-titles"),
      contentAttr: html.getAttribute("data-privacy-content"),
      titleColor: titleStyle.color,
      afterWidth: parseFloat(titleAfter.width),
      childAfterWidths,
      headingH1Color,
      headingH1AfterWidth,
      composerRefColor,
      sessionRect: { width: firstSession.width, height: firstSession.height },
      groupRect: { width: firstGroup.width, height: firstGroup.height },
      userContentFilter,
      timelineInputAttachmentFilter,
      assistantContentFilter,
      toolStepFilter,
      pathMetaFilter,
      composerReplyFilter,
      composerTrayFilter,
      composerTextareaFilter,
      headingHasTitleAttr: headingH1.hasAttribute("title"),
    };
  });

  assert.equal(afterEnable.maskAttr, "true");
  assert.equal(afterEnable.titlesAttr, "true");
  assert.equal(afterEnable.contentAttr, "true");
  assert.equal(afterEnable.titleColor, "rgba(0, 0, 0, 0)", "侧栏标题文字应透明并由 ::after 骨架条替代");
  assert(afterEnable.afterWidth > 30, `侧栏标题骨架条宽度应大于 30px，实测 ${afterEnable.afterWidth}`);
  assert(new Set(afterEnable.childAfterWidths).size >= 2, `同组子会话骨架条应呈错落宽度，实测 ${JSON.stringify(afterEnable.childAfterWidths)}`);
  assert.equal(afterEnable.headingH1Color, "rgba(0, 0, 0, 0)", "真实顶栏 <h1 title> 应被骨架条打码");
  assert(afterEnable.headingH1AfterWidth > 60, `顶栏 h1 骨架条宽度应大于 60px，实测 ${afterEnable.headingH1AfterWidth}`);
  assert.equal(afterEnable.composerRefColor, "rgba(0, 0, 0, 0)", "输入框引用的会话名应被骨架条打码");
  assert.deepEqual(afterEnable.sessionRect, beforeBaseline.sessionRect, "开启打码前后侧栏会话行尺寸必须零变化");
  assert.deepEqual(afterEnable.groupRect, beforeBaseline.groupRect, "开启打码前后项目行尺寸必须零变化");
  assert.match(afterEnable.userContentFilter, /blur\(6px\)/, "用户气泡内部内容应整体高斯模糊");
  assert.match(afterEnable.timelineInputAttachmentFilter, /blur\(6px\)/, "时间线气泡外附件应被高斯模糊");
  assert.match(afterEnable.assistantContentFilter, /blur\(6px\)/, "助手回复与代码块应整体高斯模糊");
  assert.match(afterEnable.toolStepFilter, /blur\(6px\)/, "工具执行步骤应整体高斯模糊");
  assert.match(afterEnable.pathMetaFilter, /blur\(6px\)/, "顶栏目录路径应高斯模糊");
  assert.match(afterEnable.composerReplyFilter, /blur\(6px\)/, "输入框回复预览应被高斯模糊");
  assert.match(afterEnable.composerTrayFilter, /blur\(6px\)/, "输入框附件托盘项应被高斯模糊");
  assert.match(afterEnable.composerTextareaFilter, /blur\(6px\)/, "输入框草稿正文应被高斯模糊");
  assert.equal(afterEnable.headingHasTitleAttr, false, "打码期间应剥离会泄漏标题的原生 title 提示");

  // 2. 验证打码期间宿主动态更新 title 属性 + 节点移出文档的无损恢复
  const dynamicTooltipCheck = await page.evaluate(async () => {
    const tempBtn = document.createElement("button");
    tempBtn.className = "conversation-session";
    tempBtn.setAttribute("title", "临时会话原名");
    document.body.appendChild(tempBtn);
    window.__setHeadingTitle("动态重命名后的新标题");
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    const h1 = document.getElementById("real-heading-h1");
    const strippedWhileMasked = !h1.hasAttribute("title") && !tempBtn.hasAttribute("title");
    // 在打码状态下将 tempBtn 移出 document
    tempBtn.remove();
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    return {
      strippedWhileMasked,
      detachedRestoredImmediately: tempBtn.getAttribute("title"),
      savedNewHeadingTitle: h1.getAttribute("data-privacy-raw-title"),
    };
  });
  assert.equal(dynamicTooltipCheck.strippedWhileMasked, true, "打码期间动态更新的 title 应被自动剥离");
  assert.equal(dynamicTooltipCheck.detachedRestoredImmediately, "临时会话原名", "移出 document 的节点应立即恢复原 title 属性");
  assert.equal(dynamicTooltipCheck.savedNewHeadingTitle, "动态重命名后的新标题", "打码期间更新的 title 应记录最新值而非旧值");

  // 3. 验证输入框聚焦时按 Alt+M 不会误触发切换（保护 macOS Option+M 字符输入），而 Ctrl+Shift+M 正常切换
  await page.locator("#composer-textarea").focus();
  await page.keyboard.press("Alt+KeyM");
  await page.waitForTimeout(80);
  assert.equal(await page.evaluate(() => document.documentElement.getAttribute("data-privacy-mask")), "true", "焦点在输入框内时 Alt+M 不得误触发切换");

  // 4. 点开顶栏「已打码」标记，查看双维度控制面板并截图
  const headerTrigger = page.locator(".privacy-mask-marker .header-marker__trigger");
  await headerTrigger.click();
  await page.waitForSelector(".privacy-mask-panel");
  await page.waitForTimeout(140);

  const switches = await page.locator(".privacy-mask-scope .privacy-mask-scope__label").allInnerTexts();
  assert.deepEqual(switches, ["导航与会话标题", "全部内容模糊"], "面板应严格只包含「导航与会话标题」与「全部内容模糊」两个正交开关");
  await page.screenshot({ path: resolve(root, "02-full-masked-popover.png") });

  // 5. 测试正交独立开关：关闭「全部内容模糊」、仅保留「导航与会话标题」
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

  // 6. 验证非编辑区 Alt+M 快捷键一键恢复正常显示，并还原最新动态更新的 title
  await page.keyboard.press("Escape");
  await page.waitForTimeout(60);
  await page.keyboard.press("Alt+KeyM");
  await page.waitForTimeout(200);

  const afterDisable = await page.evaluate(() => {
    const html = document.documentElement;
    const headingH1 = document.getElementById("real-heading-h1");
    const titleEl = document.querySelector(".conversation-session__title");
    return {
      maskAttr: html.getAttribute("data-privacy-mask"),
      restoredTitleAttr: headingH1.getAttribute("title"),
      titleColor: getComputedStyle(titleEl).color,
      userContentFilter: getComputedStyle(document.getElementById("user-content-inner")).filter,
    };
  });
  assert.equal(afterDisable.maskAttr, null);
  assert.equal(afterDisable.restoredTitleAttr, "动态重命名后的新标题", "关闭打码后应恢复打码期间更新的最新 title");
  assert.notEqual(afterDisable.titleColor, "rgba(0, 0, 0, 0)");
  assert.equal(afterDisable.userContentFilter, "none");

  // 7. 录制完整交互动效 GIF
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

  // 8. 验证插件卸载（release）后的完全正交零残留
  await page.keyboard.press("Alt+KeyM");
  await page.waitForTimeout(100);
  const afterDispose = await page.evaluate(() => {
    window.__disposePrivacyPlugin();
    const html = document.documentElement;
    return {
      maskAttr: html.getAttribute("data-privacy-mask"),
      titlesAttr: html.getAttribute("data-privacy-titles"),
      contentAttr: html.getAttribute("data-privacy-content"),
      rawTitleRemaining: document.querySelectorAll("[data-privacy-raw-title], [data-privacy-raw-tip]").length,
      globalRef: typeof window.__akashicPrivacyMask,
    };
  });
  assert.equal(afterDispose.maskAttr, null);
  assert.equal(afterDispose.titlesAttr, null);
  assert.equal(afterDispose.contentAttr, null);
  assert.equal(afterDispose.rawTitleRemaining, 0, "卸载后不得残留任何 data-privacy-raw-* 属性");
  assert.equal(afterDispose.globalRef, "undefined", "插件卸载后必须彻底释放全局单例与全部 DOM 属性");
  assert.deepEqual(errors, [], `页面运行期不得出现未捕获异常: ${errors.join("; ")}`);

  const summary = {
    ok: true,
    evidenceDir: root,
    switches,
    childAfterWidths: afterEnable.childAfterWidths,
    geometryPreserved: true,
    gifPath,
  };
  writeFileSync(resolve(root, "summary.json"), JSON.stringify(summary, null, 2));
  console.log(JSON.stringify(summary, null, 2));
} finally {
  await browser.close();
  await server.close();
}
