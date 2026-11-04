import type { ConversationTabView } from "@akashic/conversation-tools-v2";
import { currentTheme, subscribeTheme, type WebHostContextV1, type WebUiDisposer } from "@akashic/web-ui-v1";
import { DESIGN_CHANNEL, type DesignFromFrame, type DesignToFrame } from "../editor/contract";
import "./style.css";

const FRAME = "/design/assets/editor.html";

// 编辑器（open-pencil、CanvasKit WASM、字体）是单独构建的大包，跑在同源 iframe 里：
// 宿主不允许 Web 模块动态加载代码，标签模块只负责第一次可见时建 iframe，转交会话、可见性与主题。
class DesignTab {
  private readonly root = document.createElement("div");
  private readonly stops: (() => void)[];
  private frame: HTMLIFrameElement | null = null;
  private ready = false;

  constructor(private readonly host: HTMLElement, private readonly view: ConversationTabView) {
    this.root.className = "design-tab";
    host.replaceChildren(this.root);
    window.addEventListener("message", this.onFrame);
    this.start();
    this.stops = [
      view.onActiveChange((active) => { this.start(); this.post({ type: "active", active }); }),
      view.onSessionChange((session) => this.post({ type: "session", session })),
      subscribeTheme(() => this.post({ type: "theme", dark: dark() })),
    ];
  }

  dispose(): void {
    for (const stop of this.stops) stop();
    window.removeEventListener("message", this.onFrame);
    this.host.replaceChildren();
  }

  // 第一次可见才建 iframe：没打开过设计标签的会话不加载编辑器
  private start(): void {
    if (this.frame || !this.view.active) return;
    this.frame = document.createElement("iframe");
    this.frame.className = "design-tab-frame";
    this.frame.title = "设计编辑器";
    this.frame.src = FRAME;
    this.root.replaceChildren(this.frame);
  }

  // iframe 就绪前的消息不投递：就绪时一次性补发完整的初始状态
  private post(message: DesignToFrame): void {
    if (this.ready) this.frame?.contentWindow?.postMessage({ channel: DESIGN_CHANNEL, ...message }, location.origin);
  }

  private readonly onFrame = (event: MessageEvent<DesignFromFrame & { channel?: string }>): void => {
    if (!this.frame || event.source !== this.frame.contentWindow || event.origin !== location.origin) return;
    const data = event.data;
    if (data?.channel !== DESIGN_CHANNEL) return;
    if (data.type !== "ready") return;
    this.ready = true;
    this.post({ type: "init", session: this.view.session, dark: dark(), active: this.view.active });
  };
}

const dark = () => currentTheme().colorScheme === "dark";

function renderDesign(host: HTMLElement, rawView: unknown): WebUiDisposer {
  const view = rawView as ConversationTabView;
  if (!view || typeof view.onSessionChange !== "function" || typeof view.onActiveChange !== "function")
    throw new Error("设计标签缺少 conversation.tools.v2 view");
  const tab = new DesignTab(host, view);
  return () => tab.dispose();
}

export function activate(ctx: WebHostContextV1): WebUiDisposer {
  return ctx.ui.inject("conversation.tools.v2", (mount) => mount.register({
    id: "design", label: "设计", order: 40, hint: "和 Agent 一起编辑的设计画布",
    icon: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15.707 21.293a1 1 0 0 1-1.414 0l-1.586-1.586a1 1 0 0 1 0-1.414l5.586-5.586a1 1 0 0 1 1.414 0l1.586 1.586a1 1 0 0 1 0 1.414z"/><path d="m18 13-1.375-6.874a1 1 0 0 0-.746-.776L3.235 2.028a1 1 0 0 0-1.207 1.207L5.35 15.879a1 1 0 0 0 .776.746L13 18"/><path d="m2.3 2.3 7.286 7.286"/><circle cx="11" cy="11" r="2"/></svg>',
    render(host, _entryView, props) { return renderDesign(host, props); },
  }));
}
