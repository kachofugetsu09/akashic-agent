import type { ConversationTabView } from "@akashic/conversation-tools-v2";
import { currentTheme, subscribeTheme, type WebHostContextV1, type WebUiDisposer } from "@akashic/web-ui-v1";
import { DESIGN_API, DESIGN_CHANNEL, type DesignFromFrame, type DesignToFrame, type DesignView, type RelayedRequest } from "../editor/contract";
import "./style.css";

const FRAME = "/design/assets/editor.html";
const RETRY_MS = 3000;

// 代发编辑器的文档请求：只放行本插件的 API，响应体整块转移回 iframe。
async function relay(ctx: WebHostContextV1, request: RelayedRequest): Promise<DesignToFrame> {
  try {
    if (!request.path.startsWith(DESIGN_API)) throw new Error("设计编辑器只能请求自己的接口");
    const response = await ctx.http.request(request.path, {
      method: request.method, headers: request.headers, body: request.body,
    });
    return { type: "response", id: request.id, status: response.status,
      headers: [...response.headers.entries()], body: await response.arrayBuffer() };
  } catch (error) {
    return { type: "response", id: request.id, error: error instanceof Error ? error.message : String(error) };
  }
}

type LiveMessage = { type: "agent"; turn: string; document: string; view: DesignView }
  | { type: "changed"; document: string; revision: number } | { type: "resync" };

// 为当前会话保持一条到后端的实时连接：Agent 正在改的文档与别处保存的新修订从这里进来；断开后自动重连。
class LiveLink {
  private socket: WebSocket | null = null;
  private timer = 0;
  private session: string | null = null;

  constructor(private readonly ctx: WebHostContextV1, private readonly receive: (message: LiveMessage) => void) {}

  open(session: string | null): void {
    this.close();
    this.session = session;
    if (session) this.connect();
  }

  close(): void {
    window.clearTimeout(this.timer);
    const socket = this.socket;
    this.socket = null;
    socket?.close();
  }

  private connect(): void {
    const session = this.session;
    if (!session) return;
    let socket: WebSocket;
    try {
      socket = new WebSocket(this.ctx.http.webSocketUrl(`${DESIGN_API}live?${new URLSearchParams({ session })}`));
    } catch {
      return;
    }
    this.socket = socket;
    socket.addEventListener("open", () => {
      if (this.socket !== socket) return;
      this.receive({ type: "resync" });
    });
    // 换会话后旧连接可能还有在路上的消息：只收当前这条连接的
    socket.addEventListener("message", (event) => {
      if (this.socket !== socket) return;
      try { this.receive(JSON.parse(String(event.data)) as LiveMessage); } catch { /* 忽略坏消息，连接保持 */ }
    });
    socket.addEventListener("close", () => {
      if (this.socket !== socket) return;
      this.socket = null;
      this.timer = window.setTimeout(() => this.connect(), RETRY_MS);
    });
  }
}

// 编辑器（open-pencil、CanvasKit WASM、字体）是单独构建的大包，跑在同源 iframe 里：
// 宿主不允许 Web 模块动态加载代码，标签模块只负责第一次可见时建 iframe，转交会话、可见性与主题，并代发文档请求。
class DesignTab {
  private readonly root = document.createElement("div");
  private readonly live: LiveLink;
  private readonly stops: (() => void)[];
  private frame: HTMLIFrameElement | null = null;
  private ready = false;
  private queued: DesignToFrame[] = [];

  constructor(private readonly ctx: WebHostContextV1, private readonly host: HTMLElement, private readonly view: ConversationTabView) {
    this.root.className = "design-tab";
    host.replaceChildren(this.root);
    this.live = new LiveLink(ctx, (message) => this.onLive(message));
    window.addEventListener("message", this.onFrame);
    this.start();
    this.live.open(view.session);
    this.stops = [
      view.onActiveChange((active) => {
        this.start();
        this.post({ type: "active", active });
      }),
      view.onSessionChange((session) => { this.queued = []; this.post({ type: "session", session }); this.live.open(session); }),
      subscribeTheme(() => this.post({ type: "theme", dark: dark() })),
    ];
  }

  dispose(): void {
    for (const stop of this.stops) stop();
    this.live.close();
    window.removeEventListener("message", this.onFrame);
    this.host.replaceChildren();
  }

  // 第一次可见才建 iframe：Agent 在服务端改文档，用户不看时不必加载编辑器
  private start(): void {
    if (this.frame || !this.view.active) return;
    this.frame = document.createElement("iframe");
    this.frame.className = "design-tab-frame";
    this.frame.title = "设计编辑器";
    this.frame.src = FRAME;
    this.root.replaceChildren(this.frame);
  }

  // iframe 就绪前只留下 Agent 的文档切换与文档变化，就绪后按顺序补投
  private post(message: DesignToFrame, transfer: Transferable[] = []): void {
    if (!this.ready) {
      if (this.frame && (message.type === "agent" || message.type === "changed")) this.queued.push(message);
      return;
    }
    this.frame?.contentWindow?.postMessage({ channel: DESIGN_CHANNEL, ...message }, location.origin, transfer);
  }

  // Agent 开始改文档时桌面端顺带展开标签（每轮一次），画布切到那份文档
  private onLive(message: LiveMessage): void {
    if (message.type !== "agent") return this.post(message);
    this.view.reveal(`design:${message.turn}`);
    this.post({ type: "agent", document: message.document, view: message.view });
  }

  private readonly onFrame = (event: MessageEvent<DesignFromFrame & { channel?: string }>): void => {
    if (!this.frame || event.source !== this.frame.contentWindow || event.origin !== location.origin) return;
    const data = event.data;
    if (data?.channel !== DESIGN_CHANNEL) return;
    if (data.type === "ready") {
      this.ready = true;
      this.post({ type: "init", session: this.view.session, dark: dark(), active: this.view.active });
      const pending = this.queued;
      this.queued = [];
      for (const message of pending) this.post(message);
    } else if (data.type === "title") this.view.setTitle(data.title);
    else if (data.type === "indicator") this.view.setIndicator(data.indicator);
    else if (data.type === "request") void relay(this.ctx, data).then((reply) => this.post(reply, "body" in reply ? [reply.body] : []));
  };
}

const dark = () => currentTheme().colorScheme === "dark";

function renderDesign(ctx: WebHostContextV1, host: HTMLElement, rawView: unknown): WebUiDisposer {
  const view = rawView as ConversationTabView;
  if (!view || typeof view.onSessionChange !== "function" || typeof view.onActiveChange !== "function")
    throw new Error("设计标签缺少 conversation.tools.v2 view");
  const tab = new DesignTab(ctx, host, view);
  return () => tab.dispose();
}

export function activate(ctx: WebHostContextV1): WebUiDisposer {
  return ctx.ui.inject("conversation.tools.v2", (mount) => mount.register({
    id: "design", label: "设计", order: 40, hint: "和 Agent 一起编辑的设计画布", background: true,
    icon: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15.707 21.293a1 1 0 0 1-1.414 0l-1.586-1.586a1 1 0 0 1 0-1.414l5.586-5.586a1 1 0 0 1 1.414 0l1.586 1.586a1 1 0 0 1 0 1.414z"/><path d="m18 13-1.375-6.874a1 1 0 0 0-.746-.776L3.235 2.028a1 1 0 0 0-1.207 1.207L5.35 15.879a1 1 0 0 0 .776.746L13 18"/><path d="m2.3 2.3 7.286 7.286"/><circle cx="11" cy="11" r="2"/></svg>',
    render(host, _entryView, props) { return renderDesign(ctx, host, props); },
  }));
}
