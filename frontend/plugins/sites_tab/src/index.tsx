import type { ConversationIntent, ConversationIntentStatus, ConversationTabView } from "@akashic/conversation-tools-v2";
import type { WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";
import "./style.css";

/** site.open.v1 的 payload：{ artifact_id }，指向 show_site 发布的一份快照附件。 */
const OPEN_INTENT = "site.open.v1";
const CHANNEL = "akashic.site.v1";
const FRAME = "/sites/frame.html";
const STATE_KEY = "akashic.sites.v1:";
const WIDTH_KEY = "akashic.sites.width";
const WIDTHS = [["390", "手机"], ["768", "平板"], ["auto", "自适应"]] as const;
type Width = (typeof WIDTHS)[number][0];
const ICONS = {
  reload: '<path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7"/>',
  external: '<path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>',
};

interface SiteDocument { title: string; path: string; published_at: string; html: string }
interface Entry { artifact: string; title: string; path: string; published: string }
interface SessionSites { sites: Entry[]; current: string | null }

function element<K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text = ""): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  return node;
}

function iconButton(label: string, icon: keyof typeof ICONS, onClick: () => void): HTMLButtonElement {
  const node = element("button", "sites-icon");
  node.type = "button";
  node.title = label;
  node.setAttribute("aria-label", label);
  node.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[icon]}</svg>`;
  node.addEventListener("click", onClick);
  return node;
}

// 浏览器存储只是便利：读写失败（隐私模式、被禁用）时退回内存里的状态。
function load<T>(storage: () => Storage, key: string, fallback: T): T {
  try { return JSON.parse(storage().getItem(key) ?? "null") ?? fallback; } catch { return fallback; }
}

function save(storage: () => Storage, key: string, value: unknown): void {
  try { storage().setItem(key, JSON.stringify(value)); } catch { /* 见 load */ }
}

function versionLabel(iso: string): string {
  const time = new Date(iso);
  if (Number.isNaN(time.getTime())) return "";
  const today = time.toDateString() === new Date().toDateString();
  return `版本 ${time.toLocaleString("zh-CN", today ? { hour: "2-digit", minute: "2-digit", second: "2-digit" }
    : { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" })}`;
}

async function readSite(artifact: string): Promise<SiteDocument> {
  const response = await fetch(`/api/chat/artifacts/${encodeURIComponent(artifact)}`);
  if (!response.ok) throw new Error(`读取快照失败（${response.status}）`);
  const value = await response.json() as Partial<SiteDocument> & { schema?: unknown };
  if (value.schema !== CHANNEL || typeof value.html !== "string" || typeof value.title !== "string"
    || typeof value.path !== "string" || typeof value.published_at !== "string") throw new Error("不是网站快照");
  return value as SiteDocument;
}

// 引导页回报就绪后交付一次文档；之后来自页面的任何消息都不理会。
function deliver(target: () => Window | null, html: string): () => void {
  const receive = (event: MessageEvent) => {
    const source = target();
    if (!source || event.source !== source || event.data?.channel !== CHANNEL || event.data.type !== "ready") return;
    window.removeEventListener("message", receive);
    source.postMessage({ channel: CHANNEL, type: "init", document: html }, "*");
  };
  window.addEventListener("message", receive);
  return () => window.removeEventListener("message", receive);
}

class SiteTab {
  readonly root = element("div", "sites-root");
  private readonly bar = element("div", "sites-bar");
  private readonly stage = element("div", "sites-stage");
  private readonly meta = element("span", "sites-meta");
  private state: SessionSites = { sites: [], current: null };
  private document: SiteDocument | null = null;
  private width: Width = load(() => window.localStorage, WIDTH_KEY, "auto");
  private stopDelivery = () => {};
  private ticket = 0;

  constructor(private session: string | null) {
    this.root.append(this.bar, this.stage);
    this.setSession(session, true);
  }

  // 每个会话各记发布过的网页（按路径）与当前那一个；回到会话或刷新后接着显示。
  setSession(session: string | null, force = false): void {
    if (session === this.session && !force) return;
    this.session = session;
    this.state = session ? load(() => window.sessionStorage, STATE_KEY + session, { sites: [], current: null }) : { sites: [], current: null };
    const entry = this.state.sites.find((site) => site.path === this.state.current);
    if (entry) void this.show(entry);
    else this.empty();
  }

  // 同一路径再次发布时替换这一项，标签与宽度不变；历史卡片打开的是当时的那份快照。
  async open(intent: ConversationIntent): Promise<ConversationIntentStatus> {
    const artifact = (intent.payload as { artifact_id?: unknown } | null)?.artifact_id;
    if (typeof artifact !== "string" || !/^[0-9a-f]{8,64}$/u.test(artifact)) return "invalid";
    this.session = intent.session;
    const ticket = ++this.ticket;
    this.message("正在打开…");
    try {
      const site = await readSite(artifact);
      if (ticket !== this.ticket) return "handled";
      const entry = { artifact, title: site.title, path: site.path, published: site.published_at };
      this.state = { sites: [entry, ...this.state.sites.filter((item) => item.path !== site.path)].slice(0, 20), current: site.path };
      save(() => window.sessionStorage, STATE_KEY + intent.session, this.state);
      this.render(entry, site);
      return "handled";
    } catch (error) {
      if (ticket === this.ticket) this.message(error instanceof Error ? error.message : String(error), true);
      return "handled";
    }
  }

  dispose(): void {
    this.ticket += 1;
    this.stopDelivery();
  }

  private async show(entry: Entry): Promise<void> {
    const ticket = ++this.ticket;
    this.message("正在打开…");
    try {
      const site = await readSite(entry.artifact);
      if (ticket === this.ticket) this.render(entry, site);
    } catch (error) {
      if (ticket === this.ticket) this.message(error instanceof Error ? error.message : String(error), true);
    }
  }

  private render(entry: Entry, site: SiteDocument): void {
    this.document = site;
    this.paintBar(entry);
    this.mountFrame();
  }

  // 工具条：网页（本会话发布过多个时可切换）、版本时间、宽度、重新加载、在新标签页打开。
  private paintBar(entry: Entry): void {
    const name = element("div", "sites-name");
    if (this.state.sites.length > 1) {
      const select = element("select", "sites-select");
      select.setAttribute("aria-label", "网页");
      for (const site of this.state.sites) select.append(new Option(site.title, site.path, false, site.path === entry.path));
      select.addEventListener("change", () => {
        const next = this.state.sites.find((site) => site.path === select.value);
        if (!next) return;
        this.state.current = next.path;
        if (this.session) save(() => window.sessionStorage, STATE_KEY + this.session, this.state);
        void this.show(next);
      });
      name.append(select);
    } else name.append(element("strong", "sites-title", entry.title));
    name.title = entry.path;
    const meta = this.meta;
    meta.textContent = versionLabel(entry.published);
    const widths = element("div", "sites-widths");
    widths.setAttribute("role", "group");
    widths.setAttribute("aria-label", "预览宽度");
    for (const [value, label] of WIDTHS) {
      const option = element("button", "sites-width", label);
      option.type = "button";
      option.dataset.width = value;
      option.setAttribute("aria-pressed", String(this.width === value));
      option.addEventListener("click", () => this.setWidth(value));
      widths.append(option);
    }
    this.bar.replaceChildren(name, meta, widths, iconButton("重新加载", "reload", () => this.mountFrame()),
      iconButton("在新浏览器标签页打开", "external", () => this.openWindow()));
  }

  private setWidth(width: Width): void {
    this.width = width;
    save(() => window.localStorage, WIDTH_KEY, width);
    for (const option of this.bar.querySelectorAll<HTMLElement>(".sites-width"))
      option.setAttribute("aria-pressed", String(option.dataset.width === width));
    this.stage.dataset.width = width;
  }

  // 每次重新加载都换一个 iframe：页面状态从头开始，旧 frame 的消息不会再被接收。
  private mountFrame(): void {
    const site = this.document;
    if (!site) return;
    this.stopDelivery();
    const frame = element("iframe", "sites-frame");
    frame.sandbox.add("allow-scripts", "allow-forms", "allow-modals", "allow-popups");
    frame.referrerPolicy = "no-referrer";
    frame.title = site.title;
    this.stopDelivery = deliver(() => frame.contentWindow, site.html);
    frame.src = FRAME;
    const device = element("div", "sites-device");
    device.append(frame);
    this.stage.dataset.width = this.width;
    this.stage.replaceChildren(device);
  }

  // 新标签页同样打开隔离引导页，再经 postMessage 交付文档；不用 blob（它与 akashic 同源）。
  private openWindow(): void {
    const site = this.document;
    if (!site) return;
    const opened = window.open(FRAME, "_blank");
    if (!opened) {
      this.meta.textContent = "浏览器拦截了新标签页";
      return;
    }
    const stop = deliver(() => opened, site.html);
    window.setTimeout(stop, 15_000);
  }

  private empty(): void {
    this.document = null;
    this.bar.replaceChildren();
    this.message("Agent 用 show_site 发布网页后会在这里打开；也可以点对话里的网站卡片。");
  }

  private message(text: string, failed = false): void {
    const node = element("p", failed ? "sites-message is-error" : "sites-message", text);
    node.setAttribute("role", failed ? "alert" : "status");
    if (!this.document || failed) this.stage.replaceChildren(node);
  }
}

function renderSites(host: HTMLElement, rawView: unknown): WebUiDisposer {
  const view = rawView as ConversationTabView;
  if (!view || typeof view.onSessionChange !== "function" || typeof view.onIntent !== "function")
    throw new Error("网站标签缺少 conversation.tools.v2 view");
  const tab = new SiteTab(view.session);
  host.replaceChildren(tab.root);
  const stops = [
    view.onSessionChange((session) => tab.setSession(session)),
    view.onIntent((intent) => intent.name === OPEN_INTENT ? tab.open(intent) : "invalid"),
  ];
  return () => {
    for (const stop of stops) stop();
    tab.dispose();
    host.replaceChildren();
  };
}

export function activate(ctx: WebHostContextV1): WebUiDisposer {
  return ctx.ui.inject("conversation.tools.v2", (mount) => mount.register({
    id: "sites", label: "网站", order: 40, hint: "Agent 用 show_site 发布的网页", accepts: [OPEN_INTENT],
    icon: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.5 2.5 3.8 5.5 3.8 9s-1.3 6.5-3.8 9c-2.5-2.5-3.8-5.5-3.8-9S9.5 5.5 12 3Z"/></svg>',
    render(host, _entryView, props) { return renderSites(host, props); },
  }));
}
